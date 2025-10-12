from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import math
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold

from src.config import SAMPLE_RATE, N_CLASSES, MODEL, GRIDSEARCH, SUPERCLASSES
from src.data_loader import compute_perlead_norm_stats, load_waveform, normalize_signal
from src.models import create_model


# -------------------------------------------------------------------------
# Data utilities
# -------------------------------------------------------------------------
def _tensor_dataset_from_df(
    df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-6
) -> torch.utils.data.TensorDataset:
    """
    Build a TensorDataset (X, y) from a PTB-XL subset DataFrame.
    Uses per-lead z-score normalization with stats computed on the training fold.
    """
    X, y = [], []
    for _, row in df.iterrows():
        sig = load_waveform(row, sampling_rate=SAMPLE_RATE)  # (12, T)
        sig = normalize_signal(sig, mu, sigma, eps=eps)
        X.append(sig)
        y.append(SUPERCLASSES.index(row["y"]))
    if not X:
        X_t = torch.zeros((0, 12, 1), dtype=torch.float32)
        y_t = torch.zeros((0,), dtype=torch.long)
    else:
        X_t = torch.tensor(np.stack(X), dtype=torch.float32)  # (N, 12, T)
        y_t = torch.tensor(np.array(y), dtype=torch.long)     # (N,)
    return torch.utils.data.TensorDataset(X_t, y_t)


def _make_loader(
    df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, batch_size: int, shuffle: bool
) -> torch.utils.data.DataLoader:
    ds = _tensor_dataset_from_df(df, mu, sigma)
    return torch.utils.data.DataLoader(ds, batch_size=int(batch_size), shuffle=shuffle)


# -------------------------------------------------------------------------
# Model / training helpers
# -------------------------------------------------------------------------
def _model_ctor() -> nn.Module:
    """Factory wrapper using current MODEL config."""
    return create_model(
        n_classes=N_CLASSES,
        model_type=MODEL.get("type", "cnn"),
        hidden=MODEL.get("lstm_hidden", 128),
        layers=MODEL.get("lstm_layers", 1),
        bidir=MODEL.get("bidirectional", True),
    )


def _train_one_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    hp: Dict,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train on one fold and return (val_loss, val_acc).
    Includes optional FedProx proximal term to initial weights.
    """
    # Per-lead normalization stats from TRAIN fold only (leakage-safe)
    mu_tr, sigma_tr = compute_perlead_norm_stats(train_df, sampling_rate=SAMPLE_RATE)

    dl_tr = _make_loader(train_df, mu_tr, sigma_tr, batch_size=int(hp["batch"]), shuffle=True)
    dl_va = _make_loader(val_df,   mu_tr, sigma_tr, batch_size=max(64, int(hp["batch"])), shuffle=False)

    model = _model_ctor().to(device)
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=float(hp["lr"]), weight_decay=1e-4)

    # FedProx: proximal term to initial parameters (if provided)
    w0 = [p.detach().clone() for p in model.parameters()]
    mu_prox = float(hp.get("fedprox", 0.0))

    model.train()
    for _ in range(int(hp["epochs"])):
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            if mu_prox > 0.0:
                prox = sum(torch.sum((w - w_init) ** 2) for w, w_init in zip(model.parameters(), w0))
                loss = loss + (mu_prox / 2.0) * prox
            loss.backward()
            opt.step()

    # Validation
    model.eval()
    tot, cor, lsum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in dl_va:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            lsum += float(ce(logits, yb).item()) * len(yb)
            cor  += int((logits.argmax(1) == yb).sum())
            tot  += len(yb)

    vloss = (lsum / tot) if tot else math.inf
    vacc  = (cor / tot) if tot else 0.0
    return vloss, vacc


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------
def default_grid() -> List[Dict]:
    """Fallback grid (used if GRIDSEARCH.grid is empty/missing)."""
    return GRIDSEARCH.get("grid") or [
        {"lr": 1e-4, "batch": 32,  "epochs": 1, "fedprox": 0.0},
        {"lr": 5e-4, "batch": 64,  "epochs": 2, "fedprox": 0.0},
        {"lr": 1e-3, "batch": 64,  "epochs": 2, "fedprox": 0.001},
    ]


def run_client_cv(
    df_train: pd.DataFrame,
    k: Optional[int] = None,
    grid: Optional[List[Dict]] = None,
    device: Optional[torch.device] = None,
    save_csv: Optional[Path] = None,
    save_json: Optional[Path] = None,
) -> Dict:
    """
    Patient-aware GridSearchCV on a client's training subset.

    Args:
        df_train: DataFrame with at least ['patient_id', 'y', ...].
        k: Number of GroupKFold splits (defaults to config.GRIDSEARCH['cv']).
        grid: List of hyperparameter dicts (lr, batch, epochs, fedprox).
        device: Torch device; defaults to CUDA if available, else CPU.
        save_csv: Optional path to write a CSV of all trials/fold metrics.
        save_json: Optional path to write best params and score as JSON.

    Returns:
        {"best": <hp_dict or None>, "best_mean_acc": <float>}
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = grid or default_grid()
    k = int(k if k is not None else GRIDSEARCH.get("cv", 5))

    # Prepare GroupKFold by patient_id to avoid leakage
    assert "patient_id" in df_train.columns, "df_train must include 'patient_id'"
    assert "y" in df_train.columns, "df_train must include 'y' (superclass label)"
    idx = np.arange(len(df_train))
    groups = df_train["patient_id"].values
    labels = df_train["y"].values  # not used by GroupKFold directly, but helpful for debugging

    unique_groups = int(np.unique(groups).size)
    eff_k = min(int(k), unique_groups)
    print(f"[CV] rows={len(df_train)}, unique_patients={unique_groups}, requested_k={k}, effective_k={eff_k}")
    if eff_k < 2:
        # Not enough distinct patients to split
        if save_json is not None:
            save_json.parent.mkdir(parents=True, exist_ok=True)
            with open(save_json, "w", encoding="utf-8") as f:
                json.dump({"best_mean_val_acc": 0.0, "params": None}, f, indent=2)
        return {"best": None, "best_mean_acc": 0.0}

    gkf = GroupKFold(n_splits=eff_k)

    # Grid search over folds
    rows: List[Dict] = []
    best_score, best_hp = -1.0, None
    for trial, hp in enumerate(grid, start=1):
        fold_losses, fold_accs = [], []
        for tr_idx, va_idx in gkf.split(idx, groups=groups, y=labels):
            tr_df = df_train.iloc[tr_idx]
            va_df = df_train.iloc[va_idx]
            vloss, vacc = _train_one_fold(tr_df, va_df, hp, device)
            fold_losses.append(vloss)
            fold_accs.append(vacc)
        mean_loss = float(np.mean(fold_losses)) if fold_losses else float("inf")
        mean_acc  = float(np.mean(fold_accs))  if fold_accs  else 0.0
        rows.append({"trial": trial, **hp, "mean_val_loss": mean_loss, "mean_val_acc": mean_acc})
        if mean_acc > best_score:
            best_score, best_hp = mean_acc, hp

    # Persist results
    if save_csv is not None:
        df = pd.DataFrame(rows)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)

    if save_json is not None:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump({"best_mean_val_acc": best_score, "params": best_hp}, f, indent=2)

    return {"best": best_hp, "best_mean_acc": best_score}
