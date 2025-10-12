from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import math, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold

from src.config import (
    SAMPLE_RATE, N_CLASSES, MODEL, GRIDSEARCH, SUPERCLASSES
)
from src.data_loader import compute_perlead_norm_stats, load_waveform, normalize_signal
from src.models import create_model


# ----------------------------- data utils -------------------------------
def _tensor_dataset_from_df(df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-6):
    X, y = [], []
    cls2idx = {c: i for i, c in enumerate(range(N_CLASSES))}
    # df must already have 'y' as class index; if not, cast via provided mapping.
    for _, row in df.iterrows():
        sig = load_waveform(row, sampling_rate=SAMPLE_RATE)  # (12, T)
        sig = normalize_signal(sig, mu, sigma, eps=eps)
        X.append(sig)
        y.append(SUPERCLASSES.index(row["y"]))
    X = torch.tensor(np.stack(X), dtype=torch.float32) if len(X) else torch.zeros((0, 12, 1))
    y = torch.tensor(np.array(y), dtype=torch.long) if len(y) else torch.zeros((0,), dtype=torch.long)
    return torch.utils.data.TensorDataset(X, y)

def _make_loader(df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, batch_size: int, shuffle: bool):
    ds = _tensor_dataset_from_df(df, mu, sigma)
    return torch.utils.data.DataLoader(ds, batch_size=int(batch_size), shuffle=shuffle)


# --------------------------- model / training ---------------------------
def _model_ctor():
    return create_model(
        n_classes=N_CLASSES,
        model_type=MODEL["type"],
        hidden=MODEL.get("lstm_hidden", 128),
        layers=MODEL.get("lstm_layers", 1),
        bidir=MODEL.get("bidirectional", True),
    )

def _train_one_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, hp: Dict, device: torch.device) -> Tuple[float, float]:
    """Return (val_loss, val_acc). Uses FedProx μ if provided in hp['fedprox']."""
    mu_tr, sigma_tr = compute_perlead_norm_stats(train_df, sampling_rate=SAMPLE_RATE)
    tr = _make_loader(train_df, mu_tr, sigma_tr, batch_size=int(hp["batch"]), shuffle=True)
    va = _make_loader(val_df,   mu_tr, sigma_tr, batch_size=max(64, int(hp["batch"])), shuffle=False)

    model = _model_ctor().to(device)
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=float(hp["lr"]), weight_decay=1e-4)

    # FedProx proximal term to initial weights
    w0 = [p.detach().clone() for p in model.parameters()]
    mu_prox = float(hp.get("fedprox", 0.0))

    model.train()
    for _ in range(int(hp["epochs"])):
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            if mu_prox > 0.0:
                prox = sum(torch.sum((w - w_init) ** 2) for w, w_init in zip(model.parameters(), w0))
                loss = loss + (mu_prox / 2.0) * prox
            loss.backward()
            opt.step()

    # eval
    model.eval()
    tot, cor, lsum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in va:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            lsum += float(ce(logits, yb).item()) * len(yb)
            cor  += int((logits.argmax(1) == yb).sum())
            tot  += len(yb)

    vloss = (lsum / tot) if tot else math.inf
    vacc  = (cor / tot) if tot else 0.0
    return vloss, vacc


# ------------------------------- API ------------------------------------
def default_grid() -> List[Dict]:
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
    Patient-aware GridSearchCV on a client's train_df.
    Returns: {"best": hp_dict, "best_mean_acc": float}
    Optionally writes a CSV of trials and a JSON with best params.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if grid is None:
        grid = default_grid()
    if k is None:
        k = int(GRIDSEARCH.get("cv", 5))

    # --- GroupKFold by patient_id to avoid leakage ----------------
    idx = np.arange(len(df_train))
    groups = df_train["patient_id"].values
    labels = df_train["y"].values

    # Guard: don't request more folds than unique patients (unlikely)
    unique_groups = int(np.unique(groups).size)
    eff_k = min(int(k), unique_groups)
    print(f"[CV] rows={len(df_train)}, unique_patients={unique_groups}, requested_k={k}, effective_k={eff_k}")
    if eff_k < 2:
        return {"best": None, "best_mean_acc": 0.0}

    # Grouped by patient_id to avoid leakage
    gkf = GroupKFold(n_splits=eff_k)


    # --- Loop over hyperparameter grid ---------------------------
    rows, best_score, best_hp = [], -1.0, None
    for trial, hp in enumerate(grid, start=1):
        fold_losses, fold_accs = [], []
        for tr_idx, va_idx in gkf.split(idx, groups=groups, y=labels):
            tr_df = df_train.iloc[tr_idx]; va_df = df_train.iloc[va_idx]
            l, a = _train_one_fold(tr_df, va_df, hp, device)
            fold_losses.append(l); fold_accs.append(a)
        mean_loss = float(np.mean(fold_losses)) if fold_losses else float("inf")
        mean_acc  = float(np.mean(fold_accs))  if fold_accs  else 0.0
        rows.append({"trial": trial, **hp, "mean_val_loss": mean_loss, "mean_val_acc": mean_acc})
        if mean_acc > best_score:
            best_score, best_hp = mean_acc, hp

    if save_csv is not None:
        df = pd.DataFrame(rows)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv, index=False)

    if save_json is not None:
        save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w") as f:
            json.dump({"best_mean_val_acc": best_score, "params": best_hp}, f, indent=2)

    return {"best": best_hp, "best_mean_acc": best_score}