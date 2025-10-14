# src/tuning.py
"""
tuning.py — Client-side Hyperparameter Tuning (Cross-Validation)
-----------------------------------------------------------------

Purpose
-------
Lightweight patient-aware hyperparameter search for a *single* federated
client's local dataset. Uses GroupKFold on `patient_id` to avoid leakage
between train/validation folds.

What this module connects to
----------------------------
• Called by: src.Client
    - `Client.PTBClient.__init__` optionally imports and calls
      `run_client_cv(...)` to perform per-client CV and cache the best
      hyperparameters before federated training starts.

• Uses: src.models
    - `create_model(...)` to build the CNN/LSTM/RNN/ANN specified by config.MODEL.

• Uses: src.data_loader
    - `load_waveform(...)`, `compute_perlead_norm_stats(...)`, and
      `normalize_signal(...)` to prepare normalized per-lead tensors.

• Uses: src.config
    - Reads global knobs like SAMPLE_RATE, N_CLASSES, MODEL, GRIDSEARCH,
      SUPERCLASSES, and SEED.

• Uses: src.utils
    - `ensure_dir(...)` for robust I/O and `set_seed(...)` for reproducibility.

Outputs / Side effects
----------------------
• Returns a dict with {"best": <hp dict or None>, "best_mean_acc": <float>}
• Optionally writes:
    - CSV of all CV trials/folds (save_csv)
    - JSON of best params and score (save_json)
"""

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

from src.config import (
    SAMPLE_RATE, N_CLASSES, MODEL, GRIDSEARCH, SUPERCLASSES, SEED
)
from src.data_loader import (
    compute_perlead_norm_stats, load_waveform, normalize_signal
)
from src.models import create_model
from src.utils import ensure_dir, set_seed

# Optional grad clip from config (if defined)
try:
    from src.config import GRAD_CLIP_NORM as _GRAD_CLIP
except Exception:
    _GRAD_CLIP = 0.0


# -------------------------------------------------------------------------
# Data utilities
# -------------------------------------------------------------------------
def _tensor_dataset_from_df(
    df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-6
) -> torch.utils.data.TensorDataset:
    """
    Convert a PTB-XL subset DataFrame into a normalized TensorDataset.

    What it does
    ------------
    • Iterates rows of `df`, loads each waveform (12, T) via data_loader.load_waveform
    • Applies per-lead z-score normalization using (mu, sigma) computed on the
      *training fold* only — prevents information leakage
    • Encodes class labels as integer indices using config.SUPERCLASSES

    Returns
    -------
    torch.utils.data.TensorDataset
        X: float32 tensor of shape (N, 12, T)
        y: int64 tensor of shape (N,)

    Where used
    ----------
    • `_make_loader(...)` below to produce DataLoaders for train/val
    • Ultimately consumed inside `_train_one_fold(...)` during CV
    """
    X, y = [], []
    for _, row in df.iterrows():
        try:
            sig = load_waveform(row, sampling_rate=SAMPLE_RATE)  # (12, T)
        except Exception:
            # Robustness: fall back to a minimal zero signal if read fails
            sig = np.zeros((12, 1), dtype="float32")
        # z-score per lead using fold-specific stats (leakage-safe)
        sig = normalize_signal(sig, mu, sigma, eps=eps) if mu is not None and sigma is not None else sig
        X.append(sig.astype("float32", copy=False))
        y.append(SUPERCLASSES.index(row["y"]))
    if not X:
        # Edge case: empty split
        X_t = torch.zeros((0, 12, 1), dtype=torch.float32)
        y_t = torch.zeros((0,), dtype=torch.long)
    else:
        X_t = torch.tensor(np.stack(X), dtype=torch.float32)  # (N, 12, T)
        y_t = torch.tensor(np.array(y), dtype=torch.long)     # (N,)
    return torch.utils.data.TensorDataset(X_t, y_t)


def _make_loader(
    df: pd.DataFrame, mu: np.ndarray, sigma: np.ndarray, batch_size: int, shuffle: bool
) -> torch.utils.data.DataLoader:
    """
    Wrap `_tensor_dataset_from_df` into a simple DataLoader.

    Notes
    -----
    • Keeps DataLoader defaults lightweight (num_workers=0) for cross-platform stability.
    • Validation loader uses a larger/equal batch and no shuffling.

    Connected to
    ------------
    • `_train_one_fold(...)` for both training and validation loaders.
    """
    ds = _tensor_dataset_from_df(df, mu, sigma)
    batch = max(1, int(batch_size))
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=bool(shuffle))


# -------------------------------------------------------------------------
# Model / training helpers
# -------------------------------------------------------------------------
def _model_ctor() -> nn.Module:
    """
    Factory for the model specified in config.MODEL.

    Reads
    -----
    • MODEL['type'] in {'cnn','lstm','rnn','ann'}
    • MODEL['lstm_hidden'], MODEL['lstm_layers'], MODEL['bidirectional']

    Returns
    -------
    torch.nn.Module
        A model compatible with inputs shaped (N, 12, T) or (N, T, 12).

    Where used
    ----------
    • `_train_one_fold(...)`
    """
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
    Train on one GroupKFold split and evaluate on its validation fold.

    Pipeline
    --------
    1) Compute per-lead normalization stats (mu, sigma) on *train_df* only.
    2) Create loaders via `_make_loader` (train shuffled, val non-shuffled).
    3) Build model from `_model_ctor` and optimize with Adam.
    4) Optional FedProx proximal term (mu_prox) to initial weights snapshot.
    5) Validate with plain CrossEntropyLoss (unweighted).

    Parameters
    ----------
    train_df, val_df : pd.DataFrame
        Subsets for a single fold; must contain 'y' labels aligned with SUPERCLASSES.
    hp : dict
        Hyperparameters for this trial: {'lr','batch','epochs','fedprox'}.
    device : torch.device
        'cuda' if available else 'cpu'.

    Returns
    -------
    (val_loss, val_acc) : Tuple[float, float]

    Where used
    ----------
    • `run_client_cv(...)` inner loop to score each (trial, fold).
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

    # --- Train for hp['epochs'] epochs on this fold ---
    model.train()
    for _ in range(int(hp["epochs"])):
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce(logits, yb)
            if mu_prox > 0.0:
                # Proximal (FedProx): penalize drift from initial weights
                prox = sum(torch.sum((w - w_init) ** 2) for w, w_init in zip(model.parameters(), w0))
                loss = loss + (mu_prox / 2.0) * prox
            loss.backward()
            if _GRAD_CLIP and float(_GRAD_CLIP) > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(_GRAD_CLIP))
            opt.step()

    # --- Validation (unweighted CE) ---
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
    """
    Return the default HP grid when config.GRIDSEARCH['grid'] is empty.

    Shape of each grid element
    --------------------------
    {'lr': float, 'batch': int, 'epochs': int, 'fedprox': float}

    Where used
    ----------
    • `run_client_cv(..., grid=None)` will call this to populate the grid.
    """
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

    Why GroupKFold?
    ---------------
    • Uses `patient_id` as the grouping key so the same patient never appears
      in both train and validation splits — this avoids label leakage.

    Inputs
    ------
    df_train : DataFrame
        Must include at least ['patient_id', 'y', ...].
        (Same schema as produced in Client.PTBClient for the client's shard.)
    k : int, optional
        Number of GroupKFold splits (defaults to config.GRIDSEARCH['cv']).
    grid : list of dict, optional
        Each dict contains {'lr','batch','epochs','fedprox'}.
        If None, falls back to `default_grid()`.
    device : torch.device, optional
        Torch device to use (defaults to CUDA if available).
    save_csv : Path, optional
        If provided, writes a CSV with all trials/folds metrics.
    save_json : Path, optional
        If provided, writes {"best_mean_acc": float, "params": dict|None}.

    Returns
    -------
    dict
        {"best": <best_hp or None>, "best_mean_acc": <float>}

    Where called from
    -----------------
    • `src.Client.PTBClient.__init__` (when TUNING.enabled=True):
        - If cached results exist and reuse is on, they are loaded first.
        - Otherwise, runs CV here and persists the best config.
    """
    # Reproducibility for the entire CV routine
    set_seed(SEED)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = grid or default_grid()
    k = int(k if k is not None else GRIDSEARCH.get("cv", 5))

    # Prepare GroupKFold by patient_id to avoid leakage
    assert "patient_id" in df_train.columns, "df_train must include 'patient_id'"
    assert "y" in df_train.columns, "df_train must include 'y' (superclass label)"
    idx = np.arange(len(df_train))
    groups = df_train["patient_id"].values
    labels = df_train["y"].values  # not used by GroupKFold directly; handy for checks/logs

    # Effective K cannot exceed the number of unique groups
    unique_groups = int(np.unique(groups).size)
    eff_k = min(int(k), unique_groups)
    print(f"[CV] rows={len(df_train)}, unique_patients={unique_groups}, requested_k={k}, effective_k={eff_k}")
    if eff_k < 2:
        # Not enough distinct patients to split → persist "no result" payload
        if save_json is not None:
            ensure_dir(save_json.parent)
            with open(save_json, "w", encoding="utf-8") as f:
                json.dump({"best_mean_acc": 0.0, "params": None}, f, indent=2)
        return {"best": None, "best_mean_acc": 0.0}

    gkf = GroupKFold(n_splits=eff_k)

    # Grid search across folds
    rows: List[Dict] = []
    best_score, best_hp = -1.0, None
    for trial, hp in enumerate(grid, start=1):
        # Guardrails: coerce/clip minimal valid values
        hp = {
            **hp,
            "batch":  max(1, int(hp.get("batch", 32))),
            "epochs": max(1, int(hp.get("epochs", 1))),
            "lr":     float(hp.get("lr", 1e-3)),
            "fedprox": float(hp.get("fedprox", 0.0)),
        }
        fold_losses, fold_accs = [], []
        # Split by patient groups to avoid leakage
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

    # Persist results to disk for reuse by Client.PTBClient
    if save_csv is not None:
        ensure_dir(save_csv.parent)
        pd.DataFrame(rows).to_csv(save_csv, index=False)

    if save_json is not None:
        ensure_dir(save_json.parent)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump({"best_mean_acc": best_score, "params": best_hp}, f, indent=2)

    return {"best": best_hp, "best_mean_acc": best_score}


__all__ = [
    "run_client_cv",
    "default_grid",
]
