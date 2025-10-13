# src/Experiments.py
"""
Experiment orchestrator for the Sustainable AI in Healthcare project.

What this module does
---------------------
• Provides CLI entry points to:
  - Run the centralized deep-learning pipeline (via src.Centralized)
  - Launch a Flower server + N federated clients (Server/Client subprocesses)
  - Run K-fold cross-validation across enabled deep model families (CNN/RNN/LSTM/ANN)

• Implements a light CV loop (PyTorch) to compare model families using the same
  feature/minimal table and consistent data loaders.

Where this connects
-------------------
• src.Centralized : end-to-end centralized training + plots
• src.Server      : Flower server (federated learning)
• src.Client      : Flower clients (local training + FedProx/freezing)
• src.data_loader : make_feature_table (engineered+minimal tables)
• src.models      : create_model factory for CNN/RNN/LSTM/ANN
• src.utils       : logging, seeding, DataLoader kwargs
• src.config      : global switches (KFOLDS, CV_EPOCHS, RUN_TORCH_*...)

Outputs
-------
• CV results CSV saved under CFG.RESULTS_DIR with timestamped filename.
• Prints fold summaries and mean metrics per model to stdout.

Usage
-----
python -m src.Experiments centralized
python -m src.Experiments federated --clients 4
python -m src.Experiments cv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src import config as CFG
from src.models import create_model
from src.utils import torch_loader_kwargs, log, set_seed, ensure_dir
from src.data_loader import make_feature_table
from src.data_preprocessing import make_label_encoder


# --------------------------------------------------------------------------------------
# Launch helpers (use current interpreter for cross-platform compatibility)
# --------------------------------------------------------------------------------------
PY = sys.executable
CENTRALIZED_CMD = [PY, "-m", "src.Centralized"]
SERVER_CMD      = [PY, "-m", "src.Server"]
CLIENT_CMD      = lambda cid: [PY, "-m", "src.Client", "--cid", str(cid)]


# --------------------------------------------------------------------------------------
# CV utilities
# --------------------------------------------------------------------------------------
def _criterion(binary: bool, *, device: torch.device, ce_weights=None, bce_pos_weight=None):
    """Construct a device-aware loss function.

    Parameters
    ----------
    binary : bool
        If True, returns BCEWithLogitsLoss (optionally with pos_weight).
        Else, returns CrossEntropyLoss (optionally with class weights).
    device : torch.device
        Ensures any weight tensors are moved/created on the right device.
    ce_weights : Optional[array-like]
        Class weights for CE (one per class).
    bce_pos_weight : Optional[float]
        Positive class weight for BCEWithLogits.

    Returns
    -------
    torch.nn.modules.loss._Loss
    """
    if binary:
        if bce_pos_weight is not None:
            pw = torch.as_tensor([float(bce_pos_weight)], dtype=torch.float32, device=device)
            return nn.BCEWithLogitsLoss(pos_weight=pw)
        return nn.BCEWithLogitsLoss()
    if ce_weights is not None:
        w = torch.as_tensor(ce_weights, dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=w)
    return nn.CrossEntropyLoss()


class _PairDS(Dataset):
    """Minimal dataset: (paths, y) → (T, C) float32 tensor and int64 label.

    Notes
    -----
    • Loads WFDB records on-demand using data_loader.load_waveform_np.
    • Applies CFG.SEQ_LEN and CFG.DOWNSAMPLE_FACTOR for a consistent (T, C) shape.
    • Robust to loading errors: returns zeros tensor of shape (T, 12).
    """
    def __init__(self, paths, y):
        self.paths = list(paths)
        self.y = np.asarray(y, np.int64)
        self.T = max(1, int(CFG.SEQ_LEN // max(1, int(CFG.DOWNSAMPLE_FACTOR))))
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        from src.data_loader import load_waveform_np
        try:
            x = load_waveform_np(self.paths[i], T=self.T, factor=int(CFG.DOWNSAMPLE_FACTOR))
            if x.dtype != np.float32:
                x = x.astype("float32", copy=False)
        except Exception:
            # Robustness: unreadable record → zeros of correct shape (T, 12)
            x = np.zeros((self.T, 12), dtype=np.float32)
        return torch.from_numpy(x), int(self.y[i])


@torch.no_grad()
def _eval_epoch(model, loader, binary: bool, device: torch.device):
    """Evaluate loss/accuracy over a loader in eval() mode."""
    model.eval()
    crit = _criterion(binary, device=device)
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = crit(logits.view(-1), yb.float()) if binary else crit(logits, yb)
        loss_sum += loss.item() * xb.size(0)
        preds = (torch.sigmoid(logits.view(-1)) >= 0.5).long() if binary else logits.argmax(1)
        correct += int((preds == yb).sum().item())
        total += int(xb.size(0))
    loss = (loss_sum / total) if total else math.nan
    acc  = (correct / total) if total else math.nan
    return loss, acc


def _fit_for_epochs(model, dl_tr, dl_va, epochs: int, *, binary: bool, device: torch.device, base_lr: float):
    """Simple training loop with best-on-validation checkpointing.

    Flow
    ----
    • Train for `epochs` over dl_tr using Adam(base_lr)
    • Per epoch: evaluate on dl_va; keep best state by validation accuracy
    • Gradient clipping if CFG.GRAD_CLIP_NORM is set

    Returns
    -------
    model with best validation snapshot loaded.
    """
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=float(base_lr))
    crit = _criterion(binary, device=device)
    best_state, best_acc = None, -1.0

    for _ in range(int(epochs)):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits.view(-1), yb.float()) if binary else crit(logits, yb)
            loss.backward()
            if CFG.GRAD_CLIP_NORM:
                nn.utils.clip_grad_norm_(model.parameters(), float(CFG.GRAD_CLIP_NORM))
            opt.step()
        _, va_acc = _eval_epoch(model, dl_va, binary, device)
        if (va_acc is not None) and (va_acc > best_acc):
            best_acc = float(va_acc)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run_kfold_all(train_paths, y_train_encoded, label_encoder, *, device=None,
                  model_types=("CNN","RNN","LSTM","ANN")):
    """Run K-fold CV across selected model families.

    Parameters
    ----------
    train_paths : np.ndarray[str]
        WFDB base paths (no extension) for the full training pool (CV will split).
    y_train_encoded : np.ndarray[int]
        Integer-encoded labels aligned with `train_paths`.
    label_encoder : sklearn.preprocessing.LabelEncoder
        Fitted encoder (used to invert predictions for metrics).
    device : Optional[torch.device]
        Defaults to CUDA if available, else CPU.
    model_types : tuple[str, ...]
        Subset of {"CNN","RNN","LSTM","ANN"} to evaluate.

    Returns
    -------
    pd.DataFrame
        Row per (model, fold) with timing and macro/weighted scores.

    Notes
    -----
    • LR schedule mirrors Centralized.py: recurrent/ANN use RECURRENT_LR, others BASE_LR.
    • Uses StratifiedKFold(KFOLDS, shuffle=True, random_state=CFG.SEED).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = int(CFG.KFOLDS)
    CV_EPOCHS = int(CFG.CV_EPOCHS)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=CFG.SEED)
    all_rows = []
    is_binary = (len(label_encoder.classes_) == 2)

    for mdl_name in model_types:
        log(f"[CV-ALL] {mdl_name}: {K}-fold, epochs={CV_EPOCHS}")
        rows = []
        start_m = time.time()

        # LR policy by model family (match Centralized.py)
        base_lr = float(CFG.RECURRENT_LR) if mdl_name.lower() in {"rnn", "lstm", "ann"} else float(CFG.BASE_LR)

        for fold, (tr_i, va_i) in enumerate(skf.split(train_paths, y_train_encoded), 1):
            ds_tr = _PairDS(train_paths[tr_i], y_train_encoded[tr_i])
            ds_va = _PairDS(train_paths[va_i], y_train_encoded[va_i])
            dl_tr = DataLoader(ds_tr, **torch_loader_kwargs(True,  CFG.BATCH_SIZE, device.type))
            dl_va = DataLoader(ds_va, **torch_loader_kwargs(False, CFG.BATCH_SIZE, device.type))

            t0 = time.time()
            model = create_model(mdl_name, n_classes=len(label_encoder.classes_), binary=is_binary)
            model = _fit_for_epochs(model, dl_tr, dl_va, CV_EPOCHS, binary=is_binary, device=device, base_lr=base_lr)
            t_fold = time.time() - t0

            # Evaluate on validation fold
            y_pred_idx = []
            with torch.no_grad():
                for xb, _ in dl_va:
                    xb = xb.to(device)
                    logits = model(xb)
                    if is_binary:
                        y_hat = (torch.sigmoid(logits.view(-1)) >= 0.5).long().cpu().numpy()
                    else:
                        y_hat = logits.argmax(1).cpu().numpy()
                    y_pred_idx.extend(y_hat)

            y_true_idx = y_train_encoded[va_i]
            y_true = label_encoder.inverse_transform(y_true_idx)
            y_pred = label_encoder.inverse_transform(np.array(y_pred_idx))

            rows.append({
                "model": mdl_name, "fold": int(fold), "time_sec": float(t_fold),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            })
            log(f"[CV-ALL] {mdl_name} fold {fold}/{K} time={t_fold:.2f}s")

        total_m = time.time() - start_m
        log(f"[CV-ALL] {mdl_name} total time: {total_m:.2f}s")
        all_rows.extend(rows)

    cv_all_df = pd.DataFrame(all_rows).reset_index(drop=True)
    return cv_all_df


# --------------------------------------------------------------------------------------
# Orchestration helpers/CLI
# --------------------------------------------------------------------------------------
def run_centralized():
    """Spawn the centralized training pipeline as a subprocess.

    Equivalent to:
        python -m src.Centralized
    """
    log("[RUN] Centralized")
    subprocess.run(CENTRALIZED_CMD, check=True)


def run_federated(n_clients: int):
    """
    Launch the Flower server and N clients from this process.

    For production runs, prefer separate terminals or a process manager;
    this helper is convenient for quick local experiments.

    Behavior
    --------
    • Starts the server (Server.py) as a subprocess.
    • Staggers client launches slightly to keep logs readable.
    • Waits for all client processes to finish; then terminates the server.

    On Windows:
    • Uses CREATE_NO_WINDOW (if available) to avoid extra console windows.
    """
    log(f"[RUN] Federated: server + {n_clients} clients")
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # avoid new console windows

    server = subprocess.Popen(SERVER_CMD, creationflags=creationflags)
    procs = []
    try:
        time.sleep(1.0)  # give server a head start
        for cid in range(n_clients):
            p = subprocess.Popen(CLIENT_CMD(cid), creationflags=creationflags)
            procs.append(p)
            time.sleep(0.2)  # tiny stagger helps logs
        rcodes = [p.wait() for p in procs]
        log(f"[RUN] Clients finished: {rcodes}")
    except KeyboardInterrupt:
        log("[RUN] KeyboardInterrupt — terminating children")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        if server.poll() is None:
            server.terminate()


def run_cv():
    """
    Build minimal deep table, encode labels, and run K-Fold across
    the selected models (CNN, RNN, LSTM, ANN). Results are saved to CSV.

    Steps
    -----
    1) Seed for reproducibility.
    2) Build (feature_df, features_df) via data_loader.make_feature_table(...)
       - features_df['record_path'] and ['label'] provide the CV pool.
    3) Fit LabelEncoder on labels; integer-encode them.
    4) Determine enabled model families based on CFG.RUN_TORCH_* flags.
    5) run_kfold_all(...), then save the CV table to RESULTS_DIR with a timestamp.
    """
    set_seed(CFG.SEED)
    log("[RUN] K-Fold CV — building feature/minimal tables…")
    feat_df, features_df = make_feature_table(save_csv=True)

    # training pool = entire set (paths + labels), since this is CV
    paths = features_df["record_path"].astype(str).values
    y_series = features_df["label"].astype(str)
    le = make_label_encoder(y_series, y_series)
    y_enc = le.transform(y_series.values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enabled = {
        "CNN":  CFG.RUN_TORCH_CNN,
        "RNN":  CFG.RUN_TORCH_RNN,
        "LSTM": CFG.RUN_TORCH_LSTM,
        "ANN":  CFG.RUN_TORCH_ANN,
    }
    model_types = tuple([m for m, flag in enabled.items() if flag])

    cv_df = run_kfold_all(paths, y_enc, le, device=device, model_types=model_types)

    ensure_dir(CFG.RESULTS_DIR)
    out = Path(CFG.RESULTS_DIR) / f"cv_all_{int(time.time())}.csv"
    cv_df.to_csv(out, index=False)
    log(f"[RUN] CV results saved → {out}")
    try:
        print(cv_df.groupby("model")[["accuracy","f1_macro","f1_weighted"]].mean().round(4))
    except Exception:
        print(cv_df.head())


def _parse_args(argv=None):
    """Define CLI with subcommands: centralized | federated | cv."""
    p = argparse.ArgumentParser(description="Experiment runner")
    sub = p.add_subparsers(dest="mode", required=True)

    sub.add_parser("centralized", help="Run centralized training pipeline")

    p_fed = sub.add_parser("federated", help="Run Flower server + N clients")
    p_fed.add_argument("--clients", type=int, default=CFG.FL_N_CLIENTS, help="Number of clients to launch")

    sub.add_parser("cv", help="Run K-Fold CV over enabled deep models")
    return p.parse_args(argv)


def main(argv=None):
    """Dispatch to the requested experiment mode based on CLI args."""
    args = _parse_args(argv)
    if args.mode == "centralized":
        run_centralized()
    elif args.mode == "federated":
        run_federated(int(getattr(args, "clients", CFG.FL_N_CLIENTS)))
    elif args.mode == "cv":
        run_cv()
    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main(sys.argv[1:])
