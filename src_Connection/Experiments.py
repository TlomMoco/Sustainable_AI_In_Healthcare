from __future__ import annotations
import subprocess
import argparse
import sys
import time
from pathlib import Path

# Minimal holder to show where to add experiment loops/config sweeps

CENTRALIZED_CMD = ["python", "-m", "src.Centralized"]
SERVER_CMD = ["python", "-m", "src.Server"]
CLIENT_CMD = lambda cid: ["python", "-m", "src.Client", "--cid", str(cid)]


if __name__ == "__main__" and False:
    # Example: run centralized once
    subprocess.run(CENTRALIZED_CMD, check=True)
    # For federated, open multiple terminals or use process manager (tmux, etc.))


# ---------
# ## 14) K-Fold CV for All Models (from notebook)
# ---------
from __future__ import annotations
import time as _time, math
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from . import config as CFG
from .models import create_model
from .utils import torch_loader_kwargs, log, set_seed, ensure_dir
from .data_loader import load_metadata as _dl_load_metadata  # legacy use (kept for context)
from .data_loader import make_feature_table as _dl_make_feature_table
from .data_preprocessing import make_label_encoder

# -----------------------------
# Core CV helpers (unchanged)
# -----------------------------
def _criterion(binary: bool, ce_weights=None, bce_pos_weight=None):
    if binary:
        if bce_pos_weight:
            return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([float(bce_pos_weight)]))
        return nn.BCEWithLogitsLoss()
    else:
        if ce_weights is not None:
            w = torch.tensor(ce_weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=w)
        return nn.CrossEntropyLoss()

class _PairDS(Dataset):
    def __init__(self, paths, y):
        self.paths = list(paths)
        self.y = np.asarray(y, np.int64)
        self.T = max(1, CFG.SEQ_LEN // max(1, CFG.DOWNSAMPLE_FACTOR))
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        from .data_loader import load_waveform_np
        x = load_waveform_np(self.paths[i], T=self.T, factor=CFG.DOWNSAMPLE_FACTOR)
        return torch.from_numpy(x), int(self.y[i])

def _eval_epoch(model, loader, binary, device):
    model.eval()
    crit = _criterion(binary)
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits.view(-1), yb.float()) if binary else crit(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits.view(-1)) >= 0.5).long() if binary else logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    loss = (loss_sum / total) if total else math.nan
    acc  = (correct / total) if total else math.nan
    return loss, acc

def _fit_for_epochs(model, dl_tr, dl_va, epochs, binary, device):
    model = model.to(device)
    base_lr = CFG.RECURRENT_LR if model.__class__.__name__.lower() in {"rnnsimple","tinyecglstm","annpooled"} else CFG.BASE_LR
    opt = optim.Adam(model.parameters(), lr=base_lr)
    crit = _criterion(binary)
    best_state, best_acc = None, -1.0
    for _ in range(int(epochs)):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits.view(-1), yb.float()) if binary else crit(logits, yb)
            loss.backward()
            if CFG.GRAD_CLIP_NORM: nn.utils.clip_grad_norm_(model.parameters(), float(CFG.GRAD_CLIP_NORM))
            opt.step()
        _, va_acc = _eval_epoch(model, dl_va, binary, device)
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def run_kfold_all(train_paths, y_train_encoded, label_encoder, *, device=None,
                  model_types=("CNN","RNN","LSTM","ANN")):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = int(CFG.KFOLDS); CV_EPOCHS = int(CFG.CV_EPOCHS)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=CFG.SEED)
    all_rows = []

    for mdl_name in model_types:
        log(f"[CV-ALL] {mdl_name}: {K}-fold, epochs={CV_EPOCHS}")
        rows = []
        start_m = _time.time()
        for fold, (tr_i, va_i) in enumerate(skf.split(train_paths, y_train_encoded), 1):
            ds_tr = _PairDS(train_paths[tr_i], y_train_encoded[tr_i])
            ds_va = _PairDS(train_paths[va_i], y_train_encoded[va_i])
            dl_tr = DataLoader(ds_tr, **torch_loader_kwargs(True,  CFG.BATCH_SIZE, device.type))
            dl_va = DataLoader(ds_va, **torch_loader_kwargs(False, CFG.BATCH_SIZE, device.type))
            t0 = _time.time()
            model = create_model(mdl_name, n_classes=len(label_encoder.classes_), binary=(len(label_encoder.classes_)==2))
            model = _fit_for_epochs(model, dl_tr, dl_va, CV_EPOCHS, (len(label_encoder.classes_)==2), device)
            t_fold = _time.time() - t0

            # Evaluate
            y_pred_idx = []
            with torch.no_grad():
                for xb, _ in dl_va:
                    xb = xb.to(device); logits = model(xb)
                    if len(label_encoder.classes_) == 2:
                        y_hat = (torch.sigmoid(logits.view(-1)) >= 0.5).long().cpu().numpy()
                    else:
                        y_hat = logits.argmax(1).cpu().numpy()
                    y_pred_idx.extend(y_hat)
            y_true_idx = y_train_encoded[va_i]
            y_true = label_encoder.inverse_transform(y_true_idx)
            y_pred = label_encoder.inverse_transform(np.array(y_pred_idx))

            rows.append({
                "model": mdl_name, "fold": fold, "time_sec": t_fold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            })
            log(f"[CV-ALL] {mdl_name} fold {fold}/{K} time={t_fold:.2f}s")

        total_m = _time.time() - start_m
        log(f"[CV-ALL] {mdl_name} total time: {total_m:.2f}s")
        all_rows.extend(rows)

    cv_all_df = pd.DataFrame(all_rows).reset_index(drop=True)
    return cv_all_df


# -----------------------------
# Orchestration helpers/CLI
# -----------------------------
def run_centralized():
    log("[RUN] Centralized")
    subprocess.run(CENTRALIZED_CMD, check=True)

def run_federated(n_clients: int):
    """
    Launch the Flower server and N clients in this process.
    (For production, prefer tmux/screen or a process manager.)
    """
    log(f"[RUN] Federated: server + {n_clients} clients")
    server = subprocess.Popen(SERVER_CMD)
    procs = []
    try:
        # Give the server a brief head start
        time.sleep(1.0)
        for cid in range(n_clients):
            p = subprocess.Popen(CLIENT_CMD(cid))
            procs.append(p)
        # Wait for clients to complete
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
    """
    set_seed(CFG.SEED)
    log("[RUN] K-Fold CV — building feature/minimal tables…")
    feat_df, features_df = _dl_make_feature_table(save_csv=True)

    # training pool = entire set (paths + labels), since this is CV
    paths = features_df["record_path"].astype(str).values
    y_series = features_df["label"].astype(str)
    le = make_label_encoder(y_series, y_series)  # union(train,test) = same for CV
    y_enc = le.transform(y_series.values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cv_df = run_kfold_all(paths, y_enc, le, device=device,
                          model_types=tuple(m for m, flag in {
                              "CNN":  CFG.RUN_TORCH_CNN,
                              "RNN":  CFG.RUN_TORCH_RNN,
                              "LSTM": CFG.RUN_TORCH_LSTM,
                              "ANN":  CFG.RUN_TORCH_ANN,
                          }.items() if flag))

    ensure_dir(CFG.RESULTS_DIR)
    out = Path(CFG.RESULTS_DIR) / f"cv_all_{int(time.time())}.csv"
    cv_df.to_csv(out, index=False)
    log(f"[RUN] CV results saved → {out}")
    print(cv_df.groupby("model")[["accuracy","f1_macro","f1_weighted"]].mean().round(4))


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Experiment runner")
    sub = p.add_subparsers(dest="mode", required=True)

    sub.add_parser("centralized", help="Run centralized training pipeline")

    p_fed = sub.add_parser("federated", help="Run Flower server + N clients")
    p_fed.add_argument("--clients", type=int, default=CFG.FL_N_CLIENTS, help="Number of clients to launch")

    sub.add_parser("cv", help="Run K-Fold CV over enabled deep models")

    return p.parse_args(argv)


def main(argv=None):
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
