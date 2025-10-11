from __future__ import annotations
import subprocess


# Minimal holder to show where to add experiment loops/config sweeps

CENTRALIZED_CMD = ["python", "-m", "src.Centralized"]
SERVER_CMD = ["python", "-m", "src.Server"]
CLIENT_CMD = lambda cid: ["python", "-m", "src.Client", "--cid", str(cid)]


if __name__ == "__main__":
    # Example: run centralized once
    subprocess.run(CENTRALIZED_CMD, check=True)
    # For federated, open multiple terminals or use process manager (tmux, etc.)




# ---------
# ## 14) K-Fold CV for All Models (from notebook)
# ---------
from __future__ import annotations
import time, math
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from . import config as CFG
from .models import create_model
from .utils import torch_loader_kwargs, log
from .data_loader import load_waveform_np

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

def run_kfold_all(train_paths, y_train_encoded, label_encoder, *, device=None, model_types=("CNN","RNN","LSTM","ANN")):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    K = int(CFG.KFOLDS); CV_EPOCHS = int(CFG.CV_EPOCHS)
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=CFG.SEED)
    all_rows = []

    for mdl_name in model_types:
        log(f"[CV-ALL] {mdl_name}: {K}-fold, epochs={CV_EPOCHS}")
        rows = []
        start_m = time.time()
        for fold, (tr_i, va_i) in enumerate(skf.split(train_paths, y_train_encoded), 1):
            ds_tr = _PairDS(train_paths[tr_i], y_train_encoded[tr_i])
            ds_va = _PairDS(train_paths[va_i], y_train_encoded[va_i])
            dl_tr = DataLoader(ds_tr, **torch_loader_kwargs(True,  CFG.BATCH_SIZE, device.type))
            dl_va = DataLoader(ds_va, **torch_loader_kwargs(False, CFG.BATCH_SIZE, device.type))
            t0 = time.time()
            model = create_model(mdl_name, n_classes=len(label_encoder.classes_), binary=(len(label_encoder.classes_)==2))
            model = _fit_for_epochs(model, dl_tr, dl_va, CV_EPOCHS, (len(label_encoder.classes_)==2), device)
            t_fold = time.time() - t0

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

        total_m = time.time() - start_m
        log(f"[CV-ALL] {mdl_name} total time: {total_m:.2f}s")
        all_rows.extend(rows)

    cv_all_df = pd.DataFrame(all_rows).reset_index(drop=True)
    return cv_all_df
