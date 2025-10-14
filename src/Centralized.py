# src/Centralized.py
from __future__ import annotations

"""
Centralized.py — Deep Models (centralized training only)

This version is aligned with your current codebase:
- Imports from src.* (not src_Connection.*)
- Uses data_loader waveforms + patient-aware 3-way split
- Provides local Dataset, label-encoding, plotting, and evaluation
- Supports CNN/LSTM via src.models.create_model

Outputs:
- results/models/best_<model>.pt          (weights)
- results/models/history_<model>.csv      (training curves)
- results/centralized_eval.csv            (summary table)
- results/viz/confusion_<model>.png       (confusion matrix)
- results/viz/learning_curves.png         (per-model curves)
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from src.config import (
    RESULTS_DIR, SEED, SAMPLE_RATE, N_CLASSES, SPLITS, MODEL,
    LR as CFG_LR,  # reuse your LR
)
from src.data_loader import (
    load_metadata, map_superclasses, filter_single_label,
    stratified_patient_split_3way, load_waveform,
    compute_perlead_norm_stats, normalize_signal,
    SUPERCLASSES,
)
from src.models import create_model
from src.utils import set_seed, ensure_dir


# -----------------------------------------------------------------------------
# Local helpers (device, datasets, plotting)
# -----------------------------------------------------------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS (Apple Silicon) if available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def torch_loader_kwargs(shuffle: bool, batch: int, device_type: str) -> dict:
    # Reasonable defaults; tune if you like
    pin = device_type == "cuda"
    num_w = 2 if device_type == "cuda" else 0
    return {"batch_size": int(batch), "shuffle": bool(shuffle), "num_workers": num_w, "pin_memory": pin}


class PTBWaveformDataset(Dataset):
    """Lazily loads PTB-XL waveforms row-by-row and returns (X, y_idx)."""
    def __init__(self, df, le: LabelEncoder, mu: np.ndarray, sigma: np.ndarray, sample_rate: int):
        self.df = df.reset_index(drop=True)
        self.le = le
        self.mu = mu
        self.sigma = sigma
        self.fs = int(sample_rate)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sig = load_waveform(row, sampling_rate=self.fs)            # (12, T)
        sig = normalize_signal(sig, self.mu, self.sigma, eps=1e-6) # z-score per lead
        x = torch.tensor(sig, dtype=torch.float32)                 # (12, T)
        y_str = str(row["y"])
        y_idx = int(np.where(self.le.classes_ == y_str)[0][0])     # encode with fitted LE
        y = torch.tensor(y_idx, dtype=torch.long)
        return x, y


def plot_confusion_png(cm: np.ndarray, labels: List[str], title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    im = ax.imshow(cmn, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, label="Row-normalized")
    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_learning_curves_png(histories: Dict[str, Dict[str, List[float]]], out_dir: Path) -> None:
    ensure_dir(out_dir)
    # Loss
    fig1, ax1 = plt.subplots()
    for name, h in histories.items():
        ax1.plot(range(1, len(h["loss"]) + 1), h["loss"], label=f"{name} (train)")
        ax1.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], linestyle="--", label=f"{name} (val)")
    ax1.set_title("Loss (eval-mode train vs val)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True); ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / "learning_curves_loss.png", dpi=150)
    plt.close(fig1)

    # Accuracy
    fig2, ax2 = plt.subplots()
    for name, h in histories.items():
        ax2.plot(range(1, len(h["accuracy"]) + 1), h["accuracy"], label=f"{name} (train)")
        ax2.plot(range(1, len(h["val_accuracy"]) + 1), h["val_accuracy"], linestyle="--", label=f"{name} (val)")
    ax2.set_title("Accuracy (eval-mode train vs val)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True); ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "learning_curves_acc.png", dpi=150)
    plt.close(fig2)


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def run_deep_models(seed: int | None = None) -> None:
    # --- Repro / device -------------------------------------------------
    set_seed(int(seed if seed is not None else SEED))
    device = pick_device()
    print(f"[Torch] device={device}")

    # --- Load & split (patient-aware 3-way) -----------------------------
    ptb = load_metadata()
    df_all = filter_single_label(map_superclasses(ptb))
    tr_df, va_df, te_df = stratified_patient_split_3way(df_all, splits=(SPLITS["train"], SPLITS["val"], SPLITS["test"]), seed=SEED)

    # Label encoder on TRAIN classes
    le = LabelEncoder().fit(tr_df["y"].astype(str).values)
    n_classes = len(le.classes_)
    assert n_classes == N_CLASSES, f"Config N_CLASSES={N_CLASSES} but data has {n_classes} classes: {list(le.classes_)}"

    # Per-lead normalization stats from TRAIN only
    mu_tr, sigma_tr = compute_perlead_norm_stats(tr_df, sampling_rate=SAMPLE_RATE)

    # Datasets/loaders
    def make_loader(df, shuffle: bool, batch: int):
        ds = PTBWaveformDataset(df, le, mu_tr, sigma_tr, SAMPLE_RATE)
        return DataLoader(ds, **torch_loader_kwargs(shuffle, batch, device.type))

    batch = 64  # default; you can also read from config if you prefer
    train_loader = make_loader(tr_df, shuffle=True,  batch=batch)
    val_loader   = make_loader(va_df, shuffle=False, batch=max(64, batch))
    test_loader  = make_loader(te_df, shuffle=False, batch=max(64, batch))

    # Loss (always multi-class here)
    crit = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _eval(model, loader) -> Tuple[float, float]:
        model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits, yb)
            loss_sum += float(loss.item()) * len(yb)
            pred = logits.argmax(1)
            correct += int((pred == yb).sum())
            total += len(yb)
        return (loss_sum / total if total else math.nan,
                correct / total if total else math.nan)

    def train_one(name: str, epochs: int = 12, base_lr: float = None):
        model_type = "cnn" if name.lower() == "cnn" else "lstm"
        model = create_model(
            n_classes=n_classes,
            model_type=model_type,
            hidden=MODEL.get("lstm_hidden", 128),
            layers=MODEL.get("lstm_layers", 1),
            bidir=MODEL.get("bidirectional", True),
        ).to(device)

        lr = float(CFG_LR if base_lr is None else base_lr)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-5)

        hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        best_state, best_val, no_improve, early_pat = None, float("inf"), 0, None  # add patience in config if needed

        # Stable eval-mode train loader (same as train but shuffle False)
        train_eval_loader = make_loader(tr_df, shuffle=False, batch=max(64, batch))

        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                # Optional grad clip (set a value if you want)
                # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            tr_loss, tr_acc = _eval(model, train_eval_loader)
            va_loss, va_acc = _eval(model, val_loader)
            sch.step(va_loss if not math.isnan(va_loss) else tr_loss)

            hist["loss"].append(tr_loss)
            hist["accuracy"].append(tr_acc)
            hist["val_loss"].append(va_loss)
            hist["val_accuracy"].append(va_acc)

            print(f"[{name}] epoch {ep:02d}/{epochs}  "
                  f"train(eval loss={tr_loss:.4f} acc={tr_acc:.4f})  "
                  f"val(loss={va_loss:.4f} acc={va_acc:.4f})")

            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if early_pat is not None and no_improve >= int(early_pat):
                    print(f"[{name}] Early stop at epoch {ep}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model, hist

    # Which models to run: support both; default run both
    run_cnn  = True
    run_lstm = True

    model_dir = (Path(RESULTS_DIR) / "models"); model_dir.mkdir(parents=True, exist_ok=True)
    viz_dir   = (Path(RESULTS_DIR) / "viz");    viz_dir.mkdir(parents=True, exist_ok=True)

    histories: Dict[str, Dict[str, List[float]]] = {}
    eval_rows: List[dict] = {}

    trained: Dict[str, nn.Module] = {}

    if run_cnn:
        print("\n[Train] === CNN ===")
        cnn, h = train_one("cnn", epochs=12, base_lr=CFG_LR)
        trained["CNN"] = cnn; histories["CNN"] = h
        torch.save(cnn.state_dict(), model_dir / "best_cnn.pt")
        with open(model_dir / "history_cnn.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch","loss","val_loss","accuracy","val_accuracy"])
            for i in range(len(h["loss"])):
                w.writerow([i+1,h["loss"][i],h["val_loss"][i],h["accuracy"][i],h["val_accuracy"][i]])
        print("Saved weights and history for CNN.")

    if run_lstm:
        print("\n[Train] === LSTM ===")
        lstm, h = train_one("lstm", epochs=12, base_lr=CFG_LR)
        trained["LSTM"] = lstm; histories["LSTM"] = h
        torch.save(lstm.state_dict(), model_dir / "best_lstm.pt")
        with open(model_dir / "history_lstm.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch","loss","val_loss","accuracy","val_accuracy"])
            for i in range(len(h["loss"])):
                w.writerow([i+1,h["loss"][i],h["val_loss"][i],h["accuracy"][i],h["val_accuracy"][i]])
        print("Saved weights and history for LSTM.")

    # --- Evaluate on TEST ------------------------------------------------
    rows = []
    for name, model in trained.items():
        model.eval()
        all_true, all_pred = [], []
        loss_sum, total = 0.0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss_sum += float(crit(logits, yb).item()) * len(yb)
                pred = logits.argmax(1)
                all_true.extend(yb.cpu().numpy().tolist())
                all_pred.extend(pred.cpu().numpy().tolist())
                total += len(yb)
        test_loss = loss_sum / total if total else math.nan
        test_acc  = float(np.mean(np.equal(all_true, all_pred))) if total else math.nan
        cm = confusion_matrix(all_true, all_pred, labels=list(range(n_classes)))
        plot_confusion_png(cm, labels=list(le.classes_), title=f"Confusion — {name}", save_path=viz_dir / f"confusion_{name}.png")
        print(f"\n[{name}] TEST  loss={test_loss:.4f}  acc={test_acc:.4f}")
        print(classification_report(all_true, all_pred, target_names=list(le.classes_), digits=4))
        rows.append({"model": name, "test_loss": f"{test_loss:.6f}", "test_acc": f"{test_acc:.6f}"})

    # Save unified eval
    results_csv = Path(RESULTS_DIR) / "centralized_eval.csv"
    ensure_dir(results_csv.parent)
    with open(results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","test_loss","test_acc"])
        w.writeheader(); w.writerows(rows)
    print("Saved evaluation:", results_csv)

    # Learning curves
    plot_learning_curves_png(histories, viz_dir)
    print("\n[Done] Centralized run complete.")


def main():
    run_deep_models(seed=SEED)


if __name__ == "__main__":
    main()
