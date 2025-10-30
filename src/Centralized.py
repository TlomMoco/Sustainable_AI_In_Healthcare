from __future__ import annotations

import csv
import math
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from src.config import (
    RESULTS_DIR, SEED, SAMPLE_RATE, N_CLASSES, SPLITS, MODEL,
    LR as CFG_LR, ANOVA_FALLBACK_LEADS, ANOVA_FSCORE_THRESHOLD,
    CLASS_WEIGHTS, SUPERCLASSES
)
from src.data_loader import (
    load_metadata, map_superclasses, filter_single_label,
    stratified_patient_split_3way, load_waveform,
    compute_perlead_norm_stats, normalize_signal,
    compute_anova_lead_mask_by_threshold,
)
from src.models import create_model
from src.utils import set_seed, ensure_dir, class_weights_from_df


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def torch_loader_kwargs(shuffle: bool, batch: int, device_type: str) -> dict:
    pin = device_type == "cuda"
    num_w = 2 if device_type == "cuda" else 0
    return {"batch_size": int(batch), "shuffle": bool(shuffle), "num_workers": num_w, "pin_memory": pin}


class PTBWaveformDataset(Dataset):
    """Loads PTB-XL waveforms row-by-row; slices to selected leads and applies z-score."""
    def __init__(self, df, le: LabelEncoder, mu: np.ndarray, sigma: np.ndarray,
                 sample_rate: int, lead_idx: Optional[List[int]] = None):
        self.df = df.reset_index(drop=True)
        self.le = le
        self.mu = mu
        self.sigma = sigma
        self.fs = int(sample_rate)
        self.lead_idx = None if lead_idx is None else list(int(i) for i in lead_idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        sig = load_waveform(row, sampling_rate=self.fs)            # (12, T)
        if self.lead_idx is not None:
            sig = sig[self.lead_idx, :]                            # (K, T)
        sig = normalize_signal(sig, self.mu, self.sigma, eps=1e-6) # z-score per selected lead
        x = torch.tensor(sig, dtype=torch.float32)                 # (K, T)
        y_str = str(row["y"])
        y_idx = int(np.where(self.le.classes_ == y_str)[0][0])
        y = torch.tensor(y_idx, dtype=torch.long)
        return x, y


def plot_confusion_png(cm: np.ndarray, labels: List[str], title: str, save_path: Path, model_name: str = "") -> None:
    """Plot confusion matrix with percentage annotations in professional style."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Normalize to percentages
    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True)) * 100
    
    # Create heatmap with Blues colormap
    im = ax.imshow(cmn, aspect="auto", cmap="Blues", vmin=0, vmax=100)
    
    # Add text annotations
    n = len(labels)
    for i in range(n):
        for j in range(n):
            val = cmn[i, j]
            # White text for values > 50%, black for <= 50%
            text_color = "white" if val > 50 else "black"
            ax.text(j, i, f"{val:.2f}", 
                   ha="center", va="center",
                   color=text_color, fontsize=12, weight="bold")
    
    # Styling with model name
    if model_name:
        full_title = f"Confusion Matrix - Prediction Percentages\n{model_name.upper()}"
    else:
        full_title = "Confusion Matrix - Prediction Percentages"
    
    ax.set_title(full_title, fontsize=16, pad=20, weight="bold")
    ax.set_xlabel("Predicted Labels", fontsize=14, labelpad=10)
    ax.set_ylabel("True Labels", fontsize=14, labelpad=10)
    
    # Set ticks
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=12)
    
    # Add colorbar with % label
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("%", fontsize=14, rotation=0, labelpad=20)
    cbar.ax.tick_params(labelsize=11)
    
    # Clean layout
    fig.tight_layout()
    ensure_dir(save_path.parent)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_learning_curves_png(histories: Dict[str, Dict[str, List[float]]], out_dir: Path) -> None:
    ensure_dir(out_dir)
    # Loss
    fig1, ax1 = plt.subplots()
    for name, h in histories.items():
        ax1.plot(range(1, len(h["loss"]) + 1), h["loss"], label=f"{name} (train)")
        ax1.plot(range(1, len(h["val_loss"]) + 1), h["val_loss"], linestyle="--", label=f"{name} (val)")
    ax1.set_title("Loss (eval-mode train vs val)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
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
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.grid(True); ax2.legend()
    fig2.tight_layout()
    fig2.savefig(out_dir / "learning_curves_acc.png", dpi=150)
    plt.close(fig2)


def run_deep_models(seed: int | None = None) -> None:
    # --- Repro / device -------------------------------------------------
    set_seed(int(seed if seed is not None else SEED))
    device = pick_device()
    print(f"[Torch] device={device}")

    # --- Load & split (patient-aware 3-way) -----------------------------
    ptb = load_metadata()
    df_all = filter_single_label(map_superclasses(ptb))
    tr_df, va_df, te_df = stratified_patient_split_3way(
        df_all, splits=(SPLITS["train"], SPLITS["val"], SPLITS["test"]), seed=SEED
    )

    # --- Lead selection on TRAIN ---------------------------------------
    lead_mask, kept_leads, _ = compute_anova_lead_mask_by_threshold(
        tr_df, RESULTS_DIR, feature_set="engineered",
        fscore_threshold=ANOVA_FSCORE_THRESHOLD, fallback_k=ANOVA_FALLBACK_LEADS
    )
    lead_idx: List[int] = [int(i) for i in kept_leads]
    print(f"[Centralized] ANOVA threshold={ANOVA_FSCORE_THRESHOLD} → keeping leads (1-based): {[i+1 for i in lead_idx]}")

    # Label encoder on TRAIN classes
    le = LabelEncoder().fit(tr_df["y"].astype(str).values)
    n_classes = len(le.classes_)
    assert n_classes == N_CLASSES, f"Config N_CLASSES={N_CLASSES} but data has {n_classes} classes: {list(le.classes_)}"

    # Per-lead normalization stats from TRAIN only, then slice to selected leads
    mu_all, sigma_all = compute_perlead_norm_stats(tr_df, sampling_rate=SAMPLE_RATE)
    mu_sel, sigma_sel = mu_all[lead_idx], sigma_all[lead_idx]

    # Datasets/loaders (true lead removal)
    def make_loader(df, shuffle: bool, batch: int):
        ds = PTBWaveformDataset(df, le, mu_sel, sigma_sel, SAMPLE_RATE, lead_idx=lead_idx)
        return DataLoader(ds, **torch_loader_kwargs(shuffle, batch, device.type))

    batch = 64
    train_loader = make_loader(tr_df, shuffle=True,  batch=batch)
    val_loader   = make_loader(va_df, shuffle=False, batch=max(64, batch))
    test_loader  = make_loader(te_df, shuffle=False, batch=max(64, batch))

    # --- Loss function (class-weighted CE with optional smoothing) ---
    if CLASS_WEIGHTS.get("enabled", False):
        w_np = class_weights_from_df(
            tr_df,
            classes=SUPERCLASSES,
            boost=CLASS_WEIGHTS.get("boost")
        )
        w_t = torch.tensor(w_np, dtype=torch.float32, device=device)
        ls = float(CLASS_WEIGHTS.get("label_smoothing", 0.0))
        crit = nn.CrossEntropyLoss(weight=w_t, label_smoothing=ls)
    else:
        crit = nn.CrossEntropyLoss()

    @torch.no_grad()
    def _eval(model_in: nn.Module, loader) -> Tuple[float, float]:
        model_in.eval()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model_in(xb)
            loss = crit(logits, yb)
            loss_sum += float(loss.item()) * len(yb)
            pred = logits.argmax(1)
            correct += int(pred.eq(yb).sum().item())
            total += len(yb)
        return (loss_sum / total if total else math.nan,
                correct / total if total else math.nan)

    def train_one(name: str, epochs: int = 12, base_lr: float = None):
        model_type = "cnn" if name.lower() == "cnn" else "lstm"
        model = create_model(
            n_classes=n_classes,
            model_type=model_type,
            n_leads=len(lead_idx),                 # <– variable input channels
            hidden=MODEL.get("lstm_hidden", 128),
            layers=MODEL.get("lstm_layers", 1),
            bidir=MODEL.get("bidirectional", True),
        ).to(device)

        lr = float(CFG_LR if base_lr is None else base_lr)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-5)

        hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        best_state, best_val, no_improve = None, float("inf"), 0
        early_pat: Optional[int] = None  # set to an int (e.g., 3) to enable early stopping

        train_eval_loader = make_loader(tr_df, shuffle=False, batch=max(64, batch))

        for ep in range(1, epochs + 1):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
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
                if early_pat is not None:
                    if no_improve >= early_pat:
                        print(f"[{name}] Early stop at epoch {ep}")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model, hist

    # Which models to run
    run_cnn  = True
    run_lstm = True

    model_dir = (Path(RESULTS_DIR) / "models"); model_dir.mkdir(parents=True, exist_ok=True)
    viz_dir   = (Path(RESULTS_DIR) / "viz");    viz_dir.mkdir(parents=True, exist_ok=True)

    histories: Dict[str, Dict[str, List[float]]] = {}
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
        
        # Pass model name to confusion matrix plot
        plot_confusion_png(
            cm, 
            labels=list(le.classes_), 
            title=f"Confusion – {name}", 
            save_path=viz_dir / f"confusion_{name}.png",
            model_name=name  # ← ADD MODEL NAME HERE
        )

        # Detailed report (precision/recall/F1 per class + macro) → CSV
        rep = classification_report(all_true, all_pred, target_names=list(le.classes_), digits=4, output_dict=True)
        rep_df = pd.DataFrame(rep).transpose()
        rep_path = Path(RESULTS_DIR) / f"centralized_report_{name}.csv"
        ensure_dir(rep_path.parent)
        rep_df.to_csv(rep_path, index=True)

        macro = rep.get("macro avg", {})
        rows.append({
            "model": name,
            "test_loss": f"{test_loss:.6f}",
            "test_acc": f"{test_acc:.6f}",
            "precision_macro": f"{macro.get('precision', float('nan')):.6f}",
            "recall_macro": f"{macro.get('recall', float('nan')):.6f}",
            "f1_macro": f"{macro.get('f1-score', float('nan')):.6f}",
            "n_leads": str(len(lead_idx)),
        })

        # Also print a human-readable summary
        print(f"\n[{name}] TEST  loss={test_loss:.4f}  acc={test_acc:.4f}")
        print(classification_report(all_true, all_pred, target_names=list(le.classes_), digits=4))

    results_csv = Path(RESULTS_DIR) / "centralized_eval.csv"
    ensure_dir(results_csv.parent)
    with open(results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","test_loss","test_acc","precision_macro","recall_macro","f1_macro","n_leads"])
        w.writeheader(); w.writerows(rows)
    print("Saved evaluation:", results_csv)

    plot_learning_curves_png(histories, viz_dir)
    print("\n[Done] Centralized run complete.")


def main():
    run_deep_models(seed=SEED)


if __name__ == "__main__":
    main()