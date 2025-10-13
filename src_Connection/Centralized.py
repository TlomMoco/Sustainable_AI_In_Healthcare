"""Centralized.py — Deep Models (centralized training only)

Purpose
-------
Run *centralized* (non-federated) training and evaluation for deep models
(CNN / RNN / LSTM / ANN) in PyTorch on PTB-XL data.

High-level flow
---------------
1) Build engineered feature tables (DEEP API subset)           -> data_loader
2) Patient-safe boolean-mask train/test split                   -> data_loader
3) Label encoding (fit on train; apply to test)                 -> data_preprocessing
4) Create train/val DataLoaders + stable train-eval loader      -> data_preprocessing/utils
5) Train selected models with LR scheduling & (optional) clip   -> models/config
6) Wrap trained models in TorchAdapter & evaluate on test       -> results_visualization
7) Save confusion matrix & learning curves                      -> results_visualization

Key connections (where things live)
-----------------------------------
• src_Connection.config (as CFG)                     : paths & hyperparameters (single source of truth)
• src_Connection.utils                               : set_seed, pick_device, log, sanitize_mps_env, torch_loader_kwargs
• src_Connection.data_loader                         : make_feature_table, train_test_split
• src_Connection.data_preprocessing                  : make_label_encoder, make_train_val_loaders, ECGDataset
• src_Connection.models                              : create_model (factory for CNN/RNN/LSTM/ANN)
• src_Connection.results_visualization               : TorchAdapter, evaluate_models, plot_confusion, plot_learning_curves

I/O & side effects
------------------
• Reads PTB-XL metadata/paths as configured by CFG
• Writes figures under <CFG.RESULTS_DIR>/viz/
• Writes evaluation summary to <CFG.RESULTS_DIR>/centralized_eval.csv
• Optionally saves best weights per model to <CFG.RESULTS_DIR>/models/

Config keys referenced
----------------------
SEED, VAL_FRAC, BATCH_SIZE, EPOCHS,
BASE_LR, RECURRENT_LR, GRAD_CLIP_NORM,
RUN_TORCH_CNN, RUN_TORCH_RNN, RUN_TORCH_LSTM, RUN_TORCH_ANN,
CONFUSION_COLLAPSE_TO_3, RESULTS_DIR,
EARLY_STOP_PATIENCE (optional), SAVE_BEST_WEIGHTS (optional)

Assumptions
-----------
• create_model(name, n_classes, binary=...) returns logits shaped:
    - binary: (N,) or (N,1) usable with BCEWithLogitsLoss
    - multi : (N, C) usable with CrossEntropyLoss
• ECGDataset yields (xb, yb) tensors compatible with the chosen loss.
• TorchAdapter signature may vary (device optional) -> handled via try/except.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src_Connection import config as CFG
from src_Connection.utils import (
    set_seed,
    pick_device,
    log,
    sanitize_mps_env,
    torch_loader_kwargs,
)
from src_Connection.data_loader import (
    make_feature_table as build_feature_tables,  # returns (feature_df, features_df)
    train_test_split as mask_split,              # boolean-mask split for deep set
)
from src_Connection.data_preprocessing import (
    make_label_encoder,
    make_train_val_loaders,
    ECGDataset,
)
from src_Connection.models import create_model
from src_Connection.results_visualization import (
    TorchAdapter,
    evaluate_models,
    plot_confusion,
    plot_learning_curves,
)

# -----------------------------------------------------------------------------
# Helper to read config (supports top-level vars or NOTEBOOK[...] fallbacks)
# -----------------------------------------------------------------------------
def C(name: str, default=None):
    """Get a config value with a safe fallback."""
    return getattr(CFG, name, CFG.NOTEBOOK.get(name, default))


# -----------------------------------------------------------------------------
# Deep learning pipeline (CNN/RNN/LSTM/ANN in PyTorch)
# -----------------------------------------------------------------------------
def run_deep_models(seed: int | None = None) -> None:
    """Orchestrate centralized training & evaluation for selected deep models."""
    # ---------- Reproducibility & device selection ----------
    seed = int(seed if seed is not None else C("SEED", 42))
    set_seed(seed)
    sanitize_mps_env()
    device = pick_device()
    log(f"[Torch] device={device}")

    # ---------- Feature tables & split ----------
    # Build engineered feature table + minimal meta (DEEP subset).
    # Returns two tables; deep models use `features_df`.
    _, features_df = build_feature_tables(save_csv=True)

    # Patient-safe boolean masks avoid leakage across splits.
    train_mask, test_mask = mask_split(features_df)

    # ---------- Paths + labels for deep models ----------
    train_paths = features_df.loc[train_mask, "record_path"].astype(str).values
    test_paths  = features_df.loc[test_mask,  "record_path"].astype(str).values
    y_tr_series = features_df.loc[train_mask, "label"].astype(str)
    y_te_series = features_df.loc[test_mask,  "label"].astype(str)

    # Fit label encoder on train; transform train labels.
    # IMPORTANT: We keep raw test labels (strings) for reporting & adapters.
    le = make_label_encoder(y_tr_series, y_te_series)
    y_tr = le.transform(y_tr_series.values)
    y_te = y_te_series.values
    n_classes = len(le.classes_)
    is_binary = (n_classes == 2)

    # ---------- DataLoaders ----------
    train_loader, val_loader, tr_pos, _ = make_train_val_loaders(
        train_paths, y_tr, device=device, val_frac=C("VAL_FRAC", 0.20)
    )

    # Stable (non-shuffled) train-eval loader to compute *eval-mode* train metrics
    train_eval_ds = ECGDataset(train_paths[tr_pos], y_tr[tr_pos])
    train_eval_loader = DataLoader(
        train_eval_ds, **torch_loader_kwargs(False, C("BATCH_SIZE", 24), device.type)
    )

    # ---------- Loss selection ----------
    def _criterion():
        return nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    # ---------- Evaluation helper (single pass) ----------
    @torch.no_grad()
    def _eval_epoch(model, loader) -> Tuple[float, float]:
        """Return (loss_mean, accuracy) on `loader`."""
        model.eval()
        crit = _criterion()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = crit(logits.view(-1), yb.float()) if is_binary else crit(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits.view(-1)) >= 0.5).long() if is_binary else logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        loss_mean = loss_sum / total if total else math.nan
        acc = correct / total if total else math.nan
        return loss_mean, acc

    # ---------- Train a single model type ----------
    def train_model_torch(name: str, epochs: int | None = None):
        """Train one model (CNN/RNN/LSTM/ANN) and return (model, history)."""
        epochs = int(epochs if epochs is not None else C("EPOCHS", 12))
        model = create_model(name, n_classes, binary=is_binary).to(device)

        base_lr = C("RECURRENT_LR", 2e-4) if name.lower() in {"rnn", "lstm", "ann"} else C("BASE_LR", 3e-4)
        opt = optim.Adam(model.parameters(), lr=base_lr)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-5)
        crit = _criterion()

        hist: Dict[str, list] = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        best_state, best_val = None, float("inf")
        no_improve = 0
        early_pat = C("EARLY_STOP_PATIENCE", None)

        for ep in range(1, epochs + 1):
            # --- train phase (dropout ON) ---
            model.train()
            ep_loss = ep_total = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = crit(logits.view(-1), yb.float()) if is_binary else crit(logits, yb)
                loss.backward()
                if C("GRAD_CLIP_NORM", None):
                    nn.utils.clip_grad_norm_(model.parameters(), float(C("GRAD_CLIP_NORM")))
                opt.step()
                ep_loss += loss.item() * xb.size(0)
                ep_total += xb.size(0)

            # --- metrics in EVAL mode (dropout OFF) ---
            tr_loss, tr_acc = _eval_epoch(model, train_eval_loader)
            va_loss, va_acc = _eval_epoch(model, val_loader)
            sch.step(va_loss if not math.isnan(va_loss) else tr_loss)

            # Log eval-mode train metrics (stable) and validation metrics
            hist["loss"].append(tr_loss)
            hist["accuracy"].append(tr_acc)
            hist["val_loss"].append(va_loss)
            hist["val_accuracy"].append(va_acc)

            print(
                f"[{name}] epoch {ep:02d}/{epochs}  "
                f"train(eval loss={tr_loss:.4f} acc={tr_acc:.4f})  "
                f"val(loss={va_loss:.4f} acc={va_acc:.4f})"
            )

            # Track best by validation loss; snapshot weights (no early stop—unless configured)
            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if early_pat is not None and no_improve >= int(early_pat):
                    print(f"[{name}] Early stop at epoch {ep} (patience={early_pat})")
                    break

        # Restore best parameters before returning
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, hist

    # ---------- Train selected models based on CFG flags ----------
    histories: dict[str, dict] = {}
    deep_models_raw: dict[str, torch.nn.Module] = {}
    model_dir = Path(C("RESULTS_DIR", Path("results"))) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, flag in [
        ("cnn",  C("RUN_TORCH_CNN",  True)),
        ("rnn",  C("RUN_TORCH_RNN",  True)),
        ("lstm", C("RUN_TORCH_LSTM", True)),
        ("ann",  C("RUN_TORCH_ANN",  True)),
    ]:
        if not flag:
            print(f"[Train] Skipped {name.upper()} (flag off)")
            continue
        print(f"\n[Train] === {name.upper()} ===")
        m, h = train_model_torch(name)
        tag = f"{name.upper()} (PyTorch)"
        deep_models_raw[tag] = m
        histories[tag] = h

        # Optionally save best weights per model
        if C("SAVE_BEST_WEIGHTS", True):
            fp = model_dir / f"best_{name.lower()}.pt"
            torch.save(m.state_dict(), fp)
            print("Saved weights:", fp)

        # Also save training history as CSV for reproducibility
        hist_csv = model_dir / f"history_{name.lower()}.csv"
        with open(hist_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "loss", "val_loss", "accuracy", "val_accuracy"])
            for i in range(len(h.get("loss", []))):
                w.writerow([i + 1, h["loss"][i], h["val_loss"][i], h["accuracy"][i], h["val_accuracy"][i]])
        print("Saved history:", hist_csv)

    # ---------- Evaluate on TEST (unified interface via TorchAdapter) ----------
    try:
        adapters = {
            name: TorchAdapter(m, le, is_binary, batch=C("BATCH_SIZE", 24), device=device)
            for name, m in deep_models_raw.items()
        }
    except TypeError:
        adapters = {
            name: TorchAdapter(m, le, is_binary, batch=C("BATCH_SIZE", 24))
            for name, m in deep_models_raw.items()
        }

    unified_df = evaluate_models(adapters, test_paths, y_te, le, is_binary, device)
    print("\nUnified evaluation:")
    print(unified_df)

    # Persist evaluation table
    results_csv = Path(C("RESULTS_DIR", Path("results"))) / "centralized_eval.csv"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    unified_df.to_csv(results_csv, index=False)
    print("Saved evaluation:", results_csv)

    # ---------- Visualization outputs ----------
    viz_dir = Path(C("RESULTS_DIR", Path("results"))) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Confusion matrix for top entry (if any)
    if not unified_df.empty:
        best_name = str(unified_df.iloc[0]["model"])
        best_adapter = adapters[best_name]
        y_pred = best_adapter.predict(test_paths)
        plot_confusion(
            y_te, y_pred,
            title=f"Confusion — {best_name}",
            save_path=str(viz_dir / f"confusion_{best_name.replace(' ', '_')}.png"),
            collapse_to_3=C("CONFUSION_COLLAPSE_TO_3", False),
        )

    # Learning curves (loss/acc) per model
    plot_learning_curves(histories, out_dir=str(viz_dir))
    print("\n[Done] Centralized run complete.")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    """CLI/Script entry: run centralized deep models with CFG.SEED default."""
    run_deep_models(seed=C("SEED", 42))


if __name__ == "__main__":
    main()
