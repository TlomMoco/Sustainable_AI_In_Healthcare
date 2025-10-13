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
• Prints unified evaluation DataFrame to stdout

Config keys referenced
----------------------
SEED, VAL_FRAC, BATCH_SIZE, EPOCHS,
BASE_LR, RECURRENT_LR, GRAD_CLIP_NORM,
RUN_TORCH_CNN, RUN_TORCH_RNN, RUN_TORCH_LSTM, RUN_TORCH_ANN,
CONFUSION_COLLAPSE_TO_3, RESULTS_DIR

Assumptions
-----------
• create_model(name, n_classes, binary=...) returns logits shaped:
    - binary: (N,) or (N,1) usable with BCEWithLogitsLoss
    - multi : (N, C) usable with CrossEntropyLoss
• ECGDataset yields (xb, yb) tensors compatible with the chosen loss.
• TorchAdapter signature may vary (device optional) -> handled via try/except.
"""

from __future__ import annotations

# --- Standard / third-party
import math
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# --- Project imports
from src_Connection import config as CFG
from src_Connection.utils import (
    set_seed,
    pick_device,
    log,
    sanitize_mps_env,
    torch_loader_kwargs,   # stable non-shuffled eval loaders (no dropout/bn randomness)
)
from src_Connection.data_loader import (
    make_feature_table as build_feature_tables,  # returns (feature_df, features_df)
    train_test_split as mask_split,              # boolean-mask split for deep set
)
from src_Connection.data_preprocessing import (
    make_label_encoder,
    make_train_val_loaders,
    ECGDataset,                                  # to build a train-eval loader
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
    """Get a config value with a safe fallback.

    Tries, in order:
        1) CFG.<name> (top-level attributes in config.py)
        2) CFG.NOTEBOOK[<name>] (if running under a notebook that injects values)
        3) provided `default`

    Why this exists:
        Keeps this module runnable from both notebooks and plain scripts
        without hard-coding import order or requiring every key to exist.

    Parameters
    ----------
    name : str
        Config key to look up.
    default : Any
        Value to return if not found.

    Returns
    -------
    Any
        The resolved config value.
    """
    return getattr(CFG, name, CFG.NOTEBOOK.get(name, default))


# -----------------------------------------------------------------------------
# Deep learning pipeline (CNN/RNN/LSTM/ANN in PyTorch)
# -----------------------------------------------------------------------------
def run_deep_models(seed: int | None = None) -> None:
    """Orchestrate centralized training & evaluation for selected deep models.

    Steps performed
    ---------------
    • Seed & device init (CPU/CUDA/MPS)                       -> utils
    • Build engineered features & meta (DEEP subset)          -> data_loader.make_feature_table
    • Patient-safe train/test split                           -> data_loader.train_test_split
    • Fit label encoder on train; apply to test               -> data_preprocessing.make_label_encoder
    • Build train/val loaders + stable train-eval loader      -> data_preprocessing & utils
    • Train models (per CFG flags)                            -> models.create_model
    • Evaluate via unified adapter & produce metrics table    -> results_visualization.evaluate_models
    • Save confusion matrix & learning curves                 -> results_visualization

    Parameters
    ----------
    seed : int | None
        Optional seed override; defaults to C("SEED", 42).

    Returns
    -------
    None
    """
    # ---------- Reproducibility & device selection ----------
    seed = int(seed if seed is not None else C("SEED", 42))
    set_seed(seed)              # utils.set_seed: seeds Python/NumPy/Torch, etc.
    sanitize_mps_env()          # utils.sanitize_mps_env: fixes MPS quirks if present
    device = pick_device()      # utils.pick_device: selects 'cuda'/'mps'/'cpu'
    log(f"[Torch] device={device}")

    # ---------- Feature tables & split ----------
    # Build engineered feature table + minimal meta (DEEP API).
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
    # Slightly larger val_frac helps stabilize validation curves on smallish sets.
    train_loader, val_loader, tr_pos, _ = make_train_val_loaders(
        train_paths, y_tr, device=device, val_frac=C("VAL_FRAC", 0.20)
    )

    # Build a *stable* (non-shuffled) eval loader on the *train subset*
    # to compute train metrics in eval mode (dropout/bn OFF).
    train_eval_ds = ECGDataset(train_paths[tr_pos], y_tr[tr_pos])
    train_eval_loader = DataLoader(
        train_eval_ds, **torch_loader_kwargs(False, C("BATCH_SIZE", 24), device.type)
    )

    # ---------- Loss selection ----------
    def _criterion():
        """Select appropriate loss function for binary vs multi-class tasks."""
        return nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    # ---------- Evaluation helper (single pass) ----------
    @torch.no_grad()
    def _eval_epoch(model, loader):
        """Run evaluation pass over `loader` and compute loss & accuracy.

        • Puts model in eval() to disable dropout/batchnorm updates.
        • For binary: applies sigmoid + 0.5 threshold for class prediction.
        • For multi-class: argmax over logits.

        Returns
        -------
        (loss_mean: float, acc: float)
        """
        model.eval()
        crit = _criterion()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            # BCE expects float targets; CE expects class indices.
            loss = crit(logits.view(-1), yb.float()) if is_binary else crit(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits.view(-1)) >= 0.5).long() if is_binary else logits.argmax(1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
        return (loss_sum / total if total else math.nan), (correct / total if total else math.nan)

    # ---------- Train a single model type ----------
    def train_model_torch(name: str, epochs: int | None = None):
        """Train one model (CNN/RNN/LSTM/ANN) and return (model, history).

        Parameters
        ----------
        name : {"cnn","rnn","lstm","ann"}
            Model key understood by models.create_model.
        epochs : int | None
            Number of epochs; defaults to C("EPOCHS", 12).

        Training details
        ----------------
        • Optimizer: Adam
        • LR schedule: ReduceLROnPlateau on val_loss (min mode)
        • Optional gradient clipping via CFG.GRAD_CLIP_NORM
        • Tracks *eval-mode* train loss/acc (via stable train_eval_loader)
          and val loss/acc each epoch.
        • Keeps the best snapshot by lowest val_loss (no early stop—just best state).

        Returns
        -------
        model : torch.nn.Module
            Best-scoring model (by val_loss).
        hist : dict[str, list[float]]
            History with keys: loss, val_loss, accuracy, val_accuracy.
        """
        epochs = int(epochs if epochs is not None else C("EPOCHS", 12))
        model = create_model(name, n_classes, binary=is_binary).to(device)

        # Smaller LR for recurrent/ANN models (empirical stability).
        base_lr = C("RECURRENT_LR", 2e-4) if name.lower() in {"rnn", "lstm", "ann"} else C("BASE_LR", 3e-4)
        opt = optim.Adam(model.parameters(), lr=base_lr)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-5)
        crit = _criterion()

        hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        best_state, best_val = None, float("inf")

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

            train_loss_trainmode = ep_loss / max(1, ep_total)

            # --- metrics in EVAL mode (dropout OFF) ---
            tr_loss, tr_acc = _eval_epoch(model, train_eval_loader)
            va_loss, va_acc = _eval_epoch(model, val_loader)
            sch.step(va_loss if not math.isnan(va_loss) else tr_loss)

            # Log eval-mode train metrics (stable) and validation metrics
            hist["loss"].append(tr_loss)           # eval-mode train loss
            hist["accuracy"].append(tr_acc)        # eval-mode train acc
            hist["val_loss"].append(va_loss)
            hist["val_accuracy"].append(va_acc)

            print(
                f"[{name}] epoch {ep:02d}/{epochs}  "
                f"train(loss={train_loss_trainmode:.4f}→eval {tr_loss:.4f} acc={tr_acc:.4f})  "
                f"val(loss={va_loss:.4f} acc={va_acc:.4f})"
            )

            # Track best by validation loss; snapshot weights (no early stop)
            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        # Restore best parameters before returning
        if best_state is not None:
            model.load_state_dict(best_state)
        return model, hist

    # ---------- Train selected models based on CFG flags ----------
    histories: dict[str, dict] = {}
    deep_models_raw: dict[str, torch.nn.Module] = {}
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
        tag = f"{name.UPPER()} (PyTorch)" if hasattr(name, "UPPER") else f"{name.upper()} (PyTorch)"
        deep_models_raw[tag] = m
        histories[tag] = h

    # ---------- Evaluate on TEST (unified interface via TorchAdapter) ----------
    # TorchAdapter signature varies across versions (device may be optional).
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

    # ---------- Visualization outputs ----------
    viz_dir = C("RESULTS_DIR", Path("results")) / "viz"
    Path(viz_dir).mkdir(parents=True, exist_ok=True)

    # Confusion matrix for top entry (if any)
    if not unified_df.empty:
        best_name = str(unified_df.iloc[0]["model"])
        best_adapter = adapters[best_name]
        y_pred = best_adapter.predict(test_paths)
        plot_confusion(
            y_te, y_pred,
            title=f"Confusion — {best_name}",
            save_path=str(Path(viz_dir) / f"confusion_{best_name.replace(' ', '_')}.png"),
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