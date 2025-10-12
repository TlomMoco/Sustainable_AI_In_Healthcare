"""Centralized.py — Baselines + Deep Models (centralized training)"""

from __future__ import annotations

# --- Standard / third-party
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

# --- Project imports (use absolute package imports consistently)
from src_Connection import config as CFG
from src_Connection.utils import (
    set_seed,
    pick_device,
    log,
    sanitize_mps_env,
    torch_loader_kwargs,   # NEW: for a stable train-eval DataLoader
)
from src_Connection.data_loader import (
    load_metadata,
    map_superclasses,
    filter_single_label,
    stratified_patient_split,
    make_feature_table_legacy,                  # <- legacy: (X, y, classes)
    make_feature_table as build_feature_tables, # <- deep: (feature_df, features_df)
    train_test_split as mask_split,             # boolean-mask split for deep set
)
from src_Connection.data_preprocessing import (
    make_label_encoder,
    make_train_val_loaders,
    ECGDataset,                                 # NEW: to build a train-eval loader
)
from src_Connection.models import create_model, create_logistic_baseline
from src_Connection.results_visualization import (
    TorchAdapter,
    evaluate_models,
    plot_confusion,
    plot_learning_curves,
)


# =============================================================================
# Helper to read config (supports top-level vars or NOTEBOOK[...] fallbacks)
# =============================================================================
def C(name: str, default=None):
    """Return CFG.<name> if present, else CFG.NOTEBOOK[name], else default."""
    return getattr(CFG, name, CFG.NOTEBOOK.get(name, default))


# =============================================================================
# 1) Quick ML baselines on engineered features (LogReg, RandomForest)
# =============================================================================
def run_quick_ml_baselines(seed: int | None = None) -> None:
    seed = int(seed if seed is not None else C("SEED", 42))
    set_seed(seed)

    # Load + map labels, ensure 1 label per row, patient-safe split
    ptb = load_metadata()
    df = filter_single_label(map_superclasses(ptb))
    train_df, test_df = stratified_patient_split(df, test_size=0.2, seed=seed)

    # Build simple per-record feature tables (LEGACY API)
    Xtr, ytr, classes = make_feature_table_legacy(train_df)
    Xte, yte, _ = make_feature_table_legacy(test_df)

    # Baseline models
    logreg = create_logistic_baseline().fit(Xtr, ytr)

    rf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ]).fit(Xtr, ytr)

    for name, clf in {"LogReg": logreg, "RF": rf}.items():
        ypred = clf.predict(Xte)
        print(f"\n== {name} ==")
        print(classification_report(yte, ypred, target_names=classes, digits=3, zero_division=0))


# =============================================================================
# 2) Deep learning pipeline (CNN/RNN/LSTM/ANN in PyTorch)
# =============================================================================
def run_deep_models(seed: int | None = None) -> None:
    seed = int(seed if seed is not None else C("SEED", 42))
    set_seed(seed)
    sanitize_mps_env()
    device = pick_device()
    log(f"[Torch] device={device}")

    # Build engineered feature table + minimal meta (DEEP API)
    feature_df, features_df = build_feature_tables(save_csv=True)

    # Boolean masks for train/test (patient/leakage-safe)
    train_mask, test_mask = mask_split(features_df)

    # Dummy baseline on engineered features (sanity check)
    X_tr = (
        feature_df
        .loc[feature_df.index.intersection(features_df.index[train_mask])]
        .drop(columns=["label"], errors="ignore")
        .select_dtypes(include=[np.number])
    )
    y_tr_b = feature_df.loc[X_tr.index, "label"].astype(str)

    X_te = (
        feature_df
        .loc[feature_df.index.intersection(features_df.index[test_mask])]
        .drop(columns=["label"], errors="ignore")
        .select_dtypes(include=[np.number])
    )
    y_te_b = feature_df.loc[X_te.index, "label"].astype(str)

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler(with_mean=False)),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])
    pipe.fit(X_tr, y_tr_b)
    y_pred_dummy = pipe.predict(X_te)
    print("\n[Baseline Dummy]")
    print(classification_report(y_te_b, y_pred_dummy, digits=3, zero_division=0))

    # Paths + labels for deep models
    train_paths = features_df.loc[train_mask, "record_path"].astype(str).values
    test_paths  = features_df.loc[test_mask,  "record_path"].astype(str).values
    y_tr_series = features_df.loc[train_mask, "label"].astype(str)
    y_te_series = features_df.loc[test_mask,  "label"].astype(str)

    le = make_label_encoder(y_tr_series, y_te_series)
    y_tr = le.transform(y_tr_series.values)
    y_te = y_te_series.values
    n_classes = len(le.classes_)
    is_binary = (n_classes == 2)

    # DataLoaders (increase val_frac for steadier validation)
    train_loader, val_loader, tr_pos, va_pos = make_train_val_loaders(
        train_paths, y_tr, device=device, val_frac=0.20
    )

    # Stable train metrics: eval-mode loader (non-shuffled) on the train split
    train_eval_ds = ECGDataset(train_paths[tr_pos], y_tr[tr_pos])
    train_eval_loader = DataLoader(
        train_eval_ds, **torch_loader_kwargs(False, C("BATCH_SIZE", 24), device.type)
    )

    def _criterion():
        return nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()

    @torch.no_grad()
    def _eval_epoch(model, loader):
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
        return (loss_sum / total if total else math.nan), (correct / total if total else math.nan)

    def train_model_torch(name: str, epochs: int | None = None):
        epochs = int(epochs if epochs is not None else C("EPOCHS", 12))
        model = create_model(name, n_classes, binary=is_binary).to(device)
        base_lr = C("RECURRENT_LR", 2e-4) if name.lower() in {"rnn", "lstm", "ann"} else C("BASE_LR", 3e-4)
        opt = optim.Adam(model.parameters(), lr=base_lr)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-5)
        crit = _criterion()

        hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        best_state = None
        best_val = float("inf")

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

            hist["loss"].append(tr_loss)           # eval-mode train loss
            hist["accuracy"].append(tr_acc)        # eval-mode train acc
            hist["val_loss"].append(va_loss)
            hist["val_accuracy"].append(va_acc)

            print(
                f"[{name}] epoch {ep:02d}/{epochs}  "
                f"train(loss={train_loss_trainmode:.4f}→eval {tr_loss:.4f} acc={tr_acc:.4f})  "
                f"val(loss={va_loss:.4f} acc={va_acc:.4f})"
            )

            if va_loss < best_val:
                best_val = va_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        return model, hist

    # Train selected models based on CFG flags
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
        deep_models_raw[f"{name.upper()} (PyTorch)"] = m
        histories[f"{name.UPPER()} (PyTorch)"] = h if False else histories.setdefault(f"{name.upper()} (PyTorch)", h)  # keep order

    # Evaluate on TEST
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

    # Plots
    viz_dir = C("RESULTS_DIR", Path("results")) / "viz"
    Path(viz_dir).mkdir(parents=True, exist_ok=True)

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

    plot_learning_curves(histories, out_dir=str(viz_dir))
    print("\n[Done] Centralized run complete.")


# =============================================================================
# Entry point
# =============================================================================
def main():
    # Run both phases; comment either out if you want to skip one.
    run_quick_ml_baselines(seed=C("SEED", 42))
    run_deep_models(seed=C("SEED", 42))


if __name__ == "__main__":
    main()