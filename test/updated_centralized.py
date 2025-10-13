"""
Centralized.py — Centralized Training (FL-Compatible)
------------------------------------------------------

Purpose
-------
Run centralized (non-federated) training that exactly mirrors the FL client setup
for direct comparison of centralized vs federated performance.

Key Design
----------
• Uses SAME models as FL clients (CNN/LSTM/RNN/ANN via create_model)
• Uses SAME data pipeline as FL clients (load_waveform → normalize → transpose)
• Uses SAME hyperparameters as FL clients (with optional tuning)
• Uses SAME training loop as FL clients (class-weighted CE, Adam, optional FedProx)
• Logs results in comparable format to FL runs

Outputs
-------
• CSV: results/centralized_{model_type}_{phase}.csv (metrics per epoch)
• Plots: results/viz/CentralizedTraining{MODEL_TYPE}/*.png
• Best weights: results/models/centralized_best_{model_type}.pt
• CV results: results/tuning/centralized_cv.csv (if tuning enabled)
"""

from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    SEED, RESULTS_DIR, MODEL, SAMPLE_RATE, N_CLASSES, SUPERCLASSES,
    LR, BATCH_SIZE, EPOCHS_LOCAL, FEDPROX_MU, SPLITS, NORM,
    TUNING, GRIDSEARCH,
)
from src.data_loader import (
    load_metadata, map_superclasses, filter_single_label,
    stratified_patient_split_3way, load_waveform,
    compute_perlead_norm_stats, normalize_signal
)
from src.models import create_model
from src.utils import set_seed, pick_device, sanitize_mps_env, ensure_dir, torch_loader_kwargs


# -------------------------------------------------------------------------
# Data Pipeline (SAME as FL Client)
# -------------------------------------------------------------------------
CLASSES = list(SUPERCLASSES)

def make_tensor_dataset(df, mu=None, sigma=None):
    """Convert PTB-XL dataframe to tensors (SAME as Client.py)."""
    X, y = [], []
    for _, row in df.iterrows():
        sig = load_waveform(row)  # (12, T)
        if mu is not None and sigma is not None and NORM["enabled"]:
            sig = normalize_signal(sig, mu, sigma, eps=NORM["eps"])
        sig = sig.T.astype(np.float32, copy=False)  # -> (T, 12)
        X.append(sig)
        y.append(CLASSES.index(row["y"]))
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)
    return torch.utils.data.TensorDataset(X, y)


def get_loader(df, batch_size, mu=None, sigma=None, shuffle=True, device_type="cpu"):
    """Build DataLoader (SAME as Client.py)."""
    ds = make_tensor_dataset(df, mu, sigma)
    return DataLoader(ds, **torch_loader_kwargs(shuffle, batch_size, device_type))


# -------------------------------------------------------------------------
# Hyperparameter Tuning (Optional CV)
# -------------------------------------------------------------------------
def run_centralized_cv(train_df, mu, sigma, device):
    """
    Run GroupKFold CV on centralized training data.
    Mirrors src.tuning.run_client_cv but for centralized setup.
    """
    from sklearn.model_selection import GroupKFold
    
    if not TUNING.get("enabled", False):
        return None
    
    k = int(GRIDSEARCH.get("cv", 5))
    grid = GRIDSEARCH.get("grid", [])
    
    if not grid:
        print("[Centralized CV] No grid provided, skipping CV.")
        return None
    
    # Patient-aware splits
    idx = np.arange(len(train_df))
    groups = train_df["patient_id"].values
    labels = train_df["y"].values
    
    unique_groups = int(np.unique(groups).size)
    eff_k = min(k, unique_groups)
    
    if eff_k < 2:
        print(f"[Centralized CV] Not enough patients ({unique_groups}) for {k}-fold CV.")
        return None
    
    print(f"[Centralized CV] Running {eff_k}-fold CV with {len(grid)} hyperparameter sets...")
    
    gkf = GroupKFold(n_splits=eff_k)
    rows, best_score, best_hp = [], -1.0, None
    
    for trial, hp in enumerate(grid, start=1):
        fold_accs = []
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(idx, groups=groups, y=labels), 1):
            tr_df = train_df.iloc[tr_idx]
            va_df = train_df.iloc[va_idx]
            
            # Compute fold-specific normalization
            mu_fold, sigma_fold = compute_perlead_norm_stats(tr_df, sampling_rate=SAMPLE_RATE)
            
            tr_loader = get_loader(tr_df, int(hp["batch"]), mu_fold, sigma_fold, shuffle=True, device_type=device.type)
            va_loader = get_loader(va_df, 128, mu_fold, sigma_fold, shuffle=False, device_type=device.type)
            
            # Train small model
            model = create_model(
                n_classes=N_CLASSES,
                model_type=MODEL["type"],
                hidden=MODEL.get("lstm_hidden", 128),
                layers=MODEL.get("lstm_layers", 1),
                bidir=MODEL.get("bidirectional", True),
            ).to(device)
            
            ce = nn.CrossEntropyLoss()
            opt = optim.Adam(model.parameters(), lr=float(hp["lr"]), weight_decay=1e-4)
            
            # Train for hp['epochs']
            model.train()
            for _ in range(int(hp["epochs"])):
                for xb, yb in tr_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    opt.zero_grad()
                    logits = model(xb)
                    loss = ce(logits, yb)
                    loss.backward()
                    opt.step()
            
            # Validate
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    correct += int((logits.argmax(1) == yb).sum())
                    total += len(yb)
            
            acc = (correct / total) if total else 0.0
            fold_accs.append(acc)
            print(f"  Trial {trial}/{len(grid)}, Fold {fold}/{eff_k}: acc={acc:.4f}")
        
        mean_acc = float(np.mean(fold_accs))
        rows.append({"trial": trial, **hp, "mean_val_acc": mean_acc})
        
        if mean_acc > best_score:
            best_score, best_hp = mean_acc, hp
    
    # Save CV results
    import pandas as pd
    cv_df = pd.DataFrame(rows)
    cv_path = RESULTS_DIR / "tuning" / "centralized_cv.csv"
    ensure_dir(cv_path.parent)
    cv_df.to_csv(cv_path, index=False)
    print(f"[Centralized CV] Saved results to {cv_path}")
    print(f"[Centralized CV] Best: {best_hp} (acc={best_score:.4f})")
    
    return best_hp


# -------------------------------------------------------------------------
# Training Loop (SAME structure as FL Client)
# -------------------------------------------------------------------------
def train_centralized(train_df, val_df, test_df, hyperparams, device):
    """
    Train centralized model with SAME setup as FL client.
    
    Returns:
        model, history
    """
    # Unpack hyperparameters
    lr = float(hyperparams.get("lr", LR))
    batch_size = int(hyperparams.get("batch_size", BATCH_SIZE))
    epochs = int(hyperparams.get("epochs", EPOCHS_LOCAL * 3))  # More epochs for centralized
    fedprox_mu = float(hyperparams.get("fedprox", FEDPROX_MU))
    
    # Compute normalization stats on FULL training set
    mu, sigma = compute_perlead_norm_stats(train_df, sampling_rate=SAMPLE_RATE)
    
    # Create loaders
    train_loader = get_loader(train_df, batch_size, mu, sigma, shuffle=True, device_type=device.type)
    val_loader = get_loader(val_df, 128, mu, sigma, shuffle=False, device_type=device.type)
    test_loader = get_loader(test_df, 128, mu, sigma, shuffle=False, device_type=device.type)
    
    # Create model (SAME as FL client)
    model = create_model(
        n_classes=N_CLASSES,
        model_type=MODEL["type"],
        hidden=MODEL.get("lstm_hidden", 128),
        layers=MODEL.get("lstm_layers", 1),
        bidir=MODEL.get("bidirectional", True),
    ).to(device)
    
    print(f"\n[Centralized] Model: {MODEL['type'].upper()}")
    print(f"[Centralized] Hyperparameters: lr={lr}, batch={batch_size}, epochs={epochs}")
    
    # Class-weighted loss (SAME as FL client)
    counts_series = train_df["y"].value_counts()
    counts = np.array([int(counts_series.get(cname, 0)) for cname in CLASSES], dtype=np.int64)
    
    if counts.sum() > 0:
        total = counts.sum()
        w = np.zeros_like(counts, dtype=np.float32)
        for i, c in enumerate(counts):
            if c > 0:
                w[i] = total / (N_CLASSES * float(c))
        present = w > 0
        if present.any():
            w[present] *= (present.sum() / float(w[present].sum()))
    else:
        w = np.ones(N_CLASSES, dtype=np.float32)
    
    class_weights = torch.tensor(w, dtype=torch.float32, device=device)
    ce_train = nn.CrossEntropyLoss(weight=class_weights)
    ce_eval = nn.CrossEntropyLoss()
    
    print(f"[Centralized] Class weights: {dict(zip(CLASSES, w.round(3)))}")
    
    # Optimizer (SAME as FL client)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # FedProx snapshot (optional)
    global_params = [p.detach().clone() for p in model.parameters()] if fedprox_mu > 0 else None
    
    # Training history
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "wall_time": []}
    best_state, best_val_acc = None, -1.0
    
    # Training loop
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = ce_train(logits, yb)
            
            # FedProx term (optional)
            if fedprox_mu > 0 and global_params is not None:
                prox = sum(torch.sum((w - w0) ** 2) for w, w0 in zip(model.parameters(), global_params))
                loss = loss + (fedprox_mu / 2.0) * prox
            
            loss.backward()
            opt.step()
            
            train_loss += float(loss.item()) * len(yb)
            train_correct += int((logits.argmax(1) == yb).sum())
            train_total += len(yb)
        
        train_loss = train_loss / train_total if train_total else 0.0
        train_acc = train_correct / train_total if train_total else 0.0
        
        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = ce_eval(logits, yb)
                val_loss += float(loss.item()) * len(yb)
                val_correct += int((logits.argmax(1) == yb).sum())
                val_total += len(yb)
        
        val_loss = val_loss / val_total if val_total else 0.0
        val_acc = val_correct / val_total if val_total else 0.0
        
        wall_time = time.perf_counter() - t0
        
        # Track best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        
        # Log
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["wall_time"].append(wall_time)
        
        print(f"  Epoch {epoch:02d}/{epochs}: "
              f"train(loss={train_loss:.4f}, acc={train_acc:.4f}), "
              f"val(loss={val_loss:.4f}, acc={val_acc:.4f}), "
              f"time={wall_time:.1f}s")
    
    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Centralized] Restored best model (val_acc={best_val_acc:.4f})")
    
    return model, history, (mu, sigma)


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------
def evaluate_model(model, test_df, mu, sigma, device):
    """Evaluate model on test set and return metrics."""
    test_loader = get_loader(test_df, 128, mu, sigma, shuffle=False, device_type=device.type)
    
    model.eval()
    y_true, y_pred = [], []
    test_loss, test_correct, test_total = 0.0, 0, 0
    
    ce = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = ce(logits, yb)
            test_loss += float(loss.item()) * len(yb)
            pred = logits.argmax(1)
            test_correct += int((pred == yb).sum())
            test_total += len(yb)
            
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    
    test_loss = test_loss / test_total if test_total else 0.0
    test_acc = test_correct / test_total if test_total else 0.0
    
    # Classification report
    print("\n" + "="*60)
    print(f"TEST RESULTS - {MODEL['type'].upper()}")
    print("="*60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
        "confusion_matrix": cm
    }


# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
def save_results(history, test_results, hyperparams):
    """Save training history and test results to CSV."""
    # Determine phase
    if TUNING.get("log_phase"):
        if TUNING.get("enabled"):
            phase = TUNING["phase_labels"]["enabled"]
        elif TUNING.get("use_cached_best"):
            phase = TUNING["phase_labels"]["cached"]
        else:
            phase = TUNING["phase_labels"]["disabled"]
    else:
        phase = ""
    
    model_type = MODEL["type"].lower()
    
    # Save training history
    if TUNING.get("log_mode") == "separate" and phase:
        csv_path = RESULTS_DIR / f"centralized_{model_type}_{phase}.csv"
    else:
        csv_path = RESULTS_DIR / f"centralized_{model_type}.csv"
    
    ensure_dir(csv_path.parent)
    
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "wall_time"]
        if TUNING.get("log_phase") and TUNING.get("log_mode") != "separate":
            fieldnames.append("phase")
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(len(history["epoch"])):
            row = {
                "epoch": history["epoch"][i],
                "train_loss": f"{history['train_loss'][i]:.6f}",
                "train_acc": f"{history['train_acc'][i]:.4f}",
                "val_loss": f"{history['val_loss'][i]:.6f}",
                "val_acc": f"{history['val_acc'][i]:.4f}",
                "wall_time": f"{history['wall_time'][i]:.2f}",
            }
            if "phase" in fieldnames:
                row["phase"] = phase
            writer.writerow(row)
    
    print(f"\n[Centralized] Saved training history to {csv_path}")
    
    # Save test results
    test_csv = RESULTS_DIR / f"centralized_{model_type}_test.csv"
    with open(test_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["test_loss", f"{test_results['test_loss']:.6f}"])
        writer.writerow(["test_acc", f"{test_results['test_acc']:.4f}"])
    
    print(f"[Centralized] Saved test results to {test_csv}")


# -------------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------------
def plot_results(history, test_results):
    """Generate plots for training curves and confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    model_type = MODEL["type"].upper()
    out_dir = RESULTS_DIR / "viz" / f"CentralizedTraining{model_type}"
    ensure_dir(out_dir)
    
    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", marker='o')
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", marker='s')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{model_type} - Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history["epoch"], history["train_acc"], label="Train Acc", marker='o')
    axes[1].plot(history["epoch"], history["val_acc"], label="Val Acc", marker='s')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"{model_type} - Training Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curve_path = out_dir / "training_curves.png"
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Centralized] Saved training curves to {curve_path}")
    
    # 2. Confusion Matrix
    cm = test_results["confusion_matrix"]
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_normalized, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    
    # Annotate cells
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            text = ax.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                          ha="center", va="center",
                          color="white" if cm_normalized[i, j] > 0.5 else "black",
                          fontsize=9)
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_type} - Confusion Matrix (Test Set)")
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized Count")
    
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Centralized] Saved confusion matrix to {cm_path}")


# -------------------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------------------
def main():
    """Run centralized training pipeline."""
    print("="*60)
    print("CENTRALIZED TRAINING (FL-Compatible)")
    print("="*60)
    
    # 1. Setup
    set_seed(SEED)
    sanitize_mps_env()
    device = pick_device()
    print(f"[Centralized] Device: {device}")
    print(f"[Centralized] Model: {MODEL['type'].upper()}")
    
    # 2. Load data (SAME as FL client)
    print("\n[Centralized] Loading data...")
    ptb = load_metadata()
    df = filter_single_label(map_superclasses(ptb))
    
    train_df, val_df, test_df = stratified_patient_split_3way(
        df,
        splits=(SPLITS["train"], SPLITS["val"], SPLITS["test"]),
        seed=SEED,
    )
    
    print(f"[Centralized] Train: {len(train_df)} records")
    print(f"[Centralized] Val: {len(val_df)} records")
    print(f"[Centralized] Test: {len(test_df)} records")
    
    # 3. Optional: Hyperparameter tuning
    hyperparams = {
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS_LOCAL * 3,  # More epochs for centralized
        "fedprox": FEDPROX_MU,
    }
    
    # Compute normalization stats for CV
    mu, sigma = compute_perlead_norm_stats(train_df, sampling_rate=SAMPLE_RATE)
    
    if TUNING.get("enabled", False):
        print("\n[Centralized] Running hyperparameter tuning...")
        best_hp = run_centralized_cv(train_df, mu, sigma, device)
        if best_hp:
            hyperparams.update(best_hp)
            print(f"[Centralized] Using tuned hyperparameters: {best_hp}")
    
    # 4. Train
    print("\n[Centralized] Training model...")
    model, history, (mu, sigma) = train_centralized(train_df, val_df, test_df, hyperparams, device)
    
    # 5. Save best weights
    model_dir = RESULTS_DIR / "models"
    ensure_dir(model_dir)
    weights_path = model_dir / f"centralized_best_{MODEL['type'].lower()}.pt"
    torch.save(model.state_dict(), weights_path)
    print(f"\n[Centralized] Saved best weights to {weights_path}")
    
    # 6. Evaluate
    print("\n[Centralized] Evaluating on test set...")
    test_results = evaluate_model(model, test_df, mu, sigma, device)
    
    # 7. Save results
    save_results(history, test_results, hyperparams)
    
    # 8. Visualize
    print("\n[Centralized] Generating plots...")
    plot_results(history, test_results)
    
    print("\n" + "="*60)
    print("CENTRALIZED TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
