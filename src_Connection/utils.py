"""
utils.py — General Utilities
----------------------------

Helper functions shared across modules in the Sustainable AI in Healthcare (DSP5100) project:
  • Reproducibility: seed setting across Python/NumPy/PyTorch
  • Logging helpers
  • File/dir helpers
  • Plotting of 12-lead ECG signals
  • Basic metrics (acc/F1/AUC) for quick evals
  • Dataset summaries (patients/records/time)
  • Device/DataLoader utilities
  • Lightweight model reporting utilities
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict

import os
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src_Connection.config import RESULTS_DIR, SEED, LOW_RAM


# -------------------------------------------------------------------------
# 0) Logging / timestamps
# -------------------------------------------------------------------------
def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{ts()}] {msg}")


# -------------------------------------------------------------------------
# 1) Reproducibility
# -------------------------------------------------------------------------
def set_seed(seed: int | None = None) -> None:
    """
    Set global random seed across Python, NumPy, and PyTorch.

    Notes:
      • Makes CUDA/CuDNN deterministic (slightly slower but reproducible).
      • If seed is None, uses SEED from config.
    """
    s = int(SEED if seed is None else seed)
    random.seed(s)
    np.random.seed(s)
    try:
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        # Deterministic cuDNN for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    log(f"Seeds set (seed={s}).")


def sanitize_mps_env() -> None:
    """
    Set sensible defaults for Apple MPS memory watermarks to reduce OOMs.
    Safe on non-MPS systems (no-ops).
    """
    hi = float(os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", 0.92))
    lo = float(os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO", 0.65))
    lo = max(0.05, min(lo, hi - 0.05))
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = f"{hi:.2f}"
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = f"{lo:.2f}"
    log(f"[MPS] LOW={lo:.2f}, HIGH={hi:.2f}")


def pick_device() -> torch.device:
    """
    Prefer MPS on Apple, then CUDA, else CPU.
    """
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            _ = torch.empty(1, device="mps")  # sanity check
            return torch.device("mps")
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -------------------------------------------------------------------------
# 2) Files / directories
# -------------------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    """Create directory (and parents) if it doesn’t exist."""
    p.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# 3) Plotting
# -------------------------------------------------------------------------
def plot_signal(sig: np.ndarray, title: str = "ECG (12xT)", save: str | None = None) -> None:
    """
    Plot a 12-lead ECG (shape: 12 x T) with vertical offsets for readability.
    If `save` is provided, stores the figure under RESULTS_DIR / save.
    """
    # sig: (12, T)
    plt.figure(figsize=(10, 6))
    offset = 2.5 * np.arange(sig.shape[0])
    for i in range(sig.shape[0]):
        plt.plot(sig[i] + offset[i])
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Leads (offset)")
    plt.tight_layout()
    if save:
        ensure_dir(RESULTS_DIR)
        plt.savefig(RESULTS_DIR / save, dpi=150)
    plt.close()


# -------------------------------------------------------------------------
# 4) Metrics
# -------------------------------------------------------------------------
def compute_metrics(y_true, y_prob, labels=None) -> Dict[str, float]:
    """
    Compute basic multi-class metrics from probabilities or logits:
      - accuracy
      - macro F1
      - one-vs-rest ROC-AUC (may be NaN if classes missing)

    Args:
        y_true: array-like of shape (n,)
        y_prob: array-like of shape (n, C) — class probabilities or logits
        labels: kept for backward compatibility (unused)
    """
    y_prob = np.asarray(y_prob)
    y_pred = y_prob.argmax(axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    try:
        # roc_auc_score handles logits; softmax not required
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "f1_macro": f1m, "roc_auc_ovr": auc}


# -------------------------------------------------------------------------
# 5) Dataset summary
# -------------------------------------------------------------------------
def summarize_dataset(df, sample_rate: int = 100, title: str = "Global Dataset") -> None:
    """
    Print a concise summary of a PTB-XL dataframe (after mapping/filtering).

    Args:
        df : DataFrame from load_metadata/map_superclasses/filter_single_label
        sample_rate : 100 or 500 (Hz)
        title : header for readability
    """
    print(f"\n===== {title} =====")
    n_records = len(df)
    n_patients = df.patient_id.nunique()
    recs_per_patient = df.groupby("patient_id").size()

    mean_rpp = recs_per_patient.mean()
    med_rpp = recs_per_patient.median()
    max_rpp = recs_per_patient.max()

    # PTB-XL records are 10 seconds each
    samples_per_record = 10 * sample_rate
    total_samples = n_records * samples_per_record
    hours = (total_samples / sample_rate) / 3600.0

    print(f"Records: {n_records:,}")
    print(f"Unique patients: {n_patients:,}")
    print(f"Records per patient – mean {mean_rpp:.2f}, median {med_rpp:.0f}, max {max_rpp}")
    print(f"Sampling rate: {sample_rate} Hz  →  {samples_per_record:,} samples/record")
    print(f"Total waveform samples: {total_samples:,}  ({hours:.1f} h of ECG data)")
    print("==============================\n")


# -------------------------------------------------------------------------
# 6) Small extras for reporting (optional)
# -------------------------------------------------------------------------
def format_time(seconds: float) -> str:
    """Format seconds as 'Xm Ys'."""
    minutes, sec = divmod(int(seconds), 60)
    return f"{minutes}m {sec}s"


def count_trainable_params(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module) -> None:
    """Print a compact list of trainable parameters per layer and the total."""
    total = count_trainable_params(model)
    print("\nModel Summary")
    print("-" * 40)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<35} {param.numel():>10,d}")
    print("-" * 40)
    print(f"Total trainable parameters: {total:,}\n")


# -------------------------------------------------------------------------
# 7) DataLoader convenience
# -------------------------------------------------------------------------
def torch_loader_kwargs(shuffle: bool, batch_size: int, device_type: str) -> dict:
    """
    Centralized DataLoader kwargs used across the codebase.

    • On low-RAM runs, keep workers at 0 (safer for notebooks).
    • Pin memory only on CUDA.
    """
    num_workers = 0 if LOW_RAM else 0  # adjust here if you later add multiprocessing
    pin = (device_type == "cuda")
    return dict(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=pin,
        drop_last=False,
    )
