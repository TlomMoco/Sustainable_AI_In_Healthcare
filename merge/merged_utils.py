"""
utils.py — General Utilities
----------------------------

Helper functions shared across modules in the Sustainable AI in Healthcare (DSP5100) project:
  • Reproducibility: seed setting across Python/NumPy/PyTorch
  • Logging helpers
  • File/dir helpers
  • Plotting of 12-lead ECG signals (lazy matplotlib import)
  • Basic metrics (acc/F1/AUC) for quick evals
  • Dataset summaries (patients/records/time)
  • Device/DataLoader utilities
  • Lightweight model reporting utilities
  • CSV append with optional file locking

Where this module connects
--------------------------
• Used by:
    - src.Centralized: set_seed, pick_device, log, torch_loader_kwargs
    - src.Client: set_seed, sanitize_mps_env, pick_device, ensure_dir, torch_loader_kwargs, append_csv_locked
    - src.Server: ensure_dir, append_csv_locked
    - src.Experiments: set_seed, log, ensure_dir, torch_loader_kwargs
    - src.tuning: ensure_dir, set_seed
    - src.results_visualization: ensure_dir (viz folder creation)
    - src.data_loader: (optional) summarize_dataset, plot_signal

Side effects / environment
--------------------------
• set_seed(): makes PyTorch/CuDNN deterministic (slower but reproducible).
• sanitize_mps_env(): sets Apple Metal (MPS) memory watermarks via env vars.
• plot_signal(): writes figures under config.RESULTS_DIR when `save` is provided.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import os
import csv
import time
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Lazy import for file locking (optional dependency)
try:
    import portalocker
except ImportError:
    portalocker = None

# Import from config for paths/settings
from src.config import RESULTS_DIR, SEED, LOW_RAM

__all__ = [
    "ts", "log",
    "set_seed", "sanitize_mps_env", "pick_device",
    "ensure_dir", "plot_signal",
    "compute_metrics", "summarize_dataset",
    "format_time", "count_trainable_params", "print_model_summary",
    "torch_loader_kwargs", "append_csv_locked",
]

# -------------------------------------------------------------------------
# 0) Logging / timestamps
# -------------------------------------------------------------------------
def ts() -> str:
    """Return current local timestamp as 'YYYY-MM-DD HH:MM:SS'."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    """Print a message prefixed with a timestamp (uniform project logging)."""
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
      • Called from Centralized/Client/Experiments prior to any randomness.

    Safe fallback:
      • If torch is not present/initialized, errors are swallowed and we still
        seed Python/NumPy.
    """
    s = int(SEED if seed is None else seed)
    random.seed(s)
    np.random.seed(s)
    try:
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        # Deterministic cuDNN for reproducibility (only if CUDA is built)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        # Keep going even if torch isn't fully available in this environment
        pass
    log(f"Seeds set (seed={s}).")


def sanitize_mps_env() -> None:
    """
    Set sensible defaults for Apple MPS memory watermarks to reduce OOMs.
    Safe on non-MPS systems (no-ops).

    Env vars (if unset we set conservative defaults):
      • PYTORCH_MPS_HIGH_WATERMARK_RATIO (default ~0.92)
      • PYTORCH_MPS_LOW_WATERMARK_RATIO  (default ~0.65)

    Used by:
      • Client and Centralized pipelines before tensor allocations.
    """
    try:
        hi = float(os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", 0.92))
        lo = float(os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO", 0.65))
    except Exception:
        hi, lo = 0.92, 0.65
    # Clamp to reasonable bounds and ensure lo < hi
    hi = max(0.10, min(hi, 0.99))
    lo = max(0.05, min(lo, hi - 0.05))
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = f"{hi:.2f}"
    os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = f"{lo:.2f}"
    log(f"[MPS] LOW={lo:.2f}, HIGH={hi:.2f}")


def pick_device() -> torch.device:
    """
    Prefer MPS on Apple, then CUDA, else CPU.

    Return
    ------
    torch.device('mps'|'cuda'|'cpu')

    Connected to:
      • Centralized and Client choose device via this helper to standardize
        device selection across platforms (macOS/Windows/Linux).
    """
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            _ = torch.empty(1, device="mps")  # sanity check that MPS is usable
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
    """Create directory (and parents) if it doesn't exist (idempotent)."""
    p.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# 3) Plotting (lazy import for headless safety)
# -------------------------------------------------------------------------
def plot_signal(sig: np.ndarray, title: str = "ECG (12xT)", save: str | None = None) -> None:
    """
    Plot a 12-lead ECG (shape: 12 x T) with vertical offsets for readability.

    Parameters
    ----------
    sig : np.ndarray
        Signal array shaped [12, T].
    title : str
        Plot title.
    save : str | None
        If provided, image is saved to RESULTS_DIR / save; otherwise show-less
        (non-interactive) render is created and closed.

    Implementation notes
    --------------------
    • Uses a lazy matplotlib import and 'Agg' backend to be notebook/CI-safe.
    • Ensures output directory exists before saving.

    Typical usage
    -------------
    • Quick visual sanity checks during EDA or debugging.
    """
    try:
        import matplotlib
        # Use non-interactive backend if not already set
        if os.environ.get("MPLBACKEND") is None:
            try:
                matplotlib.use("Agg", force=False)
            except Exception:
                pass
        import matplotlib.pyplot as plt
    except Exception:
        log("plot_signal: matplotlib not available; skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    offset = 2.5 * np.arange(sig.shape[0])
    for i in range(sig.shape[0]):
        plt.plot(sig[i] + offset[i])
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Leads (offset)")
    plt.tight_layout()
    if save:
        out_path = RESULTS_DIR / save
        ensure_dir(out_path.parent)  # <-- ensure parent dir exists
        plt.savefig(out_path, dpi=150)
    plt.close()


# -------------------------------------------------------------------------
# 4) Metrics
# -------------------------------------------------------------------------
def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable-ish sigmoid for converting logits to probabilities."""
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def compute_metrics(y_true: Any, y_prob: Any, labels=None) -> Dict[str, float]:
    """
    Compute basic metrics from probabilities or logits:
      - accuracy
      - macro F1
      - one-vs-rest ROC-AUC (may be NaN if classes missing)

    Args:
        y_true: array-like of shape (n,)
        y_prob: array-like of shape (n, C) — class probabilities or logits.
                Also supports (n,) or (n,1) for binary.
        labels: kept for backward compatibility (unused)

    Shape handling
    --------------
    • If y_prob is 1D or shape (n,1), it's treated as score/logit for the
      positive class (binary) and converted to 2-column probs.

    Used by
    -------
    • Quick ad-hoc evaluations in notebooks/scripts; centralized/federated
      pipelines rely on their own evaluation utilities but the signature
      matches expected behaviors (acc/F1/AUC).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Normalize shapes: handle binary 1-column or 1D scores/logits
    if y_prob.ndim == 1:
        # (n,) → interpret as score/logit for positive class
        p1 = _sigmoid(y_prob)
        y_prob = np.stack([1.0 - p1, p1], axis=1)
    elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
        p1 = _sigmoid(y_prob[:, 0])
        y_prob = np.stack([1.0 - p1, p1], axis=1)

    # Predictions
    y_pred = y_prob.argmax(axis=1)

    # Basic scores (work with ints or strings as long as y_pred matches encoding)
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # AUC (OVR) may fail for single-class y_true or degenerate columns
    try:
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

    Prints
    ------
    • counts for records/patients
    • records-per-patient (mean/median/max)
    • implied total waveform samples & hours (PTB-XL is 10s per record)

    Common call sites
    -----------------
    • Notebooks or quick CLI checks before training.
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
    print(f"Records per patient — mean {mean_rpp:.2f}, median {med_rpp:.0f}, max {max_rpp}")
    print(f"Sampling rate: {sample_rate} Hz  →  {samples_per_record:,} samples/record")
    print(f"Total waveform samples: {total_samples:,}  ({hours:.1f} h of ECG data)")
    print("==============================\n")


# -------------------------------------------------------------------------
# 6) Small extras for reporting (optional)
# -------------------------------------------------------------------------
def format_time(seconds: float) -> str:
    """Format seconds as 'Xm Ys' (used in logs/UI)."""
    minutes, sec = divmod(int(seconds), 60)
    return f"{minutes}m {sec}s"


def count_trainable_params(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters of a model (for quick sizing)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module) -> None:
    """
    Print a compact list of trainable parameters per layer and the total.

    Handy for:
      • Verifying freezing policies (which params are trainable)
      • Comparing model capacity across architectures
    """
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
def torch_loader_kwargs(shuffle: bool, batch_size: int, device_type: str | None) -> dict:
    """
    Centralized DataLoader kwargs used across the codebase.

    Policy
    ------
    • num_workers=0 by default for maximum cross-platform stability
      (Windows notebooks, macOS, CI). Bump manually if you control runtime.
    • pin_memory=True only on CUDA to speed host→device transfer.
    • persistent_workers=False to keep memory footprint low.

    Parameters
    ----------
    shuffle : bool
        Whether to shuffle batches (True for train, False for eval/inference).
    batch_size : int
        Mini-batch size.
    device_type : {'cuda','mps','cpu', None}
        Determines pin_memory behavior.

    Used in
    -------
    • Centralized training/eval loaders
    • Client fit/eval loaders
    • Experiments K-Fold loaders
    • results_visualization TorchAdapter predict loaders
    """
    if LOW_RAM:
        num_workers = 0
    else:
        # Conservative default that is stable on macOS/Windows/Linux
        num_workers = 0  # bump to 2+ later if you want multiprocessing
    pin = (device_type == "cuda")
    return dict(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=pin,
        persistent_workers=False,
        drop_last=False,
    )


# -------------------------------------------------------------------------
# 8) CSV append with optional file locking
# -------------------------------------------------------------------------
def append_csv_locked(path: Path, row: dict, fieldnames: list[str]) -> None:
    """
    Append a CSV row atomically; creates header on first write.
    
    Parameters
    ----------
    path : Path
        Output CSV file path.
    row : dict
        Dictionary with keys matching fieldnames.
    fieldnames : list[str]
        Column names for CSV header.
    
    Notes
    -----
    • Uses portalocker for atomic writes if available (multi-process safe).
    • If portalocker unavailable, writes without locking (OK for single-process).
    • Creates parent directories if needed.
    • Writes header on first call (when file doesn't exist).
    
    Used by
    -------
    • Client._log_metrics() for per-round metrics
    • Server._append_global_row(), _append_perclass_row(), _append_confusion_rows()
    """
    if portalocker is None and not hasattr(append_csv_locked, "_warned"):
        log("[warn] portalocker not installed; CSV writes are unguarded (OK for single-process).")
        append_csv_locked._warned = True
    
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    mode = "a" if path.exists() else "w"
    
    with open(path, mode, newline="") as f:
        if portalocker:
            portalocker.lock(f, portalocker.LOCK_EX)
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())
        if portalocker:
            portalocker.unlock(f)
