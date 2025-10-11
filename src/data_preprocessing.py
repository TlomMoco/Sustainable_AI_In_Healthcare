# src/data_preprocessing.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── Optional deps ───────────────────────────────────────────────────────────
try:
    import wfdb  # reading PTB-XL records
except Exception as e:  # pragma: no cover
    wfdb = None
    _WFDB_IMPORT_ERROR = e
else:
    _WFDB_IMPORT_ERROR = None

# SciPy is optional; we fall back if missing.
try:
    from scipy.signal import butter, filtfilt, find_peaks
    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False

# ── Project config & utils (with robust fallbacks) ──────────────────────────
try:
    from src.config import RESULTS_DIR, SEQ_LEN, DOWNSAMPLE_FACTOR
except Exception:
    RESULTS_DIR = Path("results")
    SEQ_LEN = 600
    DOWNSAMPLE_FACTOR = 2

try:
    from src.utils import ensure_dir
except Exception:
    def ensure_dir(p: Path) -> None:
        Path(p).mkdir(parents=True, exist_ok=True)

# ── Constants ───────────────────────────────────────────────────────────────
STD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
FREQ_BANDS = [(0.0, 5.0), (5.0, 15.0), (15.0, 40.0)]

FEAT_DIR = RESULTS_DIR / "features"
ensure_dir(FEAT_DIR)

# ── Small helpers ───────────────────────────────────────────────────────────
def _require_wfdb() -> None:
    if wfdb is None:
        raise RuntimeError(
            "wfdb is required for feature extraction. Install with `pip install wfdb`."
        ) from _WFDB_IMPORT_ERROR

def _select_std_leads(x: np.ndarray, sig_names: Sequence[str]) -> np.ndarray:
    """Reorder/trim to the standard 12-lead order when possible; otherwise return as-is."""
    try:
        if len(sig_names) >= 12 and all(s in sig_names for s in STD_LEADS):
            idx = [sig_names.index(s) for s in STD_LEADS]
            return x[:, idx]
    except Exception:
        pass
    return x

def _downsample_and_pad(x: np.ndarray,
                        target_len: int = SEQ_LEN,
                        factor: int = DOWNSAMPLE_FACTOR) -> np.ndarray:
    if factor > 1:
        x = x[::int(factor)]
    T = int(target_len // max(int(factor), 1))
    if x.shape[0] >= T:
        return x[:T, :]
    pad = np.zeros((T - x.shape[0], x.shape[1]), dtype=x.dtype)
    return np.vstack([x, pad])

def _bandpower(sig_1d: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    """Simple FFT bandpower normalized by bin count to keep scale stable."""
    x = np.asarray(sig_1d, dtype=np.float32)
    n = len(x)
    if n == 0 or fs <= 0:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x)) ** 2
    mask = (freqs >= f_lo) & (freqs < f_hi)
    msum = int(mask.sum())
    return float(spec[mask].sum() / max(1, msum))

def _bandpass(x: np.ndarray, fs: float, lo: float = 5.0, hi: float = 15.0, order: int = 2) -> np.ndarray:
    if not _HAS_SCIPY or fs <= 0:
        y = x - np.mean(x)
        return y
    nyq = fs / 2.0
    lo_n = max(1e-6, lo / nyq)
    hi_n = min(0.999, hi / nyq)
    b, a = butter(order, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, x)

def _detect_r_peaks(signal_1d: np.ndarray, fs: float) -> np.ndarray:
    """Pan–Tompkins-like lite: bandpass → square → moving average → peak picking."""
    y = _bandpass(np.asarray(signal_1d, dtype=np.float32), fs)
    y = y * y
    win = max(1, int(0.150 * fs))
    ma = np.convolve(y, np.ones(win) / win, mode="same")
    distance = max(1, int(0.25 * fs))
    height = float(ma.mean() + 1.0 * ma.std())
    if _HAS_SCIPY:
        peaks, _ = find_peaks(ma, distance=distance, height=height)
        return peaks.astype(int)
    # Fallback peak detection without SciPy
    peaks = np.where((ma[1:-1] > ma[:-2]) & (ma[1:-1] >= ma[2:]) & (ma[1:-1] >= height))[0] + 1
    keep, last = [], -10**9
    for p in peaks:
        if p - last >= distance:
            keep.append(p); last = p
    return np.asarray(keep, dtype=int)

def _hrv_from_peaks(peaks: np.ndarray, fs: float) -> Dict[str, float]:
    rr = np.diff(peaks) / float(fs) if fs > 0 else np.array([])
    if rr.size < 2:
        return {"HR_bpm": np.nan, "SDNN_ms": np.nan, "RMSSD_ms": np.nan, "N_beats": int(peaks.size)}
    hr = 60.0 / rr.mean()
    sdnn = np.std(rr, ddof=1) * 1000.0
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0
    return {"HR_bpm": float(hr), "SDNN_ms": float(sdnn), "RMSSD_ms": float(rmssd), "N_beats": int(peaks.size)}

# ── Public loading helpers (importable elsewhere) ───────────────────────────
def load_waveform(record_path: str) -> Tuple[np.ndarray, float, List[str]]:
    """
    Load a full-length multi-lead ECG from WFDB (no downsampling/truncation).
    Returns (signals[T, C], fs, sig_names).
    """
    _require_wfdb()
    x, meta = wfdb.rdsamp(record_path)
    x = x.astype("float32")
    fs = float(meta.get("fs", 100.0))
    sig_names = [str(s) for s in (meta.get("sig_name", []) or [])]
    x = _select_std_leads(x, sig_names)
    return x, fs, sig_names

def load_ecg_numpy(record_path: str,
                   seq_len: int = SEQ_LEN,
                   downsample_factor: int = DOWNSAMPLE_FACTOR) -> np.ndarray:
    """
    Load, optionally downsample, and pad/truncate to shape (seq_len/factor, C).
    """
    _require_wfdb()
    x, meta = wfdb.rdsamp(record_path)
    x = x.astype("float32")
    sig_names = [str(s) for s in (meta.get("sig_name", []) or [])]
    x = _select_std_leads(x, sig_names)
    return _downsample_and_pad(x, target_len=int(seq_len), factor=int(downsample_factor))

# ── Feature extraction per record ───────────────────────────────────────────
def _stats_and_bands(X: np.ndarray, fs: float, max_leads: int = 12) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    C = min(int(X.shape[1]), int(max_leads))
    for j in range(C):
        s = X[:, j]
        feats[f"L{j+1}_mean"] = float(np.mean(s))
        feats[f"L{j+1}_std"]  = float(np.std(s))
        feats[f"L{j+1}_rms"]  = float(np.sqrt(np.mean(s * s)))
        feats[f"L{j+1}_ptp"]  = float(np.ptp(s))
    for j in range(C):
        s = X[:, j]
        for (a, b) in FREQ_BANDS:
            feats[f"L{j+1}_bp_{int(a)}_{int(b)}Hz"] = _bandpower(s, fs, a, b)
    return feats

def _hrv_block(record_path: str) -> Dict[str, float]:
    try:
        X_full, fs, names = load_waveform(record_path)
    except Exception:
        return {"HR_bpm": np.nan, "SDNN_ms": np.nan, "RMSSD_ms": np.nan, "N_beats": 0, "Lead_used": "NA"}
    # Prefer Lead II, else V2, else first available
    lead_idx = 0
    try:
        if "II" in names: lead_idx = names.index("II")
        elif "V2" in names: lead_idx = names.index("V2")
    except Exception:
        pass
    peaks = _detect_r_peaks(X_full[:, lead_idx], fs)
    out = _hrv_from_peaks(peaks, fs)
    out["Lead_used"] = names[lead_idx] if names else "NA"
    return out

def _features_for_record(path: str,
                         seq_len: int = SEQ_LEN,
                         downsample_factor: int = DOWNSAMPLE_FACTOR) -> Dict[str, float]:
    X = load_ecg_numpy(path, seq_len=seq_len, downsample_factor=downsample_factor)
    # Estimate fs used for bandpower: take WFDB metadata from full read
    try:
        _, fs, _ = load_waveform(path)
    except Exception:
        fs = 100.0
    feats = _stats_and_bands(X, fs, max_leads=min(12, X.shape[1]))
    feats.update(_hrv_block(path))
    return feats

# ── Public API ──────────────────────────────────────────────────────────────
@dataclass
class FeatureBuildConfig:
    """Lightweight knobs for feature extraction."""
    seq_len: int = SEQ_LEN
    downsample_factor: int = DOWNSAMPLE_FACTOR
    n_jobs: int = -1  # joblib threads; -1 = all
    progress_every: int = 200  # print heartbeat

def build_feature_table(meta_df: pd.DataFrame,
                        label_col: str = "y",
                        path_col: str = "record_path",
                        cfg: FeatureBuildConfig | None = None) -> pd.DataFrame:
    """
    Create a feature table with one row per record and columns:
    - per-lead stats (mean/std/rms/ptp)
    - per-lead bandpowers (0–5, 5–15, 15–40 Hz)
    - HR/HRV metrics (HR_bpm, SDNN_ms, RMSSD_ms, N_beats, Lead_used)
    Includes a final 'label' column copied from `label_col` if present.
    """
    if cfg is None:
        cfg = FeatureBuildConfig()

    if path_col not in meta_df.columns:
        raise ValueError(f"'{path_col}' column not found in meta_df.")

    _require_wfdb()

    paths = meta_df[path_col].astype(str)
    indices = list(paths.index)

    rows: List[Dict[str, float]] = []

    # Try joblib for speed, otherwise fall back to a simple for-loop.
    try:
        from joblib import Parallel, delayed
        def _do_one(idx: int) -> Optional[Dict[str, float]]:
            p = paths.loc[idx]
            try:
                feats = _features_for_record(p, seq_len=cfg.seq_len, downsample_factor=cfg.downsample_factor)
                return {"index": int(idx), **feats}
            except Exception:
                return None
        out = Parallel(n_jobs=int(cfg.n_jobs), prefer="threads")(
            delayed(_do_one)(idx) for idx in indices
        )
        rows = [r for r in out if r is not None]
    except Exception:
        for k, idx in enumerate(indices, 1):
            p = paths.loc[idx]
            try:
                rows.append({"index": int(idx), **_features_for_record(p, cfg.seq_len, cfg.downsample_factor)})
            except Exception:
                pass
            if cfg.progress_every and (k % int(cfg.progress_every) == 0):
                print(f"[Features] {k}/{len(indices)} done…")

    if not rows:
        raise RuntimeError("No feature rows produced. Check record paths and WFDB availability.")

    feat_df = pd.DataFrame(rows).set_index("index").sort_index()

    # Attach label if present
    if label_col in meta_df.columns:
        feat_df["label"] = meta_df.loc[feat_df.index, label_col].astype(str)

    return feat_df

def save_feature_table(df: pd.DataFrame,
                       filename: str = "basic_signal_features.csv",
                       out_dir: Path | str = FEAT_DIR) -> Path:
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    fp = out_dir / filename
    df.to_csv(fp)
    print(f"[Features] Saved → {fp.resolve()}")
    return fp

def make_feature_table(meta_df: pd.DataFrame,
                       label_col: str = "y",
                       path_col: str = "record_path",
                       cfg: FeatureBuildConfig | None = None,
                       save_csv: bool = False,
                       csv_name: str = "basic_signal_features.csv") -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Convenience wrapper expected by other modules:
    Returns (X_numeric_df, y_array, classes_list). Optionally saves CSV.
    """
    feats = build_feature_table(meta_df, label_col=label_col, path_col=path_col, cfg=cfg)

    # y / classes
    if "label" in feats.columns:
        y = feats["label"].astype(str).values
    elif label_col in meta_df.columns:
        y = meta_df.loc[feats.index, label_col].astype(str).values
        feats["label"] = y
    else:
        y = np.array(["NA"] * len(feats), dtype=object)
    classes = sorted(pd.unique(pd.Series(y).astype(str)))

    # X numeric only
    X = feats.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number]).copy()

    if save_csv:
        save_feature_table(feats, filename=csv_name)

    return X, y, classes

__all__ = [
    "FeatureBuildConfig",
    "build_feature_table",
    "make_feature_table",
    "save_feature_table",
    "load_waveform",
    "load_ecg_numpy",
]
