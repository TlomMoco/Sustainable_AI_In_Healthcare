"""
data_preprocessing.py
Centralized feature engineering for PTB-XL.

Public API
----------
- load_ecg_numpy(record_path, seq_len=None, downsample_factor=None) -> (T, C) float32
- features_for_record(record_path) -> dict
- build_features_table(minimal_df, save_csv=True, csv_name="basic_signal_features.csv") -> pd.DataFrame

Notes
-----
- Uses src.config for SAMPLE_RATE and RESULTS_DIR.
- Selects standard 12 leads if available, else keeps original order.
- Stats per-lead: mean/std/rms/ptp
- Bandpower per-lead in 3 bands: 0–5, 5–15, 15–40 Hz
- HR/HRV from a full (non-truncated) lead (II preferred)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd

# ------------------ Config (robust import with fallbacks) ------------------
try:
    from src.config import SAMPLE_RATE, RESULTS_DIR
except Exception:
    SAMPLE_RATE = 100
    RESULTS_DIR = Path("results")

RESULTS_DIR = Path(RESULTS_DIR)

# ------------------ Optional deps ------------------
try:
    import wfdb  # type: ignore
    _HAS_WFDB = True
except Exception:
    _HAS_WFDB = False

try:
    from scipy.signal import butter, filtfilt, find_peaks  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

STD_LEADS: List[str] = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# ------------------ Small utils ------------------
def _require_wfdb() -> None:
    if not _HAS_WFDB:
        raise RuntimeError("WFDB not installed. Please `pip install wfdb` to read PTB-XL records.")


def _ensure_dir(p: Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _select_std_leads(x: np.ndarray, sig_names: List[str]) -> np.ndarray:
    try:
        if len(sig_names) >= 12 and all(s in sig_names for s in STD_LEADS):
            idx = [sig_names.index(s) for s in STD_LEADS]
            return x[:, idx]
    except Exception:
        pass
    return x


# ------------------ Core I/O ------------------
def _read_full_record(path: str) -> Tuple[np.ndarray, float, List[str]]:
    """Return full-length signal (float32), fs, names (no truncation/padding)."""
    _require_wfdb()
    sig, meta = wfdb.rdsamp(path)
    sig = sig.astype("float32")
    fs = float(meta.get("fs", float(SAMPLE_RATE)))
    names = [str(s) for s in (meta.get("sig_name", []) or [])]
    sig = _select_std_leads(sig, names)
    return sig, fs, names


def load_ecg_numpy(
    record_path: str,
    seq_len: int | None = None,
    downsample_factor: int | None = None,
) -> np.ndarray:
    """
    Load ECG into shape (T, C) float32, select standard 12-lead order when available,
    downsample to the configured SAMPLE_RATE (if downsample_factor is None),
    and pad/truncate to seq_len (defaults to 10 * SAMPLE_RATE).
    """
    sig, fs, _ = _read_full_record(record_path)

    # Default 10-second window
    if seq_len is None:
        seq_len = int(10 * int(SAMPLE_RATE))

    # Infer ds factor if not provided
    if downsample_factor is None and fs > 0 and SAMPLE_RATE > 0:
        downsample_factor = max(1, int(round(fs / float(SAMPLE_RATE))))

    if downsample_factor and downsample_factor > 1:
        sig = sig[::int(downsample_factor)]

    # Pad / trim to seq_len
    T = int(seq_len)
    if sig.shape[0] >= T:
        sig = sig[:T, :]
    else:
        pad = np.zeros((T - sig.shape[0], sig.shape[1]), dtype=sig.dtype)
        sig = np.vstack([sig, pad])
    return sig


# ------------------ Feature primitives ------------------
def _bandpower(sig_1d: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
    x = np.asarray(sig_1d, dtype=np.float32)
    n = len(x)
    if n == 0 or fs <= 0:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spec = np.abs(np.fft.rfft(x)) ** 2
    mask = (freqs >= f_lo) & (freqs < f_hi)
    msum = int(mask.sum())
    return float(spec[mask].sum() / max(1, msum))


def _bandpass_ecg(x: np.ndarray, fs: float, lo: float = 5.0, hi: float = 15.0, order: int = 2) -> np.ndarray:
    if not _HAS_SCIPY or fs <= 0:
        return x - np.mean(x)
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def _detect_r_peaks(signal_1d: np.ndarray, fs: float) -> np.ndarray:
    y = _bandpass_ecg(signal_1d, fs)
    y = y ** 2
    win = max(1, int(0.150 * fs))
    ma = np.convolve(y, np.ones(win) / win, mode="same")
    distance = max(1, int(0.25 * fs))
    height = float(ma.mean() + 1.0 * ma.std())
    if _HAS_SCIPY:
        peaks, _ = find_peaks(ma, distance=distance, height=height)
        return peaks.astype(int)
    # Fallback: simple distance-thresholded maxima
    cand = np.where((ma[1:-1] > ma[:-2]) & (ma[1:-1] >= ma[2:]) & (ma[1:-1] >= height))[0] + 1
    keep: list[int] = []
    for p in cand:
        if not keep or (p - keep[-1]) >= distance:
            keep.append(p)
    return np.asarray(keep, dtype=int)


def _hrv_from_peaks(peaks: np.ndarray, fs: float) -> Dict[str, Any]:
    rr = np.diff(peaks) / float(fs) if fs > 0 else np.array([])
    if rr.size < 2:
        return {"HR_bpm": np.nan, "SDNN_ms": np.nan, "RMSSD_ms": np.nan, "N_beats": int(peaks.size)}
    HR_bpm = 60.0 / rr.mean()
    SDNN_ms = float(np.std(rr, ddof=1) * 1000.0)
    RMSSD_ms = float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0)
    return {"HR_bpm": float(HR_bpm), "SDNN_ms": SDNN_ms, "RMSSD_ms": RMSSD_ms, "N_beats": int(peaks.size)}


# ------------------ Per-record features ------------------
def features_for_record(record_path: str) -> Dict[str, Any]:
    """
    Compute feature dict for a single record_path.
      - per-lead stats (mean/std/rms/ptp) on a 10s window at SAMPLE_RATE
      - per-lead bandpowers (0–5, 5–15, 15–40 Hz) on the same window
      - HR/HRV from the full (non-truncated) signal (lead II preferred)
    """
    # Windowed (downsampled) for stats/bandpower
    X = load_ecg_numpy(record_path)  # (T, C) float32 at SAMPLE_RATE
    _, C = X.shape
    feats: Dict[str, Any] = {}

    # Stats
    for j in range(min(12, C)):
        s = X[:, j]
        feats[f"L{j+1}_mean"] = float(np.mean(s))
        feats[f"L{j+1}_std"] = float(np.std(s))
        feats[f"L{j+1}_rms"] = float(np.sqrt(np.mean(s * s)))
        feats[f"L{j+1}_ptp"] = float(np.ptp(s))

    # Bandpowers
    bands = [(0.0, 5.0), (5.0, 15.0), (15.0, 40.0)]
    fs_win = float(SAMPLE_RATE)
    for j in range(min(12, C)):
        s = X[:, j]
        for (a, b) in bands:
            feats[f"L{j+1}_bp_{int(a)}_{int(b)}Hz"] = _bandpower(s, fs_win, a, b)

    # HR/HRV on full record (not truncated)
    Xfull, fs_full, names = _read_full_record(record_path)
    lead_name = "II" if "II" in names else (names[0] if names else None)
    if lead_name is not None:
        idx = names.index(lead_name)
        peaks = _detect_r_peaks(Xfull[:, idx], fs_full)
        feats.update(_hrv_from_peaks(peaks, fs_full))
        feats["Lead_used"] = str(lead_name)
    else:
        feats.update({"HR_bpm": np.nan, "SDNN_ms": np.nan, "RMSSD_ms": np.nan, "N_beats": 0, "Lead_used": "NA"})

    return feats


# ------------------ Batch builder ------------------
def build_features_table(
    minimal_df: pd.DataFrame,
    save_csv: bool = True,
    csv_name: str = "basic_signal_features.csv",
) -> pd.DataFrame:
    """
    Parameters
    ----------
    minimal_df : DataFrame
        Must contain columns: ['record_path', 'label'] and index = sample ids.
    save_csv : bool
        Save to RESULTS_DIR / csv_name.
    csv_name : str
        Output CSV filename.

    Returns
    -------
    feature_df : DataFrame
        Index aligned to minimal_df index, contains engineered features + 'label'.
    """
    assert {"record_path", "label"}.issubset(minimal_df.columns), \
        "minimal_df must contain ['record_path', 'label']"

    rows: list[dict] = []
    for k, (idx, rec_path) in enumerate(minimal_df["record_path"].astype(str).items(), start=1):
        try:
            feats = features_for_record(rec_path)
            rows.append({"index": int(idx), **feats})
        except Exception:
            # Skip unreadable rows rather than crashing the whole run
            continue
        if (k % 200) == 0:
            print(f"[Features] {k}/{len(minimal_df)} done…")

    feature_df = pd.DataFrame(rows).set_index("index").sort_index()
    feature_df["label"] = minimal_df.loc[feature_df.index, "label"].astype(str)

    print("[Features] shape:", feature_df.shape)
    if save_csv:
        out = RESULTS_DIR / csv_name
        _ensure_dir(out.parent)
        feature_df.to_csv(out)
        print("Saved:", out)
    return feature_df
