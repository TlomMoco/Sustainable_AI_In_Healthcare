"""
data_loader.py — PTB-XL Dataset Utilities
-----------------------------------------

Handles all data operations for the Sustainable AI in Healthcare project:
  • Load and parse the PTB-XL metadata (CSV files)
  • Map diagnostic SCP codes to 5 diagnostic superclasses
  • Patient-aware train/val/test splits (no leakage)
  • ECG waveform loading via WFDB
  • Per-lead normalization (z-score)
  • Feature extraction for baseline ML models

All functions are deterministic and use only patient IDs for stratification,
ensuring non-overlapping patient sets between clients and global splits.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import wfdb
import ast
from src.config import SAMPLE_RATE as _FS
from src.config import (
    PTBXL_CSV, SCP_CSV, DATA_ROOT, SAMPLE_RATE,SUPERCLASSES
)

# -------------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------------
@dataclass
class PTBXL:
    """Container for PTB-XL metadata and SCP aggregation table."""
    df: pd.DataFrame
    agg: pd.DataFrame


# -------------------------------------------------------------------------
# Metadata cleaning
# -------------------------------------------------------------------------
def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    PTB-XL hygiene:
      - Parse dtypes
      - Handle 'age == 300' (PTB-XL privacy code for 90+) → set NaN for analysis
      - Clamp impossible ages (<0 or >120) to NaN
      - Normalize 'sex' to {'Male','Female','Unknown'}
      - Drop rows missing waveform paths
      - Drop exact duplicate ecg_id rows if any
    """
    out = df.copy()

    # Dtypes
    for col in ["age", "patient_id", "ecg_id"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Age cleaning: 300 means 90+ in PTB-XL → use NaN for analysis/EDA
    # (Your coworker already treated it this way in EDA; we make it systematic here.)
    if "age" in out.columns:
        out.loc[out["age"] == 300, "age"] = np.nan
        out.loc[(out["age"] < 0) | (out["age"] > 120), "age"] = np.nan

    # Sex normalization
    if "sex" in out.columns:
        s = out["sex"].astype("string")  # preserves <NA>
        s = s.str.strip()

        # numeric encodings first (common pitfalls)
        # adjust the map if your CSV uses the opposite convention
        s = s.replace({"0": "Female", "1": "Male", "2": "Unknown"})

        # textual encodings (case-insensitive)
        txt = s.str.lower()
        txt_map = {
            "female": "Female", "f": "Female", "woman": "Female",
            "male": "Male", "m": "Male", "man": "Male",
        }
        s2 = txt.map(txt_map)

        # prefer mapped textual label when available, else keep numeric-mapped value
        out["sex"] = s2.fillna(s).fillna("Unknown")

    # Ensure waveform paths exist
    need_cols = ["filename_lr", "filename_hr"]
    have = [c for c in need_cols if c in out.columns]
    if have:
        mask_paths = out[have].notna().any(axis=1)
        out = out[mask_paths].copy()

    # Drop exact duplicated ecg_id rows (rare, safety net)
    if "ecg_id" in out.columns:
        out = out.drop_duplicates(subset=["ecg_id"])

    return out


# -------------------------------------------------------------------------
# Metadata loading
# -------------------------------------------------------------------------
def load_metadata() -> PTBXL:
    """
    Load and preprocess PTB-XL metadata and SCP aggregation table.

    Returns:
        PTBXL: dataclass containing main dataframe and diagnostic mappings.
    """
    # Load main PTB-XL CSV
    df = pd.read_csv(PTBXL_CSV)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    # Clean metadata before any label mapping/splitting
    df = clean_metadata(df)

    agg = pd.read_csv(SCP_CSV, index_col=0)
    agg = agg[agg["diagnostic"] == 1][["diagnostic_class"]]
    return PTBXL(df=df, agg=agg)


# -------------------------------------------------------------------------
# Label mapping and filtering
# -------------------------------------------------------------------------
def map_superclasses(ptb: PTBXL) -> pd.DataFrame:
    """
    Map each ECG record's SCP codes to diagnostic superclasses.

    Args:
        ptb: PTBXL object containing df and agg tables.

    Returns:
        pd.DataFrame: copy of PTB-XL dataframe with column
                      'diagnostic_superclasses' (list[str]).
    """
    df = ptb.df.copy()

    def to_superclasses(code_dict: Dict[str, float]) -> List[str]:
        labels = set()
        for code in code_dict.keys():
            if code in ptb.agg.index:
                labels.add(ptb.agg.loc[code, "diagnostic_class"])
        return sorted(labels)

    df["diagnostic_superclasses"] = df["scp_codes"].apply(to_superclasses)
    return df


def filter_single_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only samples with exactly one diagnostic superclass
    (≈16k of the 21k PTB-XL records).

    Returns:
        Filtered dataframe with column 'y' representing the single class.
    """
    one = df[df["diagnostic_superclasses"].apply(lambda x: len(x) == 1)].copy()
    one["y"] = one["diagnostic_superclasses"].str[0]
    one = one[one["y"].isin(SUPERCLASSES)]
    return one


# -------------------------------------------------------------------------
# Patient-aware splitting
# -------------------------------------------------------------------------
def stratified_patient_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group-aware split by patient_id to prevent leakage.

    Each patient's *majority label* is used to approximately preserve
    class proportions.

    Args:
        df: Filtered dataframe.
        test_size: Fraction of patients for test set.
        seed: Random seed for reproducibility.

    Returns:
        (train_df, test_df)
    """
    rng = np.random.default_rng(seed)

    # Determine majority label per patient
    patient_labels = df.groupby("patient_id")["y"].agg(lambda s: s.value_counts().idxmax())
    patients = patient_labels.index.to_numpy()

    # Stratified split by label
    test_patients = []
    for c in patient_labels.unique():
        p_cls = patients[patient_labels.values == c]
        rng.shuffle(p_cls)
        n_test = max(1, int(len(p_cls) * test_size))
        test_patients.extend(p_cls[:n_test])

    test_patients = set(test_patients)
    train = df[~df["patient_id"].isin(test_patients)].copy()
    test = df[df["patient_id"].isin(test_patients)].copy()
    return train, test


def stratified_patient_split_3way(
    df: pd.DataFrame, splits: Tuple[float, float, float] = (0.70, 0.15, 0.15), seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-aware 3-way split: (train, val, test).

    Ensures disjoint patient sets and approximate label balance.
    Implemented in two steps using stratified_patient_split.

    Args:
        df: Full dataframe.
        splits: Fractions for (train, val, test).
        seed: Random seed.

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"

    train_frac, val_frac, test_frac = splits

    # Step 1: train vs temp
    train_df, temp_df = stratified_patient_split(df, test_size=(1.0 - train_frac), seed=seed)

    # Step 2: val vs test from temp
    if len(temp_df) == 0:
        raise ValueError("Temp split is empty; dataset too small for requested ratios.")

    val_prop_within_temp = val_frac / (val_frac + test_frac)
    val_df, test_df = stratified_patient_split(temp_df, test_size=(1.0 - val_prop_within_temp), seed=seed + 1)
    return train_df, val_df, test_df


# -------------------------------------------------------------------------
# Waveform loading and preprocessing
# -------------------------------------------------------------------------
def load_waveform(row: pd.Series, sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load a single ECG signal from disk via WFDB.

    Args:
        row: Row from PTB-XL dataframe (contains filename_lr/hr).
        sampling_rate: 100 or 500 Hz.

    Returns:
        np.ndarray: shape (12, T)
    """
    rel = row["filename_lr"] if sampling_rate == 100 else row["filename_hr"]
    rec_path = (DATA_ROOT / f"{rel}").as_posix()
    sig, _ = wfdb.rdsamp(rec_path)
    return np.asarray(sig).T  # (T, 12) → (12, T)


# -------------------------------------------------------------------------
# Basic feature extraction (for ML baselines)
# -------------------------------------------------------------------------
def basic_features(signal: np.ndarray) -> np.ndarray:
    """Compute simple per-lead statistics: mean, std, RMS."""
    means = signal.mean(axis=1)
    stds = signal.std(axis=1)
    rms = np.sqrt((signal ** 2).mean(axis=1))
    return np.concatenate([means, stds, rms], axis=0)

# --- Engineered features (extend this with your extra stats/HRV/bands) ----
def engineered_features(signal: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    signal : np.ndarray of shape (12, T)
        Lead-major ECG (12 leads by time).

    Returns
    -------
    np.ndarray, shape (87,)
        [ per-lead stats (48) | per-lead bandpowers (36) | HRV (3) ]

    Notes
    -----
    • Per-lead stats: mean, std, RMS, peak-to-peak
    • Bandpowers per lead: 0–5 Hz, 5–15 Hz, 15–40 Hz (FFT PSD average in band)
    • HR/HRV on the "most peaky" lead (chosen by highest std): HR_bpm, SDNN_ms, RMSSD_ms
    """
    X = np.asarray(signal, dtype=np.float32)
    assert X.ndim == 2 and X.shape[0] == 12, "engineered_features expects (12, T)"

    # ---------- per-lead stats ----------
    means = X.mean(axis=1)  # 12
    stds = X.std(axis=1)  # 12
    rms = np.sqrt((X ** 2).mean(axis=1))  # 12
    ptp = np.ptp(X, axis=1) # 12
    stats_vec = np.concatenate([means, stds, rms, ptp], axis=0)  # (48,)

    # ---------- per-lead bandpowers via FFT PSD ----------
    def _bandpower(sig1d: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
        x = np.asarray(sig1d, dtype=np.float32)
        n = int(x.size)
        if n == 0:
            return 0.0
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spec = np.abs(np.fft.rfft(x)) ** 2
        m = (freqs >= f_lo) & (freqs < f_hi)
        cnt = int(m.sum())
        return float(spec[m].sum() / max(1, cnt))

    bands = [(0.0, 5.0), (5.0, 15.0), (15.0, 40.0)]
    bp_list = []
    for lead in range(12):
        s = X[lead]
        for (a, b) in bands:
            bp_list.append(_bandpower(s, float(_FS), a, b))
    bp_vec = np.array(bp_list, dtype=np.float32)  # (36,)

    # ---------- HR/HRV from R-peaks on the "best" lead ----------
    # choose lead with highest std to increase chance of clear R-peaks
    best_lead = int(np.argmax(stds))
    s = X[best_lead].astype(np.float32, copy=False)

    # Simple Pan–Tompkins-style energy envelope (NumPy only):
    y = s - np.median(s)  # detrend a bit
    y = y ** 2  # energy
    win = max(1, int(0.150 * _FS))  # 150 ms moving-average
    env = np.convolve(y, np.ones(win, dtype=np.float32) / win, mode="same")

    # Adaptive threshold + refractory period (250 ms)
    thr = float(env.mean() + 1.0 * env.std())
    refractory = max(1, int(0.250 * _FS))

    # Peak picking (local maxima over threshold with refractory)
    cand = np.where((env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] >= thr))[0] + 1
    peaks = []
    last = -10 ** 9
    for p in cand:
        if (p - last) >= refractory:
            peaks.append(int(p))
            last = p
    peaks = np.array(peaks, dtype=int)

    # RR intervals (s) and HRV metrics
    if peaks.size >= 3:
        rr = np.diff(peaks) / float(_FS)  # seconds
        hr_bpm = 60.0 / rr.mean()
        sdnn_ms = rr.std(ddof=1) * 1000.0
        rmssd_ms = np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0
    else:
        hr_bpm, sdnn_ms, rmssd_ms = np.nan, np.nan, np.nan

    hrv_vec = np.array([hr_bpm, sdnn_ms, rmssd_ms], dtype=np.float32)  # (3,)

    # ---------- merge ----------
    out = np.concatenate([stats_vec, bp_vec, hrv_vec], axis=0)  # (87,)
    return out


# Feature names for inspection/ANOVA labeling:
def engineered_feature_names() -> list[str]:
    names = []
    for j in range(12):
        names += [f"L{j+1}_mean", f"L{j+1}_std", f"L{j+1}_rms", f"L{j+1}_ptp"]
    for j in range(12):
        names += [f"L{j+1}_bp_0_5Hz", f"L{j+1}_bp_5_15Hz", f"L{j+1}_bp_15_40Hz"]
    names += ["HR_bpm", "SDNN_ms", "RMSSD_ms"]
    return names


def make_feature_table(
    df: pd.DataFrame,
    limit: int | None = None,
    feature_set: str = "basic",   # "basic" (default) or "engineered"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Materialize ML features for a subset (or all) rows.

    Returns:
        X: (n, d) feature matrix (d depends on feature_set)
        y: (n,) class indices
        CLASSES: list of class names
    """
    sel = df if limit is None else df.iloc[:limit]

    # Choose feature function
    feat_fn = engineered_features if feature_set == "engineered" else basic_features

    X, y = [], []
    for _, row in sel.iterrows():
        sig = load_waveform(row)                   # (12, T)
        X.append(feat_fn(sig))                     # 1D vector
        y.append(SUPERCLASSES.index(row["y"]))     # int label

    X = np.vstack(X)
    y = np.array(y, dtype=int)
    return X, y, SUPERCLASSES


# -------------------------------------------------------------------------
# Normalization utilities
# -------------------------------------------------------------------------
def compute_perlead_norm_stats(df: pd.DataFrame, sampling_rate: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-lead mean and std using only the provided DataFrame
    (usually the training split).

    Returns:
        mu: np.ndarray shape (12,)
        sigma: np.ndarray shape (12,)
    """
    means, stds = [], []
    for _, row in df.iterrows():
        sig = load_waveform(row, sampling_rate)
        means.append(sig.mean(axis=1))
        stds.append(sig.std(axis=1))
    mu = np.stack(means).mean(axis=0) if means else np.zeros(12)
    sigma = np.stack(stds).mean(axis=0) if stds else np.ones(12)
    sigma = np.maximum(sigma, 1e-6)
    return mu, sigma


def normalize_signal(sig: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply z-score normalization per lead."""
    return (sig - mu[:, None]) / (sigma[:, None] + eps)