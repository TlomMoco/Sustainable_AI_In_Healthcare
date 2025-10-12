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
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import wfdb
import ast
from pathlib import Path

from src_Connection.config import (
    PTBXL_CSV, SCP_CSV, DATA_ROOT, SAMPLE_RATE,
    N_CLASSES, SUPERCLASSES,
    # Notebook/deep config values:
    RECORD_FILE_COL, SEQ_LEN, DOWNSAMPLE_FACTOR, SAVE_FEATURES_CSV,
    ART_DIR, SEED
)
from joblib import Parallel, delayed

# =============================================================================
# ========== Classic PTB-XL dataclass-based API (baseline compatibility) ======
# =============================================================================

# -------------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------------
@dataclass
class PTBXL:
    """Container for PTB-XL metadata and SCP aggregation table."""
    df: pd.DataFrame
    agg: pd.DataFrame


# -------------------------------------------------------------------------
# Metadata loading (classic)
# -------------------------------------------------------------------------
def load_metadata() -> PTBXL:
    """
    Load and preprocess PTB-XL metadata and SCP aggregation table.

    Returns:
        PTBXL: dataclass containing main dataframe and diagnostic mappings.
    """
    df = pd.read_csv(PTBXL_CSV)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    agg = pd.read_csv(SCP_CSV, index_col=0)
    agg = agg[agg["diagnostic"] == 1][["diagnostic_class"]]
    return PTBXL(df=df, agg=agg)


# -------------------------------------------------------------------------
# Label mapping and filtering
# -------------------------------------------------------------------------
def map_superclasses(ptb: PTBXL) -> pd.DataFrame:
    """
    Map each ECG record's SCP codes to diagnostic superclasses.
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
    Keep only samples with exactly one diagnostic superclass.
    Adds column 'y' as the single class.
    """
    one = df[df["diagnostic_superclasses"].apply(lambda x: len(x) == 1)].copy()
    one["y"] = one["diagnostic_superclasses"].str[0]
    one = one[one["y"].isin(SUPERCLASSES)]
    return one


# -------------------------------------------------------------------------
# Patient-aware splitting (classic)
# -------------------------------------------------------------------------
def stratified_patient_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group-aware split by patient_id to prevent leakage.
    """
    rng = np.random.default_rng(seed)

    # Majority label per patient
    patient_labels = df.groupby("patient_id")["y"].agg(lambda s: s.value_counts().idxmax())
    patients = patient_labels.index.to_numpy()

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
    """
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"

    train_frac, val_frac, test_frac = splits

    # Step 1: train vs temp
    train_df, temp_df = stratified_patient_split(df, test_size=(1.0 - train_frac), seed=seed)

    if len(temp_df) == 0:
        raise ValueError("Temp split is empty; dataset too small for requested ratios.")

    # Step 2: val vs test from temp
    val_prop_within_temp = val_frac / (val_frac + test_frac)
    val_df, test_df = stratified_patient_split(temp_df, test_size=(1.0 - val_prop_within_temp), seed=seed + 1)
    return train_df, val_df, test_df


# -------------------------------------------------------------------------
# Waveform loading & normalization (classic)
# -------------------------------------------------------------------------
def load_waveform(row: pd.Series, sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    """
    Load a single ECG signal from disk via WFDB.

    Returns:
        np.ndarray: shape (12, T)
    """
    rel = row["filename_lr"] if sampling_rate == 100 else row["filename_hr"]
    rec_path = (DATA_ROOT / f"{rel}").as_posix()
    sig, _ = wfdb.rdsamp(rec_path)
    return np.asarray(sig).T  # (T, 12) → (12, T)


def basic_features(signal: np.ndarray) -> np.ndarray:
    """Compute simple per-lead statistics: mean, std, RMS."""
    means = signal.mean(axis=1)
    stds = signal.std(axis=1)
    rms = np.sqrt((signal ** 2).mean(axis=1))
    return np.concatenate([means, stds, rms], axis=0)


def make_feature_table_legacy(df: pd.DataFrame, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Legacy baseline features API (kept for old code):
        X: (n, 36) feature matrix
        y: (n,) class indices
        classes: SUPERCLASSES list
    """
    sel = df if limit is None else df.iloc[:limit]
    X, y = [], []
    for _, row in sel.iterrows():
        sig = load_waveform(row)
        X.append(basic_features(sig))
        y.append(SUPERCLASSES.index(row["y"]))
    return np.vstack(X), np.array(y), SUPERCLASSES


def compute_perlead_norm_stats(df: pd.DataFrame, sampling_rate: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-lead mean and std using only the provided DataFrame.
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


# =============================================================================
# ======================= Notebook / deep-learning API ========================
# =============================================================================
# These functions back your newer Centralized.py path (feature_df, features_df).

# Local aliases from config (already imported above)
DB_CSV  = Path(PTBXL_CSV)
SCP_CSV_PATH = Path(SCP_CSV)
ROOT    = DB_CSV.parent
RECORDS_DIR = ROOT / ("records100" if RECORD_FILE_COL == "filename_lr" else "records500")

def load_metadata_raw() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Raw CSV loaders used by the notebook/deep pipeline.
    Returns:
        (db, scp_raw)
    """
    assert DB_CSV.exists(), f"Missing {DB_CSV}"
    assert SCP_CSV_PATH.exists(), f"Missing {SCP_CSV_PATH}"
    db  = pd.read_csv(DB_CSV)
    scp_raw = pd.read_csv(SCP_CSV_PATH, encoding="utf-8-sig")
    scp_raw.columns = [str(c).strip() for c in scp_raw.columns]
    return db, scp_raw


def _extract_scp_keys(db_series: pd.Series, n=2000):
    keys = set()
    for s in db_series.head(min(n, len(db_series))):
        try:
            keys.update(map(str, ast.literal_eval(s).keys()))
        except Exception:
            pass
    return keys


def build_scp_mapping(db: pd.DataFrame, scp_raw: pd.DataFrame):
    obj_cols = [c for c in scp_raw.columns if scp_raw[c].dtype == "object"]
    known_keys = _extract_scp_keys(db["scp_codes"], n=2000)
    scores = {c: len(set(scp_raw[c].astype(str)).intersection(known_keys)) for c in obj_cols} or {obj_cols[0]:1}
    code_col = max(scores, key=scores.get)

    scp_raw["code"] = scp_raw[code_col].astype(str)
    scp = scp_raw.set_index("code", drop=False); scp.index.name = "code"

    for req in ["diagnostic", "diagnostic_class", "diagnostic_subclass"]:
        if req not in scp.columns:
            raise ValueError(f"scp_statements.csv missing column: {req}")

    diag_col = scp["diagnostic"].astype(str).str.strip().str.lower()
    diag_mask = pd.to_numeric(diag_col, errors="coerce").fillna(0).gt(0) | diag_col.isin({"1","1.0","true","t","yes","y","on"})
    diag_only = scp.loc[diag_mask, ["diagnostic_class","diagnostic_subclass"]].copy()
    valid_codes = set(diag_only.index)

    # Label field chosen by config (5-class or 10-class)
    from src_Connection.config import LABEL_MODE, SCP_MIN_CONF
    label_field = "diagnostic_class" if LABEL_MODE == "5class" else "diagnostic_subclass"

    def to_diag_label(scp_codes_dict, min_conf=float(SCP_MIN_CONF)):
        try:
            items = [(str(k), float(v)) for k, v in ast.literal_eval(scp_codes_dict).items()] \
                    if isinstance(scp_codes_dict, str) else [(str(k), float(v)) for k, v in scp_codes_dict.items()]
        except Exception:
            return None
        items = [(k, v) for k, v in items if k in valid_codes and v >= min_conf]
        if not items:
            return None
        agg: Dict[str, float] = {}
        for k, w in items:
            lab = diag_only.loc[k, label_field]
            agg[lab] = agg.get(lab, 0.0) + w
        return max(agg.items(), key=lambda kv: kv[1])[0]

    return diag_only, valid_codes, to_diag_label


META_FIELDS = ["ecg_id","patient_id","age","sex","device","recording_date","scp_codes","strat_fold"]

def _row_min(r: dict, to_diag_label) -> dict | None:
    row: Dict[str, Any] = {}
    for k in META_FIELDS:
        v = r.get(k, np.nan)
        if k == "recording_date":
            try:
                row[k] = pd.to_datetime(v).date().isoformat()
            except Exception:
                row[k] = np.nan
        else:
            row[k] = v
    y = to_diag_label(r.get("scp_codes", "{}"))
    if y is None:
        return None
    rec_dat = ROOT / r[RECORD_FILE_COL]
    rec_noext = str(rec_dat)[:-4] if str(rec_dat).endswith(".dat") else str(rec_dat)
    row["record_path"] = rec_noext
    row["label"] = y
    return row


def build_minimal_table(db: pd.DataFrame, to_diag_label, max_records=None) -> pd.DataFrame:
    rows = db.to_dict(orient="records")
    if max_records:
        rows = rows[:max_records]
    res = Parallel(n_jobs=-1, prefer="threads")(delayed(_row_min)(r, to_diag_label) for r in rows)
    df = pd.DataFrame([r for r in res if r is not None])
    if df.empty:
        raise RuntimeError("No rows produced — check file paths/mappings.")
    return df


# ----------------- Waveforms for deep pipeline -----------------
def _sr_from_cfg(): 
    return 100 if RECORD_FILE_COL == "filename_lr" else 500


def load_waveform_np(rec_path: str, T: int, factor: int):
    STD_LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    x, meta = wfdb.rdsamp(rec_path)
    x = x.astype("float32")
    try:
        sig_names = meta.get("sig_name", [])
        if len(sig_names) >= 12 and all(s in sig_names for s in STD_LEADS):
            idx = [sig_names.index(s) for s in STD_LEADS]
            x = x[:, idx]
    except Exception:
        pass
    if factor > 1:
        x = x[::factor]
    if x.shape[0] >= T:
        x = x[:T, :]
    else:
        pad = np.zeros((T - x.shape[0], x.shape[1]), dtype="float32")
        x = np.vstack([x, pad])
    return x


# ----------------- Basic engineered features + HRV -----------------
def _bandpower(sig_1d, fs, f_lo, f_hi):
    x = np.asarray(sig_1d, dtype=np.float32)
    n = len(x)
    if n == 0:
        return 0.0
    freqs = np.fft.rfftfreq(n, d=1.0/fs)
    spec  = np.abs(np.fft.rfft(x))**2
    mask  = (freqs >= f_lo) & (freqs < f_hi)
    msum  = int(mask.sum())
    return float(spec[mask].sum() / max(1, msum))


def _stats_and_bands(X, fs, leads=12):
    feats = {}
    for j in range(min(leads, X.shape[1])):
        s = X[:, j]
        feats[f"L{j+1}_mean"] = float(np.mean(s))
        feats[f"L{j+1}_std"]  = float(np.std(s))
        feats[f"L{j+1}_rms"]  = float(np.sqrt(np.mean(s*s)))
        feats[f"L{j+1}_ptp"]  = float(np.ptp(s))
    bands = [(0.0,5.0),(5.0,15.0),(15.0,40.0)]
    for j in range(min(leads, X.shape[1])):
        s = X[:, j]
        for (a,b) in bands:
            feats[f"L{j+1}_bp_{int(a)}_{int(b)}Hz"] = _bandpower(s, fs, a, b)
    return feats


# HR/HRV with SciPy fallback
try:
    from scipy.signal import butter, filtfilt, find_peaks
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _bandpass(x, fs, lo=5.0, hi=15.0, order=2):
    x = np.asarray(x, np.float32)
    if not _HAS_SCIPY:
        return x - x.mean()
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band')
    return filtfilt(b, a, x)


def _rpeaks(signal_1d, fs):
    y = _bandpass(signal_1d, fs)
    y = y**2
    win = max(1, int(0.150 * fs))
    ma = np.convolve(y, np.ones(win)/win, mode='same')
    distance = max(1, int(0.25 * fs))
    height = float(ma.mean() + 1.0*ma.std())
    if _HAS_SCIPY:
        peaks, _ = find_peaks(ma, distance=distance, height=height)
    else:
        cand = np.where((ma[1:-1] > ma[:-2]) & (ma[1:-1] > ma[2:]) & (ma[1:-1] >= height))[0] + 1
        keep = []
        last = -10**9
        for p in cand:
            if not keep or (p - last) >= distance:
                keep.append(p); last = p
        peaks = np.asarray(keep, dtype=int)
    return peaks


def _hrv_metrics(peaks, fs):
    rr = np.diff(peaks) / float(fs)
    if rr.size < 2:
        return {"n_beats": int(peaks.size), "HR_bpm": np.nan, "SDNN_ms": np.nan, "RMSSD_ms": np.nan}
    hr = 60.0 / rr.mean()
    sdnn = np.std(rr, ddof=1) * 1000.0
    rmssd = np.sqrt(np.mean(np.diff(rr)**2)) * 1000.0
    return {"n_beats": int(peaks.size), "HR_bpm": float(hr), "SDNN_ms": float(sdnn), "RMSSD_ms": float(rmssd)}


def hrv_for_record(path: str):
    x, meta = wfdb.rdsamp(path)
    fs = float(meta.get("fs", _sr_from_cfg()))
    names = meta.get("sig_name", []) or []
    lead_idx = names.index("II") if "II" in names else 0
    s = x[:, lead_idx].astype("float32")
    peaks = _rpeaks(s, fs)
    return _hrv_metrics(peaks, fs)


def make_feature_table(save_csv: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Notebook-style engineered features + minimal table for deep models.

    Returns:
        feature_df: DataFrame of engineered features (+label)
        features_df: minimal table with record_path/label/metadata for deep loaders
    """
    from src_Connection.config import MAX_RECORDS  # defer to avoid circular import issues
    db, scp_raw = load_metadata_raw()
    diag_only, valid_codes, to_diag_label = build_scp_mapping(db, scp_raw)

    # Minimal table
    features_df = build_minimal_table(db, to_diag_label, MAX_RECORDS)

    # Engineered features
    fs = _sr_from_cfg()
    T  = max(1, int(SEQ_LEN // max(1, DOWNSAMPLE_FACTOR)))

    rows = []
    for idx, p in features_df["record_path"].astype(str).items():
        try:
            X = load_waveform_np(p, T=T, factor=int(DOWNSAMPLE_FACTOR))
            feats = _stats_and_bands(X, fs, leads=min(12, X.shape[1]))
            hrv = hrv_for_record(p)
            rows.append({"index": int(idx), **feats, **hrv})
        except Exception:
            continue

    feat = pd.DataFrame(rows).set_index("index").sort_index()
    feat["label"] = features_df.loc[feat.index, "label"].astype(str)

    # Cache to CSV
    if save_csv and SAVE_FEATURES_CSV:
        out = Path(ART_DIR) / "basic_signal_features.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        feat.to_csv(out)

    return feat, features_df


# ----------------- Leakage-safe Split for deep features -----------------
def train_test_split(features_df: pd.DataFrame):
    """
    Group-wise split with PTB-XL strat_fold if available; otherwise GroupShuffleSplit by patient_id.
    """
    from sklearn.model_selection import GroupShuffleSplit
    if "strat_fold" in features_df.columns and features_df["strat_fold"].notna().all():
        train_mask = features_df["strat_fold"].astype(int).isin(range(1,10))
        test_mask  = features_df["strat_fold"].astype(int) == 10
    else:
        groups = features_df["patient_id"].values if "patient_id" in features_df.columns else np.arange(len(features_df))
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        tr_i, te_i = next(gss.split(features_df, features_df["label"], groups))
        train_mask = features_df.index.isin(tr_i); test_mask = features_df.index.isin(te_i)
    return train_mask, test_mask
