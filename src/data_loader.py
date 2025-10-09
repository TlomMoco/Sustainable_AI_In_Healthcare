from __future__ import annotations
import numpy as np
import pandas as pd
import wfdb
import ast
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Config variable imports
from src.config import (PTBXL_CSV, SCP_CSV, DATA_ROOT, SAMPLE_RATE, N_CLASSES, SUPERCLASSES)


@dataclass
class PTBXL:
    """PTB-XL dataset loader and processor."""
    df: pd.DataFrame
    agg: pd.DataFrame

def load_metadata():
    """Load and preprocess PTB-XL metadata and the scp mapping table."""
    df = pd.read_csv(PTBXL_CSV)
    # Parse dict‑like strings to dict
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    agg = pd.read_csv(SCP_CSV, index_col=0)
    agg = agg[agg["diagnostic"] == 1][["diagnostic_class"]]
    return PTBXL(df=df, agg=agg)


def map_superclasses(ptb: PTBXL) -> pd.DataFrame:
    """ Map scp codes to the 5 diagnostic superclasses per sample.
    Returns df with columns: diagnostic_superclasses (list), y (None initially) """
    df = ptb.df.copy()
    def to_superclasses(code_dict: Dict[str, float]) -> List[str]:
        labels = set()
        for code in code_dict.keys():
            if code in ptb.agg.index:
                labels.add(ptb.agg.loc[code, "diagnostic_class"])
        return sorted(list(labels))

    df["diagnostic_superclasses"] = df["scp_codes"].apply(to_superclasses)
    return df


def filter_single_label(df: pd.DataFrame) -> pd.DataFrame:
    """Keep samples with exactly one diagnostic superclass (≈16k in PTB‑XL)."""
    one = df[df["diagnostic_superclasses"].apply(lambda x: len(x) == 1)].copy()
    one["y"] = one["diagnostic_superclasses"].str[0]
    one = one[one["y"].isin(SUPERCLASSES)]
    return one


def stratified_patient_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Group‑aware split by patient_id to avoid leakage.
    We aim to preserve label proportions at the patient level. """

    rng = np.random.default_rng(seed)

    # Patient‑level majority label for stratification surrogate
    patient_labels = df.groupby("patient_id")["y"].agg(lambda s: s.value_counts().idxmax())
    patients = patient_labels.index.to_numpy()

    # stratify by label proxy
    idx = np.arange(len(patients))

    # per‑class split
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


def stratified_patient_split_3way(df: pd.DataFrame, splits: Tuple[float, float, float] = (0.70, 0.15, 0.15), seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Patient-aware 3-way split into (train, val, test).
    Guarantees disjoint patients and approximates label balance by patient-majority label.

    We implement as a two-step split to reuse stratified_patient_split:
    1) train vs temp
    2) val vs test from temp
    """
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    train_frac, val_frac, test_frac = splits
    # step 1: train vs temp
    train_df, temp_df = stratified_patient_split(df, test_size=(1.0 - train_frac), seed=seed)
    # step 2: val vs test
    # val proportion within temp:
    if len(temp_df) == 0:
        raise ValueError("Temp split is empty; dataset too small for requested ratios.")
    val_prop_within_temp = val_frac / (val_frac + test_frac)
    val_df, test_df = stratified_patient_split(temp_df, test_size=(1.0 - val_prop_within_temp), seed=seed + 1)
    return train_df, val_df, test_df


def load_waveform(row: pd.Series, sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    """ Load a single ECG (12xT) using wfdb based on PTB‑XL metadata row.
    Returns np.ndarray shape (12, T). """
    rel = row["filename_lr"] if sampling_rate == 100 else row["filename_hr"]
    rec_path = (DATA_ROOT / f"{rel}").as_posix()
    sig, _ = wfdb.rdsamp(rec_path)
    # (T, 12) -> (12, T)
    return np.asarray(sig).T


def basic_features(signal: np.ndarray) -> np.ndarray:
    """ Compute simple per‑lead features: mean, std, RMS. signal=(12,T). """
    means = signal.mean(axis=1)
    stds = signal.std(axis=1)
    rms = np.sqrt((signal ** 2).mean(axis=1))
    return np.concatenate([means, stds, rms], axis=0)


def make_feature_table(df: pd.DataFrame, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """ Materialize ML features for a subset (or all) rows.
    Returns X (n,36), y_idx (n,), classes. """
    classes = ["NORM", "MI", "STTC", "HYP", "CD"]
    sel = df if limit is None else df.iloc[:limit]
    X, y = [], []
    for _, row in sel.iterrows():
        sig = load_waveform(row)
        X.append(basic_features(sig))
        y.append(classes.index(row["y"]))
    return np.vstack(X), np.array(y), classes


def compute_perlead_norm_stats(df: pd.DataFrame, sampling_rate: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-lead mean/std using only the provided df (train split).
    Returns (mu[12], sigma[12]) for z-scoring.
    """
    means, stds, n = [], [], 0
    for _, row in df.iterrows():
        sig = load_waveform(row, sampling_rate)
        # per-record per-lead stats
        means.append(sig.mean(axis=1))
        stds.append(sig.std(axis=1))
        n += 1
    mu = np.stack(means).mean(axis=0) if n > 0 else np.zeros(12)
    sigma = np.stack(stds).mean(axis=0) if n > 0 else np.ones(12)
    sigma = np.maximum(sigma, 1e-6)
    return mu, sigma


def normalize_signal(sig: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Z-score per lead: (12, T) -> (12, T)."""
    return (sig - mu[:, None]) / (sigma[:, None] + eps)