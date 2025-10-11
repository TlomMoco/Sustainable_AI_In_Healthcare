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

from src.config import (
    PTBXL_CSV, SCP_CSV, DATA_ROOT, SAMPLE_RATE,
    N_CLASSES, SUPERCLASSES
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
# Metadata loading
# -------------------------------------------------------------------------
def load_metadata() -> PTBXL:
    """
    Load and preprocess PTB-XL metadata and SCP aggregation table.

    Returns:
        PTBXL: dataclass containing main dataframe and diagnostic mappings.
    """
    # Load PTB-XL database
    df = pd.read_csv(PTBXL_CSV)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

    # Load diagnostic class mapping
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


def make_feature_table(df: pd.DataFrame, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Materialize ML features for a subset (or all) rows.

    Returns:
        X: (n, 36) feature matrix
        y: (n,) class indices
        CLASSES: list of class names
    """
    sel = df if limit is None else df.iloc[:limit]
    X, y = [], []
    for _, row in sel.iterrows():
        sig = load_waveform(row)
        X.append(basic_features(sig))
        y.append(SUPERCLASSES.index(row["y"]))
    return np.vstack(X), np.array(y), SUPERCLASSES


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