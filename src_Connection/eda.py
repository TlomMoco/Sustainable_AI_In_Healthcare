# src_Connection/eda.py
# ---------
# ## 6) EDA — summaries and a few saved plots (condensed)
# ---------
"""
Exploratory Data Analysis (EDA) utilities for PTB-XL.

What this module provides
-------------------------
• Small, headless-safe plotting helpers that save figures to disk
• Basic dataset sanity checks: class balance, years, ages, sex distribution
• Optional waveform snapshots (guarded by CFG.EDA_SKIP_HEAVY)

Where this fits in the project
------------------------------
• Typically called from notebooks or after building tables via:
    - src_Connection.data_loader.make_feature_table(...)
• Plots are saved under: <CFG.ART_DIR>/figs
• Uses load_waveform_np from data_loader to render sample ECGs

Inputs expected
---------------
• features_df: the minimal/feature table from make_feature_table(...) containing
  at least: 'label', 'record_path', and (optionally) 'recording_date', 'age', 'sex'

Notes
-----
• Uses matplotlib's 'Agg' backend for headless environments.
• All functions are defensive: they check for required columns and no-op if missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

from . import config as CFG
from .data_loader import load_waveform_np
from .utils import log

# Figure output directory (created on import for convenience)
ART_DIR = Path(CFG.ART_DIR) / "figs"
ART_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    """Save a figure to ART_DIR/<name> and close it to free memory."""
    out = ART_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


def _order_by_superclasses(index_like) -> list[str]:
    """Order labels by CFG.SUPERCLASSES when available, else sort alphabetically.

    Why:
        Keeps class bars consistent across runs (useful for comparisons).
    """
    sup = getattr(CFG, "SUPERCLASSES", None)
    idx = [str(x) for x in index_like]
    if isinstance(sup, (list, tuple)) and sup:
        # keep only those present, in configured order
        ordered = [c for c in sup if c in idx]
        # append any unexpected labels at the end (sorted)
        tail = sorted([c for c in idx if c not in sup])
        return ordered + tail
    return sorted(idx)


def class_counts_bar(features_df: pd.DataFrame) -> None:
    """Plot and save a bar chart of class counts using features_df['label'].

    Expects:
        • 'label' column present in features_df.

    Output:
        • 'eda_class_counts.png' under ART_DIR
    """
    labs = features_df.get("label")
    if labs is None:
        log("class_counts_bar: 'label' column missing — skipping.")
        return
    labs = labs.astype(str)
    counts = labs.value_counts()
    if counts.empty:
        log("class_counts_bar: no labels to plot — skipping.")
        return
    # Reindex to a consistent order if possible
    order = _order_by_superclasses(counts.index)
    counts = counts.reindex(order).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(counts))
    ax.bar(xs, counts.values, width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(counts.index, rotation=0)
    ax.set_title("Class counts")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "eda_class_counts.png")


def records_per_year(features_df: pd.DataFrame) -> None:
    """Plot count of records per year using features_df['recording_date'].

    Expects:
        • 'recording_date' column parseable by pandas.to_datetime

    Output:
        • 'eda_records_per_year.png' under ART_DIR
    """
    if "recording_date" not in features_df.columns:
        log("records_per_year: 'recording_date' not present — skipping.")
        return
    dt = pd.to_datetime(features_df["recording_date"], errors="coerce")
    yrs = dt.dt.year.dropna().astype(int).value_counts().sort_index()
    if yrs.empty:
        log("records_per_year: no valid years — skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(yrs))
    ax.bar(xs, yrs.values, width=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(yrs.index.astype(str), rotation=50, ha="right")
    ax.set_title("Records per year")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "eda_records_per_year.png")


def age_histogram(meta_df: pd.DataFrame) -> None:
    """Plot a histogram of patient ages.

    Expects:
        • 'age' column (numeric or parsable)
        • PTB-XL quirk: age value 300 is a sentinel; we convert it to NaN.

    Output:
        • 'eda_age_hist.png' under ART_DIR
    """
    if "age" not in meta_df.columns:
        log("age_histogram: 'age' not present — skipping.")
        return
    ages = pd.to_numeric(meta_df["age"], errors="coerce").replace({300: np.nan}).dropna()
    if ages.empty:
        log("age_histogram: no usable ages — skipping.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ages.values, bins=20)
    ax.set_title("Age distribution (300→NaN excluded)")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "eda_age_hist.png")


def _normalize_sex(series: pd.Series) -> pd.Series:
    """Normalize common 'sex' encodings to {'male', 'female', 'unknown'} for plotting.

    Handles:
        • numeric (1/0), text prefixes ('m','f'), and full words.
    """
    s = series.astype("string").str.strip().str.lower()
    out = pd.Series("unknown", index=s.index, dtype="string")

    # numeric / coded
    num = pd.to_numeric(s, errors="coerce")
    out.loc[num == 1] = "male"
    out.loc[num == 0] = "female"

    # text variants
    out.loc[s.str.startswith("m", na=False)] = "male"
    out.loc[s.str.startswith("f", na=False)] = "female"
    out.loc[s.isin({"male", "man"})] = "male"
    out.loc[s.isin({"female", "woman"})] = "female"

    return out


def sex_distribution(meta_df: pd.DataFrame) -> None:
    """Plot a pie chart of sex distribution (normalized encodings).

    Expects:
        • 'sex' column (string/numeric variants supported via _normalize_sex)

    Output:
        • 'eda_sex_pie.png' under ART_DIR
    """
    if "sex" not in meta_df.columns:
        log("sex_distribution: 'sex' not present — skipping.")
        return
    norm = _normalize_sex(meta_df["sex"])
    counts = norm.value_counts()
    if counts.empty:
        log("sex_distribution: empty — skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%")
    ax.set_title("Sex distribution")
    _save(fig, "eda_sex_pie.png")


def sample_signal_plots(features_df: pd.DataFrame, n: int = 3) -> None:
    """Optionally save a few example ECG waveforms.

    Guard:
        • Skips entirely when CFG.EDA_SKIP_HEAVY is True (default).
          This keeps quick runs lightweight.

    Expects:
        • 'record_path' (WFDB base path without extension)
        • (Optional) 'ecg_id', 'label' for nicer titles

    Output:
        • 'example_signal_{i}.png' under ART_DIR
    """
    if getattr(CFG, "EDA_SKIP_HEAVY", True):
        log("sample_signal_plots: CFG.EDA_SKIP_HEAVY=True — skipping waveform plots.")
        return
    if "record_path" not in features_df.columns:
        log("sample_signal_plots: 'record_path' missing — skipping.")
        return
    if len(features_df) == 0:
        log("sample_signal_plots: empty dataframe — skipping.")
        return

    # Reproducible sampling
    ex = features_df.sample(min(n, len(features_df)), random_state=getattr(CFG, "SEED", 42))

    # Effective time steps after downsampling
    T = max(1, int(getattr(CFG, "SEQ_LEN", 600) // max(1, int(getattr(CFG, "DOWNSAMPLE_FACTOR", 2)))))

    for i, (_, row) in enumerate(ex.iterrows(), start=1):
        path = str(row.get("record_path", ""))  # WFDB base path (no extension)
        if not path:
            continue
        try:
            X = load_waveform_np(path, T=T, factor=int(getattr(CFG, "DOWNSAMPLE_FACTOR", 2)))  # (T, C)
            # Ensure we have at least one lead to plot
            if X.ndim != 2 or X.shape[1] == 0:
                raise ValueError("waveform has invalid shape")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(X[:, 0])  # Lead I (or first column after mapping)
            ax.set_title(f"Example #{i} — id={row.get('ecg_id','?')} label={row.get('label','?')}")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            ax.grid(alpha=0.25)
            _save(fig, f"example_signal_{i}.png")
        except Exception as e:
            log(f"sample_signal_plots: failed on {path} ({e})")


def run_basic_eda(features_df: pd.DataFrame, meta_df: Optional[pd.DataFrame] = None) -> None:
    """Generate a small EDA pack (bars, hist, pie, and optional waveforms).

    Parameters
    ----------
    features_df : DataFrame
        The minimal/features table from data_loader.make_feature_table(...).
    meta_df : Optional[DataFrame]
        Richer metadata table to use for age/sex/year plots; if None, falls
        back to columns in features_df when available.

    Side effects
    ------------
    • Saves multiple figures into ART_DIR.
    • Logs progress via utils.log.
    """
    meta = meta_df if meta_df is not None else features_df
    class_counts_bar(features_df)
    records_per_year(meta)
    age_histogram(meta)
    sex_distribution(meta)
    sample_signal_plots(features_df)
    log("EDA complete.")


__all__ = [
    "run_basic_eda",
    "class_counts_bar",
    "records_per_year",
    "age_histogram",
    "sex_distribution",
    "sample_signal_plots",
]