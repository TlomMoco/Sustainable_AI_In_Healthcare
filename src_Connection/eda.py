# src_Connection/eda.py
# ---------
# ## 6) EDA — summaries and a few saved plots (from notebook, condensed)
# ---------
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config as CFG
from .data_loader import load_waveform_np
from .utils import log

ART_DIR = Path(CFG.ART_DIR) / "figs"
ART_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, name: str) -> None:
    out = ART_DIR / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)


def class_counts_bar(features_df: pd.DataFrame) -> None:
    labs = features_df.get("label")
    if labs is None:
        log("class_counts_bar: 'label' column missing — skipping.")
        return
    labs = labs.astype(str)
    counts = labs.value_counts().sort_index()
    if counts.empty:
        log("class_counts_bar: no labels to plot — skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(counts))
    ax.bar(xs, counts.values, width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(counts.index)
    ax.set_title("Class counts")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "eda_class_counts.png")


def records_per_year(features_df: pd.DataFrame) -> None:
    if "recording_date" not in features_df.columns:
        log("records_per_year: 'recording_date' not present — skipping.")
        return
    dt = pd.to_datetime(features_df["recording_date"], errors="coerce")
    yrs = dt.dt.year.value_counts().sort_index()
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
    if "age" not in meta_df.columns:
        log("age_histogram: 'age' not present — skipping.")
        return
    ages = meta_df["age"].replace({300: np.nan}).dropna()
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


def sex_distribution(meta_df: pd.DataFrame) -> None:
    if "sex" not in meta_df.columns:
        log("sex_distribution: 'sex' not present — skipping.")
        return
    counts = meta_df["sex"].value_counts()
    if counts.empty:
        log("sex_distribution: empty — skipping.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%")
    ax.set_title("Sex distribution")
    _save(fig, "eda_sex_pie.png")


def sample_signal_plots(features_df: pd.DataFrame, n: int = 3) -> None:
    """Optional: save a few example ECGs (guarded by CFG.EDA_SKIP_HEAVY)."""
    if CFG.EDA_SKIP_HEAVY:
        log("sample_signal_plots: CFG.EDA_SKIP_HEAVY=True — skipping waveform plots.")
        return
    if "record_path" not in features_df.columns:
        log("sample_signal_plots: 'record_path' missing — skipping.")
        return

    if len(features_df) == 0:
        log("sample_signal_plots: empty dataframe — skipping.")
        return

    ex = features_df.sample(min(n, len(features_df)), random_state=7)
    T = max(1, CFG.SEQ_LEN // max(1, CFG.DOWNSAMPLE_FACTOR))

    for i, (_, row) in enumerate(ex.iterrows(), start=1):
        path = str(row.get("record_path", ""))
        if not path:
            continue
        try:
            X = load_waveform_np(path, T=T, factor=CFG.DOWNSAMPLE_FACTOR)  # (T, C)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(X[:, 0])  # lead I (or first column after channel mapping)
            ax.set_title(f"Example #{i} — id={row.get('ecg_id','?')} label={row.get('label','?')}")
            ax.set_xlabel("Samples")
            ax.set_ylabel("Amplitude")
            ax.grid(alpha=0.25)
            _save(fig, f"example_signal_{i}.png")
        except Exception as e:
            log(f"sample_signal_plots: failed on {path} ({e})")


def run_basic_eda(features_df: pd.DataFrame, meta_df: Optional[pd.DataFrame] = None) -> None:
    """
    Generate a small EDA pack into ART_DIR using the feature/minimal tables.
    Pass the same `features_df` returned by data_loader.make_feature_table(...).
    If you also keep a richer meta table, pass it as `meta_df`; otherwise this
    function will use columns present in `features_df` when available.
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
