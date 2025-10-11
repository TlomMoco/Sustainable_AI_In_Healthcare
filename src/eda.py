from __future__ import annotations
import pandas as pd
import numpy as np
from rich import print

from src.config import SAMPLE_RATE
from src.data_loader import load_metadata, map_superclasses, filter_single_label, load_waveform
from src.utils import plot_signal, summarize_dataset


if __name__ == "__main__":
    ptb = load_metadata()
    df = map_superclasses(ptb)
    one = filter_single_label(df)

    summarize_dataset(one, sample_rate=SAMPLE_RATE, title="PTB-XL (Filtered Single-Label)")


    print(f"Total records: {len(df):,}")
    print(f"Single‑label records: {len(one):,}")


    # Inspect distributions
    print(one["y"].value_counts())
    by_patient = one.groupby("patient_id").size()
    print("Records per patient — median:", by_patient.median())


    # Age/Sex distribution (age 300 is 90+ per PTB‑XL privacy; treat specially in analysis)
    ages = one["age"].replace({300: np.nan})
    print("Age — mean (excluding 300):", ages.mean())
    print(one["sex"].value_counts())

"""
    # Plot a few example signals
    for i, (_, row) in enumerate(one.sample(3, random_state=7).iterrows()):
        sig = load_waveform(row, sampling_rate=SAMPLE_RATE)
        plot_signal(sig, title=f"ecg_id={row.ecg_id} class={row.y}", save=f"example_{i+1}.png")
        print("Saved example plots to results/.")"""
        
# ---------
# ## 6) EDA — summaries and a few saved plots (from notebook, condensed)
# ---------
import numpy as np, pandas as pd, matplotlib.pyplot as plt, textwrap
from pathlib import Path
from . import config as CFG

ART_DIR = Path(CFG.ART_DIR) / "figs"
ART_DIR.mkdir(parents=True, exist_ok=True)

def _save(fig, name):
    out = ART_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved:", out)

def class_counts_bar(features_df: pd.DataFrame):
    labs = features_df["label"].astype(str)
    counts = labs.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10,5))
    xs = np.arange(len(counts))
    ax.bar(xs, counts.values, width=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(counts.index)
    ax.set_title("Class counts"); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
    _save(fig, "eda_class_counts.png"); plt.close(fig)

def records_per_year(features_df: pd.DataFrame):
    if "recording_date" not in features_df.columns: return
    dt = pd.to_datetime(features_df["recording_date"], errors="coerce")
    yrs = dt.dt.year.value_counts().sort_index()
    if yrs.empty: return
    fig, ax = plt.subplots(figsize=(11,5))
    xs = np.arange(len(yrs))
    ax.bar(xs, yrs.values, width=0.7)
    ax.set_xticks(xs); ax.set_xticklabels(yrs.index.astype(str), rotation=50, ha="right")
    ax.set_title("Records per year"); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.3)
    _save(fig, "eda_records_per_year.png"); plt.close(fig)

