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
        

