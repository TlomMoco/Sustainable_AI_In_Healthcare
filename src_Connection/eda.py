# src_Connection/EDA.py
"""
EDA.py — Lightweight exploratory data analysis for PTB-XL/derived tables.

What this does
--------------
• Loads PTB-XL metadata and the engineered feature tables used by deep models.
• Summarizes label/class balance overall and by basic demographics (sex, age bins).
• (If present) inspects per-patient record counts and verifies split leakage safety.
• Writes compact CSVs plus publication-ready plots to <RESULTS_DIR>/viz.

Design notes
------------
• Headless-safe matplotlib backend ("Agg") for server/CI use.
• All steps are optional/guarded — if a column is missing, that subsection is skipped.
• Uses the same config helper style as Centralized.py (C("KEY", default)).
• Does not require heavy waveform I/O; only metadata + engineered features.

Outputs (files)
---------------
viz/
  ├─ eda_label_distribution.csv / .png
  ├─ eda_label_by_sex.csv       / .png   (if `sex` available)
  ├─ eda_label_by_agebin.csv    / .png   (if `age` available)
  ├─ eda_records_per_patient.csv/ .png   (if `patient_id` available)
  ├─ eda_split_leakage_check.csv        (if train/test split applicable)
  └─ eda_meta_overview.txt               (small human-readable summary)

CLI
---
Run as a script to generate everything with defaults:

    python -m src_Connection.EDA
"""

from __future__ import annotations

# --- plotting backend (headless-safe) -----------------------------------------
import matplotlib
matplotlib.use("Agg")

# --- stdlib / third-party -----------------------------------------------------
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- project imports (lightweight) -------------------------------------------
try:
    from . import config as CFG
except Exception:
    # Fallback when relative import fails (e.g., running as plain script).
    from src_Connection import config as CFG  # type: ignore

from src_Connection import ensure_dir, log
from src_Connection import (
    load_metadata,
    make_feature_table as build_feature_tables,  # returns (feature_df, features_df)
    train_test_split as mask_split,              # patient-safe boolean split
)

# ------------------------------------------------------------------------------
# Config accessor like in Centralized.py
# ------------------------------------------------------------------------------
def C(name: str, default=None):
    return getattr(CFG, name, getattr(CFG, "NOTEBOOK", {}).get(name, default))

RESULTS_DIR = Path(C("RESULTS_DIR", Path("results")))
VIZ_DIR = RESULTS_DIR / "viz"
ensure_dir(VIZ_DIR)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _safe_write_csv(df: pd.DataFrame, name: str) -> Path:
    p = VIZ_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    print("Saved:", p)
    return p

def _safe_write_text(text: str, name: str) -> Path:
    p = VIZ_DIR / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print("Saved:", p)
    return p

def _bar(ax, x, y, title, xlabel, ylabel):
    ax.bar(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.2)

def _save_fig(fig, name: str):
    out = VIZ_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out)

def _age_to_bin(series: pd.Series) -> pd.Series:
    # Coerce numeric age and bin; labels kept short to fit plots
    s = pd.to_numeric(series, errors="coerce")
    return pd.cut(s, bins=[0, 40, 60, 200], labels=["<=40", "41-60", "61+"], include_lowest=True)

def _normalize_sex(series: pd.Series) -> pd.Series:
    # Create a normalized 'sex_norm' (male/female/unknown)
    out = pd.Series("unknown", index=series.index, dtype="string")
    s_num = pd.to_numeric(series, errors="ignore")
    with np.errstate(invalid="ignore"):
        out.loc[s_num == 1] = "male"
        out.loc[s_num == 0] = "female"
    s = series.astype("string").str.strip().str.lower()
    out.loc[s.str.startswith("m", na=False)] = "male"
    out.loc[s.str.startswith("f", na=False)] = "female"
    out.loc[s.isin({"male", "man"})] = "male"
    out.loc[s.isin({"female", "woman"})] = "female"
    return out


# ------------------------------------------------------------------------------
# Core EDA routines
# ------------------------------------------------------------------------------
def summarize_labels(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Path, Path]:
    """
    Overall class distribution from engineered features_df which must have 'label'.
    Returns (table_df, csv_path, fig_path)
    """
    if "label" not in features_df.columns:
        raise ValueError("features_df missing required column: 'label'")

    counts = (features_df["label"]
              .astype("string")
              .fillna("UNKNOWN")
              .value_counts(dropna=False)
              .rename_axis("label")
              .reset_index(name="count"))
    counts["percent"] = (counts["count"] / max(1, counts["count"].sum())) * 100.0
    csv_p = _safe_write_csv(counts, "eda_label_distribution.csv")

    # Plot
    fig = plt.figure(figsize=(8.5, 4.5))
    ax = fig.add_subplot(111)
    _bar(ax, counts["label"], counts["count"], "Label Distribution", "Label", "Count")
    ax.tick_params(axis="x", rotation=20)
    fig_p = "eda_label_distribution.png"
    _save_fig(fig, fig_p)
    return counts, csv_p, VIZ_DIR / fig_p


def summarize_by_sex(features_df: pd.DataFrame, meta_df: Optional[pd.DataFrame]) -> Optional[Tuple[pd.DataFrame, Path, Path]]:
    """
    Join features with metadata to compute label distribution by sex (if available).
    Returns (pivot_df, csv_path, fig_path) or None when not applicable.
    """
    df = features_df.copy()
    if meta_df is not None:
        # Join on a robust key — prefer 'record_path' if present, else try 'ecg_id'
        join_cols = set(df.columns) & set(meta_df.columns)
        key = "record_path" if "record_path" in join_cols else ("ecg_id" if "ecg_id" in join_cols else None)
        if key:
            df = df.merge(meta_df[[key, "sex"]].drop_duplicates(), on=key, how="left")
    if "sex" not in df.columns:
        print("[EDA] Skipping sex stratification (no 'sex' column).")
        return None

    df["sex_norm"] = _normalize_sex(df["sex"])
    grp = (df.groupby(["label", "sex_norm"], dropna=False)
             .size()
             .reset_index(name="count"))
    # Pivot to label x sex table with proportions per label
    piv = grp.pivot_table(index="label", columns="sex_norm", values="count", aggfunc="sum", fill_value=0)
    piv = piv.reindex(columns=["male", "female", "unknown"], fill_value=0)
    piv["total"] = piv.sum(axis=1)
    for c in ["male", "female", "unknown"]:
        piv[f"pct_{c}"] = (piv[c] / piv["total"].replace(0, np.nan)) * 100.0
    piv = piv.reset_index()

    csv_p = _safe_write_csv(piv, "eda_label_by_sex.csv")

    # Plot stacked proportions per label
    fig = plt.figure(figsize=(9, 4.8))
    ax = fig.add_subplot(111)
    labels = piv["label"].astype(str).tolist()
    male = piv["pct_male"].fillna(0).values
    female = piv["pct_female"].fillna(0).values
    unknown = piv["pct_unknown"].fillna(0).values

    x = np.arange(len(labels))
    ax.bar(x, male, label="Male")
    ax.bar(x, female, bottom=male, label="Female")
    ax.bar(x, unknown, bottom=male + female, label="Unknown")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("% within label"); ax.set_title("Sex Composition by Label")
    ax.legend(ncol=3, fontsize="small", frameon=False); ax.grid(True, axis="y", alpha=0.25)
    fig_p = "eda_label_by_sex.png"
    _save_fig(fig, fig_p)
    return piv, csv_p, VIZ_DIR / fig_p


def summarize_by_agebin(features_df: pd.DataFrame, meta_df: Optional[pd.DataFrame]) -> Optional[Tuple[pd.DataFrame, Path, Path]]:
    """
    Join features with metadata to compute label distribution across age bins.
    Returns (pivot_df, csv_path, fig_path) or None when not applicable.
    """
    if meta_df is None or "age" not in meta_df.columns:
        print("[EDA] Skipping age stratification (no 'age' in metadata).")
        return None

    df = features_df.copy()
    # Join on robust key
    join_cols = set(df.columns) & set(meta_df.columns)
    key = "record_path" if "record_path" in join_cols else ("ecg_id" if "ecg_id" in join_cols else None)
    if not key:
        print("[EDA] Skipping age stratification (no common join key).")
        return None

    tmp = meta_df[[key, "age"]].drop_duplicates()
    tmp["age_bin"] = _age_to_bin(tmp["age"])
    df = df.merge(tmp[[key, "age_bin"]], on=key, how="left")
    if "age_bin" not in df.columns:
        print("[EDA] Skipping age stratification (failed to create age_bin).")
        return None

    grp = (df.groupby(["label", "age_bin"], dropna=False)
             .size()
             .reset_index(name="count"))
    piv = grp.pivot_table(index="label", columns="age_bin", values="count", aggfunc="sum", fill_value=0)
    piv = piv.reindex(columns=["<=40", "41-60", "61+"], fill_value=0)
    piv["total"] = piv.sum(axis=1)
    for c in ["<=40", "41-60", "61+"]:
        piv[f"pct_{c}"] = (piv[c] / piv["total"].replace(0, np.nan)) * 100.0
    piv = piv.reset_index()
    csv_p = _safe_write_csv(piv, "eda_label_by_agebin.csv")

    # Plot stacked proportions per label
    fig = plt.figure(figsize=(9.2, 4.8))
    ax = fig.add_subplot(111)
    labels = piv["label"].astype(str).tolist()
    a = piv["pct_<=40"].fillna(0).values
    b = piv["pct_41-60"].fillna(0).values
    c = piv["pct_61+"].fillna(0).values
    x = np.arange(len(labels))
    ax.bar(x, a, label="<=40")
    ax.bar(x, b, bottom=a, label="41-60")
    ax.bar(x, c, bottom=a + b, label="61+")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("% within label"); ax.set_title("Age Composition by Label")
    ax.legend(ncol=3, fontsize="small", frameon=False); ax.grid(True, axis="y", alpha=0.25)
    fig_p = "eda_label_by_agebin.png"
    _save_fig(fig, fig_p)
    return piv, csv_p, VIZ_DIR / fig_p


def summarize_records_per_patient(features_df: pd.DataFrame) -> Optional[Tuple[pd.DataFrame, Path, Path]]:
    """
    Count records per patient (if patient_id is available).
    Returns (table_df, csv_path, fig_path) or None when not applicable.
    """
    if "patient_id" not in features_df.columns:
        print("[EDA] Skipping per-patient counts (no 'patient_id').")
        return None
    grp = (features_df.groupby("patient_id", dropna=False)
                        .size()
                        .reset_index(name="n_records")
                        .sort_values("n_records", ascending=False))
    csv_p = _safe_write_csv(grp, "eda_records_per_patient.csv")

    # Histogram
    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_subplot(111)
    vals = grp["n_records"].values
    bins = np.arange(1, int(vals.max()) + 2) if len(vals) else np.arange(1, 2)
    ax.hist(vals, bins=bins, align="left", rwidth=0.9)
    ax.set_xlabel("# records per patient"); ax.set_ylabel("Count of patients")
    ax.set_title("Records per Patient"); ax.grid(True, axis="y", alpha=0.25)
    fig_p = "eda_records_per_patient.png"
    _save_fig(fig, fig_p)
    return grp, csv_p, VIZ_DIR / fig_p


def split_leakage_check(features_df: pd.DataFrame) -> Optional[Path]:
    """
    Run the project’s patient-safe split and verify zero patient overlap.
    Writes a small CSV with counts and overlap summary.
    """
    if "patient_id" not in features_df.columns:
        print("[EDA] Skipping leakage check (no 'patient_id').")
        return None
    tr_mask, te_mask = mask_split(features_df)
    tr_pats = set(features_df.loc[tr_mask, "patient_id"].astype(str))
    te_pats = set(features_df.loc[te_mask,  "patient_id"].astype(str))
    overlap = sorted(tr_pats & te_pats)
    row = {
        "n_train": int(tr_mask.sum()),
        "n_test": int(te_mask.sum()),
        "n_patients_train": int(len(tr_pats)),
        "n_patients_test": int(len(te_pats)),
        "n_patient_overlap": int(len(overlap)),
        "overlap_example": ";".join(overlap[:10]),
    }
    df = pd.DataFrame([row])
    return _safe_write_csv(df, "eda_split_leakage_check.csv")


# ------------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------------
def run_eda(save_feature_csv: bool = True) -> Dict[str, Any]:
    """
    Execute the full EDA suite and return a dictionary of paths/frames produced.
    """
    out: Dict[str, Any] = {}

    # 1) Load metadata + engineered features table
    meta = None
    try:
        meta = load_metadata()
        # Many implementations return (ptb, scp) — keep the first as meta table if tuple
        if isinstance(meta, (tuple, list)) and len(meta) >= 1:
            meta = meta[0]
        if not isinstance(meta, pd.DataFrame):
            meta = None
    except Exception as e:
        log(f"[EDA] load_metadata failed: {e}")

    try:
        feature_df, features_df = build_feature_tables(save_csv=save_feature_csv)
    except Exception as e:
        # Fallback: if your builder returns a single table for deep models
        log(f"[EDA] make_feature_table returned unexpected shape: {e}")
        built = build_feature_tables.__wrapped__ if hasattr(build_feature_tables, "__wrapped__") else None
        raise

    # The deep model table is the second (features_df)
    df = features_df.copy()

    # 2) Small overview text
    overview = {
        "n_rows_features": int(len(df)),
        "n_cols_features": int(df.shape[1]),
        "has_label": bool("label" in df.columns),
        "has_sex": bool((meta is not None) and ("sex" in meta.columns)),
        "has_age": bool((meta is not None) and ("age" in meta.columns)),
        "has_patient_id": bool("patient_id" in df.columns),
        "head_features_cols": list(df.columns[:20]),
    }
    _safe_write_text(json.dumps(overview, indent=2), "eda_meta_overview.txt")
    out["overview"] = overview

    # 3) Label distribution
    if "label" in df.columns:
        table, csv_p, fig_p = summarize_labels(df)
        out["label_dist_df"] = table; out["label_dist_csv"] = str(csv_p); out["label_dist_fig"] = str(fig_p)
    else:
        log("[EDA] 'label' column not found — skipping label distribution.")

    # 4) By sex (if available)
    sex_out = summarize_by_sex(df, meta)
    if sex_out:
        piv, csv_p, fig_p = sex_out
        out["by_sex_df"] = piv; out["by_sex_csv"] = str(csv_p); out["by_sex_fig"] = str(fig_p)

    # 5) By age bins (if available)
    age_out = summarize_by_agebin(df, meta)
    if age_out:
        piv, csv_p, fig_p = age_out
        out["by_agebin_df"] = piv; out["by_agebin_csv"] = str(csv_p); out["by_agebin_fig"] = str(fig_p)

    # 6) Per-patient counts (if present)
    pat_out = summarize_records_per_patient(df)
    if pat_out:
        tab, csv_p, fig_p = pat_out
        out["per_patient_df"] = tab; out["per_patient_csv"] = str(csv_p); out["per_patient_fig"] = str(fig_p)

    # 7) Leakage/split sanity check (if applicable)
    leak_csv = split_leakage_check(df)
    if leak_csv:
        out["split_leakage_csv"] = str(leak_csv)

    print(f"[EDA] Complete. Outputs in: {VIZ_DIR.resolve()}")
    return out


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
def main():
    run_eda(save_feature_csv=True)


if __name__ == "__main__":
    main()