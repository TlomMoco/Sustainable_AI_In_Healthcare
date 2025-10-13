#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
results_visualization.py — unified plotting script (prettified + numbers)

Enhancements in this update
---------------------------
- Fix Matplotlib 3.9 deprecation: boxplot(labels=...) -> boxplot(tick_labels=...)
- Add value labels on bars across EDA plots
- Export the numbers behind every figure to CSVs:
    results/viz/eda/tables/
      - table_class_counts.csv
      - table_sex_overall_counts.csv
      - table_class_by_sex_counts.csv
      - table_class_by_sex_percent.csv
      - table_records_per_year.csv
      - table_class_by_fold_counts.csv
      - table_age_summary_overall.csv
      - table_age_summary_by_class.csv
      - table_age_summary_by_sex.csv
      - table_feature_corr_spearman.csv
      - table_feature_corr_pruned_spearman.csv
      - table_hrv_summary_by_class.csv
      - table_hrv_summary_by_sex.csv
      - table_anova_top.csv
    results/viz/
      - confusion_<tag>_rXX_counts.csv
      - confusion_<tag>_rXX_percent.csv
"""
from __future__ import annotations
import os
import re
import ast
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import patheffects as pe

# --------------------------------------------------------------------------------------
# Basic setup & helpers
# --------------------------------------------------------------------------------------

def ts():
    import time
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str):
    print(f"[{ts()}] {msg}")

def ensure_outdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def savefig(fig: plt.Figure, outdir: Path, name: str):
    ensure_outdir(outdir)
    fp = outdir / name
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp}")

def _savetab(df: pd.DataFrame | pd.Series, outdir: Path, name: str):
    """Save numbers to CSV, printing a friendly line."""
    ensure_outdir(outdir)
    fp = outdir / name
    (df.to_frame() if isinstance(df, pd.Series) else df).to_csv(fp)
    print(f"Saved table: {fp}")

# --------------------------------------------------------------------------------------
# Paths & defaults
# --------------------------------------------------------------------------------------

DEFAULT_DATASET_ROOT = Path("dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
DEFAULT_DB_CSV = DEFAULT_DATASET_ROOT / "ptbxl_database.csv"
DEFAULT_SCP_CSV = DEFAULT_DATASET_ROOT / "scp_statements.csv"
DEFAULT_FEATURES_CSV_CANDIDATES = [
    Path("test/artifacts/basic_signal_features.csv"),
    Path("artifacts/basic_signal_features.csv"),
    Path("results/basic_signal_features.csv"),
]
DEFAULT_OUTDIR = Path("results/viz")

# Ordered class palette (5-class)
ORDER_5 = ["NORM","MI","CD","STTC","HYP"]
SEX_COLORS = {"female": "#E66BB5", "male": "#4A90E2", "unknown": "#9B9B9B"}
CLASS_COLORS = {"NORM": "#4C78A8", "MI": "#F58518", "CD": "#54A24B", "STTC": "#E45756", "HYP": "#B279A2"}

# --------------------------------------------------------------------------------------
# PTB-XL label mapping (SCP → diagnostic_class/subclass) as in the notebook
# --------------------------------------------------------------------------------------

def load_db_and_scp(db_csv: Path, scp_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    if not db_csv.exists():
        print(f"Missing: {db_csv}")
        return pd.DataFrame(), pd.DataFrame(), "code"
    if not scp_csv.exists():
        print(f"Missing: {scp_csv}")
        return pd.DataFrame(), pd.DataFrame(), "code"
    db = pd.read_csv(db_csv)
    scp_raw = pd.read_csv(scp_csv, encoding="utf-8-sig")
    scp_raw.columns = [str(c).strip() for c in scp_raw.columns]

    # Pick the column that best intersects with db[scp_codes]
    def _extract_scp_keys(db_series, n=2000):
        keys = set()
        for s in db_series.head(min(n, len(db_series))):
            try: keys.update(map(str, ast.literal_eval(s).keys()))
            except Exception: pass
        return keys

    known_keys = _extract_scp_keys(db.get("scp_codes", pd.Series([], dtype=object)))
    obj_cols = [c for c in scp_raw.columns if scp_raw[c].dtype == "object"]
    scores = {c: len(set(scp_raw[c].astype(str)).intersection(known_keys)) for c in obj_cols} or {obj_cols[0]:1}
    code_col = max(scores, key=scores.get)

    scp_raw["code"] = scp_raw[code_col].astype(str)
    scp = scp_raw.set_index("code", drop=False)
    return db, scp, code_col


def make_label_mapper(scp: pd.DataFrame, label_mode: str = "5class"):
    for req in ["diagnostic", "diagnostic_class", "diagnostic_subclass"]:
        if req not in scp.columns:
            raise ValueError(f"scp_statements.csv missing column: {req}")

    diag_col = scp["diagnostic"].astype(str).str.strip().str.lower()
    diag_mask = pd.to_numeric(diag_col, errors="coerce").fillna(0).gt(0) | diag_col.isin({"1","1.0","true","t","yes","y","on"})
    diag_only = scp.loc[diag_mask, ["diagnostic_class","diagnostic_subclass"]].copy()
    valid_codes = set(diag_only.index)

    label_field = "diagnostic_class" if label_mode == "5class" else "diagnostic_subclass"

    def to_diag_label(scp_codes_dict, min_conf=0.0):
        try:
            items = [(str(k), float(v)) for k, v in scp_codes_dict.items()]
        except Exception:
            return None
        items = [(k, v) for k, v in items if k in valid_codes and v >= min_conf]
        if not items: return None
        agg: Dict[str, float] = {}
        for k, w in items:
            lab = diag_only.loc[k, label_field]
            agg[lab] = agg.get(lab, 0.0) + w
        return max(agg.items(), key=lambda kv: kv[1])[0]

    return to_diag_label, valid_codes, diag_only, label_field


def build_minimal_table(db: pd.DataFrame, to_diag_label, record_file_col: str = "filename_lr", min_conf: float = 0.0) -> pd.DataFrame:
    META_FIELDS = ["ecg_id","patient_id","age","sex","device","recording_date","scp_codes","strat_fold"]
    rows = []
    for _, r in db.iterrows():
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
        try:
            scp_codes = ast.literal_eval(r.get("scp_codes", "{}"))
        except Exception:
            continue
        y = to_diag_label(scp_codes, min_conf)
        if y is None:
            continue
        rec_dat = (DEFAULT_DATASET_ROOT / r[record_file_col]) if not os.path.isabs(str(r[record_file_col])) else Path(r[record_file_col])
        rec_noext = str(rec_dat)[:-4] if str(rec_dat).endswith(".dat") else str(rec_dat)
        row["record_path"] = rec_noext
        row["label"] = y
        rows.append(row)
    if not rows:
        raise RuntimeError("No rows produced — check PTB-XL paths/mappings.")
    df = pd.DataFrame(rows)
    return df


# --------------------------------------------------------------------------------------
# EDA helpers (styling etc.) — parity with notebook visuals
# --------------------------------------------------------------------------------------

BAR_BASE = 0.32
SPREAD_BASE = 1.00
GAP_FRAC_GROUPED = 0.08

def _wrap(names: Iterable[str], width: int = 12) -> list[str]:
    import textwrap
    return ["\n".join(textwrap.wrap(str(n), width=width)) for n in names]

def _x_positions(n: int, spread: float = SPREAD_BASE):
    return np.arange(n) * spread

def _bar_width(n: int, base: float = BAR_BASE):
    return float(np.clip(base, 0.18, 0.45))

def _sex_norm(series: pd.Series) -> pd.Series:
    out = pd.Series("unknown", index=series.index, dtype="string")
    s_num = pd.to_numeric(series, errors="coerce")
    out.loc[s_num == 1] = "male"
    out.loc[s_num == 0] = "female"
    s = series.astype("string").str.strip().str.lower()
    out.loc[s.str.startswith("m", na=False)] = "male"
    out.loc[s.str.startswith("f", na=False)] = "female"
    out.loc[s.isin({"male","man"})] = "male"
    out.loc[s.isin({"female","woman"})] = "female"
    return out

def _prettify_axes(ax: plt.Axes):
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{int(v):,}"))

def _text_with_outline(ax, x, y, s, fontsize=10, color="#111", outline_width=2.0, outline_color="white", **kwargs):
    t = ax.text(x, y, s, ha="center", va="bottom", fontsize=fontsize, color=color, **kwargs)
    t.set_path_effects([pe.Stroke(linewidth=outline_width, foreground=outline_color), pe.Normal()])
    return t

def _fd_bins(x: np.ndarray, min_bins=20, max_bins=60):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return min_bins
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr <= 0:
        return min(max_bins, max(min_bins, int(np.sqrt(x.size))))
    bw = 2 * iqr * (x.size ** (-1/3))
    if bw <= 0:
        return min(max_bins, max(min_bins, int(np.sqrt(x.size))))
    bins = int(np.ceil((x.max() - x.min()) / bw))
    return int(np.clip(bins, min_bins, max_bins))

def _annot_bars(ax, xs, ys, fmt="{:,}", dy_frac=0.02, fontsize=10):
    """Add numbers above bars."""
    if len(ys) == 0:
        return
    ymax = float(np.nanmax(ys)) if len(ys) else 1.0
    dy = max(1.0, ymax * dy_frac)
    for x, y in zip(xs, ys):
        if np.isfinite(y):
            s = fmt.format(int(round(y))) if fmt == "{:,}" else fmt.format(y)
            _text_with_outline(ax, x, y + dy, s, fontsize=fontsize)

# --------------------------------------------------------------------------------------
# EDA plotters (with table exports)
# --------------------------------------------------------------------------------------

def plot_missingness_top(features_df: pd.DataFrame, outdir: Path):
    if features_df.empty:
        return
    NA_TOKENS = {"", " ", "nan", "none", "null", "NaN", "None", "NULL"}
    edf2 = features_df.copy()
    for col in edf2.columns:
        if edf2[col].dtype == "object":
            edf2[col] = edf2[col].replace(list(NA_TOKENS), np.nan)
    na_rate = edf2.isna().mean().sort_values(ascending=False)
    nz = (100 * na_rate[na_rate > 0]).sort_values(ascending=False)
    if nz.empty:
        print("No missing values detected — skipping missingness plot.")
        return
    top = nz.iloc[:15].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ys = _x_positions(len(top), 1.00)
    ax.barh(ys, top.values, height=_bar_width(len(top)))
    ax.set_yticks(ys)
    ax.set_yticklabels(top.index)
    ymax = float(top.max())
    ax.set_xlim(0, ymax * 1.08)
    ax.set_xlabel("NA rate (%)")
    ax.set_title("Missingness (top columns)")
    ax.grid(axis="x", alpha=0.3)
    for y, v in zip(ys, top.values):
        ax.text(v + ymax * 0.01, y, f"{v:.2f}%", va="center", fontsize=9)
    plt.tight_layout()
    savefig(fig, outdir, "eda_missingness_top.png")
    _savetab(nz.sort_values(ascending=False), outdir / "tables", "table_missingness_percent.csv")

def plot_class_counts(meta_df: pd.DataFrame, outdir: Path):
    if meta_df.empty or "label" not in meta_df.columns:
        return
    labs = meta_df["label"].astype(str)
    order = [c for c in ORDER_5 if c in set(labs)] or sorted(labs.unique())
    cls_counts = labs.value_counts().reindex(order).fillna(0).astype(int)

    # Save numbers
    tab = pd.DataFrame({"count": cls_counts, "percent": (cls_counts / cls_counts.sum() * 100).round(2)})
    _savetab(tab, outdir / "tables", "table_class_counts.csv")

    n = len(order)
    xs = _x_positions(n, 1.00)
    w = _bar_width(n)
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    colors = [CLASS_COLORS.get(c, None) for c in order]
    ax.bar(xs, cls_counts.values, width=w, color=colors)
    ymax = float(np.nanmax(cls_counts.values)) if len(cls_counts) else 1.0
    ax.set_ylim(0, ymax * 1.08)
    ax.set_xticks(xs)
    ax.set_xticklabels(_wrap(order, 10))
    ax.set_title("Class counts")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    _annot_bars(ax, xs, cls_counts.values)
    plt.tight_layout()
    savefig(fig, outdir, "eda_class_counts.png")

def plot_sex_overall_and_by_class(meta_df: pd.DataFrame, outdir: Path):
    if meta_df.empty or "sex" not in meta_df.columns:
        return
    df = meta_df.copy()
    df["sex_norm"] = _sex_norm(df["sex"])
    tables_dir = outdir / "tables"

    # overall
    vc = df["sex_norm"].value_counts().reindex(["female","male","unknown"]).dropna().astype(int)
    _savetab(vc, tables_dir, "table_sex_overall_counts.csv")

    k = len(vc)
    group_width = 0.80
    bar_gap_frac = 0.10
    bar_w = group_width / max(k, 1)
    eff_w = bar_w * (1 - bar_gap_frac)
    xs = [(-group_width/2 + j*bar_w + bar_w/2) for j in range(k)]

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ymax = float(max(vc.values)) if k else 1.0
    for xi, (lab, v) in zip(xs, vc.items()):
        ax.bar(xi, v, width=eff_w, color=SEX_COLORS.get(lab), edgecolor="white", linewidth=0.6, alpha=0.95)
        _text_with_outline(ax, xi, v + max(1, 0.02*ymax), f"{int(v):,}", fontsize=10)
    ax.set_ylim(0, ymax*1.10)
    ax.set_xticks(xs)
    ax.set_xticklabels(list(vc.index))
    ax.set_title("Sex distribution (overall)")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    plt.tight_layout()
    savefig(fig, outdir, "eda_sex_overall.png")

    # class x sex
    if "label" in df.columns:
        ct = pd.crosstab(df["label"].astype(str), df["sex_norm"]).reindex(ORDER_5, fill_value=0).fillna(0).astype(int)
        _savetab(ct, tables_dir, "table_class_by_sex_counts.csv")
        sex_order = [s for s in ["female","male","unknown"] if s in ct.columns] or list(ct.columns)

        x = _x_positions(len(ct.index), 1.00)
        k = len(sex_order)
        group_width = 0.80
        bar_w = group_width / max(k, 1)
        eff_w = bar_w * (1 - GAP_FRAC_GROUPED)

        fig, ax = plt.subplots(figsize=(10.6, 5.6))
        ymax = 0.0
        for j, col in enumerate(sex_order):
            offs = -group_width/2 + j*bar_w + bar_w/2
            vals = ct[col].values
            ymax = max(ymax, float(np.nanmax(vals)) if len(vals) else 0.0)
            bars = ax.bar(x + offs, vals, width=eff_w, label=col, color=SEX_COLORS.get(col),
                          edgecolor="white", linewidth=0.6)
            _annot_bars(ax, x + offs, vals)
        ax.set_ylim(0, ymax * 1.18)
        ax.set_xticks(x)
        ax.set_xticklabels(_wrap(ct.index, 10))
        ax.set_ylabel("Count")
        ax.set_title("Class × Sex (counts)")
        ax.legend()
        _prettify_axes(ax)
        plt.tight_layout()
        savefig(fig, outdir, "eda_class_by_sex_counts.png")

        pct = (ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100).fillna(0).round(2)
        _savetab(pct, tables_dir, "table_class_by_sex_percent.csv")

        fig, ax = plt.subplots(figsize=(10.6, 5.6))
        for j, col in enumerate(sex_order):
            offs = -group_width/2 + j*bar_w + bar_w/2
            vals = pct[col].values
            ax.bar(x + offs, vals, width=eff_w, label=col, color=SEX_COLORS.get(col), edgecolor="white", linewidth=0.6)
            _annot_bars(ax, x + offs, vals, fmt="{:.2f}")
        ax.set_xticks(x)
        ax.set_xticklabels(_wrap(pct.index, 10))
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 100)
        ax.set_title("Class × Sex (normalized)")
        ax.legend()
        # override y formatter to raw (not thousands)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
        plt.tight_layout()
        savefig(fig, outdir, "eda_class_by_sex_percent.png")

def plot_age_distributions(meta_df: pd.DataFrame, outdir: Path):
    if meta_df.empty or "age" not in meta_df.columns:
        return
    tables_dir = outdir / "tables"
    df = meta_df.copy()
    a = pd.to_numeric(df["age"], errors="coerce").dropna().values
    if a.size:
        # overall age summary
        overall = pd.Series({
            "n": int(a.size),
            "mean": float(np.mean(a)),
            "std": float(np.std(a, ddof=1)) if a.size > 1 else float("nan"),
            "median": float(np.median(a))
        }).to_frame("age_overall")
        _savetab(overall.T.round(2), tables_dir, "table_age_summary_overall.csv")

        bins = _fd_bins(a, min_bins=24, max_bins=64)
        fig, ax = plt.subplots(figsize=(11.5, 5.6))
        ax.hist(a, bins=bins, rwidth=0.94, edgecolor="white", linewidth=0.6, alpha=0.9)
        mu, med = float(np.mean(a)), float(np.median(a))
        ax.axvline(mu, linestyle="--", linewidth=1.4, color="k", label=f"Mean {mu:.1f}")
        ax.axvline(med, linestyle=":", linewidth=1.6, color="k", label=f"Median {med:.1f}")
        ax.set_title("Age distribution", fontsize=14)
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        _prettify_axes(ax)
        ax.legend(frameon=False, loc="upper center", ncol=2)
        plt.tight_layout()
        savefig(fig, outdir, "eda_age_hist_overall_pretty.png")

    # Age by class
    if "label" in df.columns:
        data_c, labs_c = [], []
        for c in [c for c in ORDER_5 if c in set(df["label"])]:
            v = pd.to_numeric(df.loc[df["label"].astype(str) == c, "age"], errors="coerce").dropna().values
            if v.size >= 5:
                data_c.append(v)
                labs_c.append(c)
        if data_c:
            # summary table
            rows = []
            for c, v in zip(labs_c, data_c):
                rows.append({"label": c, "n": int(v.size), "mean": np.mean(v), "std": (np.std(v, ddof=1) if v.size > 1 else np.nan),
                             "median": np.median(v)})
            _savetab(pd.DataFrame(rows).set_index("label").round(2).loc[labs_c], tables_dir, "table_age_summary_by_class.csv")

            fig, ax = plt.subplots(figsize=(10.8, 5.2))
            ax.boxplot(data_c, tick_labels=_wrap(labs_c, 10), showfliers=False)  # CHANGED: tick_labels
            # annotate n & medians
            ymin, ymax = ax.get_ylim(); dy = 0.03*(ymax - ymin)
            for i, v in enumerate(data_c, start=1):
                ax.text(i, np.percentile(v, 75) + dy, f"n={len(v):,}\nmed={np.median(v):.1f}",
                        ha="center", va="bottom", fontsize=9)
            ax.set_title("Age by diagnostic class", fontsize=14)
            ax.set_ylabel("Age")
            _prettify_axes(ax)
            plt.tight_layout()
            savefig(fig, outdir, "eda_age_by_class_box_pretty.png")

    # Age by sex (overlay + box)
    if "sex" in df.columns:
        df["sex_norm"] = _sex_norm(df["sex"]) if "sex_norm" not in df.columns else df["sex_norm"]
        af = pd.to_numeric(df.loc[df["sex_norm"] == "female", "age"], errors="coerce").dropna().values
        am = pd.to_numeric(df.loc[df["sex_norm"] == "male", "age"], errors="coerce").dropna().values

        if af.size or am.size:
            # summary by sex
            ss = []
            if af.size: ss.append({"sex":"female","n":int(af.size),"mean":np.mean(af),"std":(np.std(af, ddof=1) if af.size>1 else np.nan),"median":np.median(af)})
            if am.size: ss.append({"sex":"male","n":int(am.size),"mean":np.mean(am),"std":(np.std(am, ddof=1) if am.size>1 else np.nan),"median":np.median(am)})
            if ss:
                _savetab(pd.DataFrame(ss).set_index("sex").round(2), tables_dir, "table_age_summary_by_sex.csv")

            base = (np.concatenate([af, am]) if (af.size and am.size) else (af if af.size else am))
            bins = _fd_bins(base, min_bins=24, max_bins=64)
            fig, ax = plt.subplots(figsize=(11.5, 5.6))
            if af.size:
                ax.hist(af, bins=bins, rwidth=0.90, alpha=0.55, label="female", color=SEX_COLORS.get("female"),
                        edgecolor="white", linewidth=0.6)
            if am.size:
                ax.hist(am, bins=bins, rwidth=0.90, alpha=0.55, label="male", color=SEX_COLORS.get("male"),
                        edgecolor="white", linewidth=0.6)
            ax.set_title("Age distribution by sex (overlay)", fontsize=14)
            ax.set_xlabel("Age")
            ax.set_ylabel("Count")
            _prettify_axes(ax)
            ax.legend(frameon=False, ncol=2, loc="upper center")
            plt.tight_layout()
            savefig(fig, outdir, "eda_age_hist_by_sex_overlay_pretty.png")

            if af.size and am.size:
                fig, ax = plt.subplots(figsize=(8.2, 5.0))
                sexes = ["female", "male"]
                data = [af, am]
                bp = ax.boxplot(data, tick_labels=sexes, showfliers=False, patch_artist=True,  # CHANGED
                                boxprops=dict(linewidth=1.2), medianprops=dict(linewidth=1.6),
                                whiskerprops=dict(linewidth=1.0), capprops=dict(linewidth=1.0))
                for i, lab in enumerate(sexes):
                    c = SEX_COLORS.get(lab)
                    bp["boxes"][i].set_facecolor(c)
                    bp["boxes"][i].set_alpha(0.25)
                    bp["boxes"][i].set_edgecolor(c)
                    bp["medians"][i].set_color(c)
                    ax.text(i+1, np.percentile(data[i], 75) + 1.5, f"n={len(data[i]):,}\nmed={np.median(data[i]):.1f}",
                            ha="center", va="bottom", fontsize=9)
                ax.set_title("Age by sex", fontsize=14)
                ax.set_ylabel("Age")
                _prettify_axes(ax)
                plt.tight_layout()
                savefig(fig, outdir, "eda_age_by_sex_box_pretty.png")

def plot_records_per_year(meta_df: pd.DataFrame, outdir: Path):
    if meta_df.empty or "recording_date" not in meta_df.columns:
        return
    dt = pd.to_datetime(meta_df["recording_date"], errors="coerce")
    yrs = dt.dt.year.value_counts().sort_index()
    if yrs.size == 0:
        return
    _savetab(yrs, outdir / "tables", "table_records_per_year.csv")

    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    xs = _x_positions(len(yrs), spread=1.00)
    bar_width = 0.72
    ax.bar(xs, yrs.values, width=bar_width, edgecolor="white", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(yrs.index.astype(str), rotation=50, ha="right")
    ax.set_title("Records per year", fontsize=14)
    ax.set_ylabel("Count")
    _prettify_axes(ax)
    ymax = float(yrs.values.max())
    ax.set_ylim(0, ymax * 1.14)
    _annot_bars(ax, xs, yrs.values)
    plt.tight_layout()
    savefig(fig, outdir, "eda_records_per_year_pretty_alllabels.png")

def plot_strat_fold_grouped(meta_df: pd.DataFrame, outdir: Path):
    if meta_df.empty or "strat_fold" not in meta_df.columns or "label" not in meta_df.columns:
        return
    ct = pd.crosstab(meta_df["strat_fold"].astype(int), meta_df["label"].astype(str))
    cols = [c for c in ORDER_5 if c in ct.columns] or list(ct.columns)
    ct = ct[cols]
    _savetab(ct, outdir / "tables", "table_class_by_fold_counts.csv")

    folds = ct.index.astype(str).tolist()
    x = _x_positions(len(folds), 1.00)

    k = len(cols)
    group_width = 0.84
    bar_w = group_width / max(k, 1)
    eff_w = bar_w * (1 - GAP_FRAC_GROUPED)

    fig, ax = plt.subplots(figsize=(12.0, 5.9))
    ymax = 0.0
    for j, lab in enumerate(cols):
        offs = -group_width/2 + j*bar_w + bar_w/2
        vals = ct[lab].values
        ymax = max(ymax, float(np.nanmax(vals)) if len(vals) else 0.0)
        ax.bar(
            x + offs, vals, width=eff_w,
            label=str(lab), color=CLASS_COLORS.get(lab, None), edgecolor="white", linewidth=0.6, alpha=0.92
        )
        _annot_bars(ax, x + offs, vals)
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.set_xlabel("strat_fold")
    ax.set_ylabel("Count")
    ax.set_title("Class counts per strat_fold (grouped)")
    _prettify_axes(ax)
    ax.set_ylim(0, ymax * 1.30)
    ax.legend(ncol=min(5, len(cols)), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10), handlelength=1.2, columnspacing=1.2)
    plt.tight_layout()
    savefig(fig, outdir, "eda_fold_class_counts_grouped_pretty.png")

# --------------------------------------------------------------------------------------
# Engineered-features EDA: correlation heatmaps & pruning
# --------------------------------------------------------------------------------------

def plot_feature_correlations(feature_df: pd.DataFrame, outdir: Path):
    if feature_df.empty:
        return
    tables_dir = outdir / "tables"
    num = feature_df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        print("(Not enough numeric columns for correlation heatmap.)")
        return

    import re as _re

    def parse_col(c):
        m = _re.match(r"L(\d+)_(mean|std|rms|ptp)$", c)
        if m:
            return ("stat", m.group(2), int(m.group(1)))
        m = _re.match(r"L(\d+)_bp_(\d+)_(\d+)Hz$", c)
        if m:
            band = f"bp{m.group(2)}_{m.group(3)}"
            return ("bp", band, int(m.group(1)))
        return ("other", c, 999)

    families = [("stat","mean"),("stat","std"),("stat","rms"),("stat","ptp"),
                ("bp","bp0_5"),("bp","bp5_15"),("bp","bp15_40")]

    cols = list(num.columns)
    parsed = [parse_col(c) for c in cols]
    order = []
    for kind, fam in families:
        for lead in range(1, 13):
            for c, p in zip(cols, parsed):
                if p[0]==kind and p[1]==fam and p[2]==lead:
                    order.append(c)
    order += [c for c in cols if c not in order]

    corr = num[order].corr(method="spearman").round(3).abs()
    _savetab(corr, tables_dir, "table_feature_corr_spearman.csv")

    fig, ax = plt.subplots(figsize=(9.5, 8.0))
    im = ax.imshow(corr.values, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("|r|")
    ax.set_title("Feature correlation heatmap — grouped by family (|r|)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    savefig(fig, outdir, "eda_feature_corr_heatmap_grouped_simple.png")

    # Prune highly correlated features
    C = corr.copy()
    keep = []
    dropped = []
    seen = set()
    thr = 0.95
    for c in C.columns:
        if c in seen:
            continue
        keep.append(c)
        seen.add(c)
        drop_mask = (C[c] >= thr) & (C.index != c)
        for d in C.index[drop_mask]:
            if d not in seen:
                seen.add(d)
                dropped.append((d, c, float(C.loc[d, c])))

    num_p = num[keep]
    corr_p = num_p.corr(method="spearman").round(3).abs()
    _savetab(pd.DataFrame({"kept_features": keep}), tables_dir, "table_feature_corr_pruned_kept.csv")
    _savetab(pd.DataFrame(dropped, columns=["dropped","kept_with","|r|"]).sort_values("|r|", ascending=False),
             tables_dir, "table_feature_corr_pruned_dropped.csv")
    _savetab(corr_p, tables_dir, "table_feature_corr_pruned_spearman.csv")

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr_p.values, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax).set_label("|r|")
    ax.set_title(f"Correlation heatmap after pruning (|r|≥0.95) — {num_p.shape[1]} features kept")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    savefig(fig, outdir, "eda_feature_corr_heatmap_pruned.png")

# --------------------------------------------------------------------------------------
# HR/HRV boxplots by class/sex (from engineered features table)
# --------------------------------------------------------------------------------------

def plot_hrv_boxes(feature_df: pd.DataFrame, outdir: Path):
    if feature_df.empty:
        return
    df = feature_df.copy()
    for c in ["HR_bpm","SDNN_ms","RMSSD_ms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    METRICS = [c for c in ["HR_bpm","SDNN_ms","RMSSD_ms"] if c in df.columns]
    if not METRICS:
        print("(HR/HRV columns not found in features CSV; skipping HRV boxplots.)")
        return

    # Summary tables
    tables_dir = (outdir / "tables")
    if "label" in df.columns:
        g = df.groupby("label")[METRICS].agg(["count","mean","std","median"])
        # flatten multiindex
        g.columns = [f"{m}_{stat}" for m, stat in g.columns]
        _savetab(g.sort_index(), tables_dir, "table_hrv_summary_by_class.csv")

    sex_col = None
    for c in ["sex_norm", "sex"]:
        if c in df.columns:
            sex_col = c
            break
    if sex_col is not None:
        if sex_col == "sex":
            df["sex_norm"] = _sex_norm(df["sex"])
            sex_col = "sex_norm"
        g2 = df.groupby(sex_col)[METRICS].agg(["count","mean","std","median"])
        g2.columns = [f"{m}_{stat}" for m, stat in g2.columns]
        _savetab(g2, tables_dir, "table_hrv_summary_by_sex.csv")

    def _boxplot(df: pd.DataFrame, metric: str, group_col: str, order=None, title=None, fname=None, figsize=(10,5)):
        g_order = (order or sorted(df[group_col].dropna().astype(str).unique()))
        groups, labels = [], []
        for g in g_order:
            v = pd.to_numeric(df.loc[df[group_col].astype(str)==g, metric], errors="coerce").dropna().values
            if v.size >= 5:
                groups.append(v)
                labels.append(str(g))
        if not groups:
            return
        fig, ax = plt.subplots(figsize=figsize)
        ax.boxplot(groups, tick_labels=labels, showfliers=False)  # CHANGED
        # annotate n & med
        ymin, ymax = ax.get_ylim(); dy = 0.03*(ymax - ymin)
        for i, arr in enumerate(groups, start=1):
            ax.text(i, np.percentile(arr, 75) + dy, f"n={len(arr):,}\nmed={np.median(arr):.1f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_title(title or f"{metric} by {group_col}")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        savefig(fig, outdir, fname)

    if "label" in df.columns:
        class_order = [c for c in ["CD","HYP","MI","NORM","STTC"] if c in set(df["label"].astype(str))] or sorted(df["label"].astype(str).unique())
        for m in METRICS:
            _boxplot(df, m, "label", order=class_order, title=f"{m} by diagnostic class", fname=f"box_{m}_by_class_side.png")

    if sex_col is not None:
        sex_order = [s for s in ["female","male","unknown"] if s in set(df[sex_col].astype(str))]
        for m in METRICS:
            _boxplot(df, m, sex_col, order=sex_order, title=f"{m} by sex", fname=f"box_{m}_by_sex_side.png", figsize=(7.5,4.8))

# --------------------------------------------------------------------------------------
# ANOVA F-test on engineered features (SelectKBest) — robust to NaNs & constants
# --------------------------------------------------------------------------------------

def run_anova_selectkbest(feature_df: pd.DataFrame, outdir: Path):
    try:
        from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import LabelEncoder
    except Exception:
        print("(sklearn not available — skipping ANOVA SelectKBest)")
        return
    if feature_df.empty or "label" not in feature_df.columns:
        return

    X_num = feature_df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number]).copy()
    y_all = feature_df["label"].astype(str).copy()
    X_num = X_num.replace([np.inf, -np.inf], np.nan)

    vt = VarianceThreshold(threshold=1e-12)
    try:
        X_vt = vt.fit_transform(X_num)
    except Exception:
        print("(ANOVA) No numeric features after preprocessing.")
        return
    cols_vt = X_num.columns[vt.get_support()]
    if X_vt.shape[1] == 0:
        print("(ANOVA) All numeric features were constant or removed.")
        return

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_vt)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_all.values)

    k = int(min(20, X_imp.shape[1]))
    selector = SelectKBest(score_func=f_classif, k=k).fit(X_imp, y_enc)
    scores_series = pd.Series(selector.scores_, index=cols_vt).sort_values(ascending=False)

    # Save numbers
    _savetab(scores_series.rename("F_score"), outdir / "tables", "table_anova_top.csv")

    top = scores_series.head(25)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = np.arange(len(top))
    ax.bar(xs, top.values)
    ax.set_xticks(xs)
    ax.set_xticklabels(top.index, rotation=65, ha="right")
    ax.set_title("ANOVA F-scores (top 25)")
    ax.set_ylabel("F-score")
    ax.grid(axis="y", alpha=0.3)
    # labels
    _annot_bars(ax, xs, top.values, fmt="{:.1f}", dy_frac=0.015, fontsize=9)
    plt.tight_layout()
    savefig(fig, outdir, "anova_selectkbest_top25.png")

# --------------------------------------------------------------------------------------
# Optional: confusion heatmaps from FL CSVs (if present) + tables
# --------------------------------------------------------------------------------------

def try_plot_confusions_from_csv(cm_csv: Path, superclasses: Iterable[str], name: str, outdir: Path, last_k: int = 3):
    if not cm_csv.exists():
        print(f"Missing: {cm_csv}")
        return
    df = pd.read_csv(cm_csv)
    req = {"round","true","pred","count"}
    if not req.issubset(set(df.columns.str.lower())):
        print(f"Confusion CSV {cm_csv} missing columns {req}")
        return

    # normalize column cases
    m = {c.lower(): c for c in df.columns}
    df = df.rename(columns={m.get("round"): "round", m.get("true"): "true", m.get("pred"): "pred", m.get("count"): "count"})

    rounds = sorted(df["round"].dropna().unique())[-int(last_k):]
    sc = list(superclasses)
    n = len(sc)
    for r in rounds:
        sub = df[df["round"] == r].copy()
        piv = sub.pivot_table(index="true", columns="pred", values="count", aggfunc="sum", fill_value=0)
        piv.index = pd.to_numeric(piv.index, errors="coerce")
        piv.columns = pd.to_numeric(piv.columns, errors="coerce")
        cm_counts = (piv.reindex(index=range(n), columns=range(n), fill_value=0).astype(int))
        _savetab(cm_counts, outdir, f"confusion_{name}_r{int(r):02d}_counts.csv")

        cm = cm_counts.to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            row_sum = cm.sum(axis=1, keepdims=True)
            cmn = np.nan_to_num(cm / np.maximum(row_sum, 1), nan=0.0, posinf=0.0, neginf=0.0)
        _savetab(pd.DataFrame(cmn * 100, index=sc, columns=sc).round(2),
                 outdir, f"confusion_{name}_r{int(r):02d}_percent.csv")

        fig, ax = plt.subplots(figsize=(6.4, 5.3))
        im = ax.imshow(cmn, aspect="auto", vmin=0, vmax=1.0, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {name} (round {int(r)})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(n)); ax.set_xticklabels(sc, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(sc)
        cbar = fig.colorbar(im, ax=ax); cbar.set_label("Row-normalized")
        for i in range(n):
            for j in range(n):
                val = cmn[i, j] * 100.0
                ax.text(j, i, f"{val:0.2f}", ha="center", va="center",
                        color=("white" if val >= 50 else "black"), fontsize=9)
        plt.tight_layout()
        savefig(fig, outdir, f"confusion_{name}_r{int(r):02d}.png")

# --------------------------------------------------------------------------------------
# Driver: orchestrate everything
# --------------------------------------------------------------------------------------

def run_eda_and_optional_fl(args):
    outdir = ensure_outdir(Path(args.outdir))
    eda_dir = ensure_outdir(outdir / "eda")

    # 1) Load PTB-XL DB/SCP and build minimal metadata table (features_df)
    db_csv = Path(args.db_csv) if args.db_csv else DEFAULT_DB_CSV
    scp_csv = Path(args.scp_csv) if args.scp_csv else DEFAULT_SCP_CSV
    db, scp, _ = load_db_and_scp(db_csv, scp_csv)
    features_df = pd.DataFrame()
    if not db.empty and not scp.empty:
        to_label, valid_codes, diag_only, label_field = make_label_mapper(scp, label_mode="5class")
        try:
            features_df = build_minimal_table(db, to_label, record_file_col=args.record_file_col, min_conf=float(args.scp_min_conf))
        except Exception as e:
            print(f"(Minimal table) failed: {e}")
    else:
        print("(PTB-XL CSVs missing — skipping demographics/fold EDA that relies on them.)")

    # 2) Load engineered features (feature_df)
    feature_df = pd.DataFrame()
    feats_path = Path(args.features_csv) if args.features_csv else None
    if feats_path is None:
        for c in DEFAULT_FEATURES_CSV_CANDIDATES:
            if c.exists():
                feats_path = c
                break
    if feats_path is None:
        print("Missing: basic_signal_features.csv (engineered features) — correlation & HRV EDA will be skipped.")
    else:
        try:
            feature_df = pd.read_csv(feats_path, index_col=0)
        except Exception as e:
            print(f"Failed to read features CSV ({feats_path}): {e}")

    # 3) EDA plots (mostly notebook parity + numbers)
    if not features_df.empty:
        plot_missingness_top(features_df, eda_dir)
        plot_class_counts(features_df, eda_dir)
        plot_sex_overall_and_by_class(features_df, eda_dir)
        plot_age_distributions(features_df, eda_dir)
        plot_records_per_year(features_df, eda_dir)
        plot_strat_fold_grouped(features_df, eda_dir)
    else:
        print("(Minimal metadata table unavailable — many EDA plots will be skipped.)")

    if not feature_df.empty:
        plot_feature_correlations(feature_df, eda_dir)
        plot_hrv_boxes(feature_df, eda_dir / "hrv")
        run_anova_selectkbest(feature_df, eda_dir)

    # 4) Optional: Confusion matrices from FL CSVs (if present)
    SUPERCLASSES = [c for c in ORDER_5]
    results_dir = Path(args.results_dir) if args.results_dir else Path("results")
    try_plot_confusions_from_csv(results_dir / "non_frozen_run_cm.csv", SUPERCLASSES, "non_frozen", outdir)
    try_plot_confusions_from_csv(results_dir / "frozen_run_cm.csv", SUPERCLASSES, "frozen", outdir)

    print(f"Saved plots to: {outdir}")

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="PTB-XL EDA + optional FL results visualization")
    p.add_argument("--dataset-root", type=str, default=str(DEFAULT_DATASET_ROOT), help="Root folder for PTB-XL dataset")
    p.add_argument("--db-csv", type=str, default=str(DEFAULT_DB_CSV), help="Path to ptbxl_database.csv")
    p.add_argument("--scp-csv", type=str, default=str(DEFAULT_SCP_CSV), help="Path to scp_statements.csv")
    p.add_argument("--record-file-col", type=str, default="filename_lr", choices=["filename_lr","filename_hr"], help="Which PTB-XL path column to use")
    p.add_argument("--scp-min-conf", type=float, default=0.0, help="Min confidence for SCP code to count toward label")
    p.add_argument("--features-csv", type=str, default=None, help="Engineered features CSV (from notebook §5c)")
    p.add_argument("--results-dir", type=str, default="results", help="Folder where FL CSVs live")
    p.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR), help="Where to save figures and tables")
    return p

def main():
    args = build_argparser().parse_args()
    run_eda_and_optional_fl(args)

if __name__ == "__main__":
    main()