# src_Connection/results_visualization.py
"""
Results & EDA Visualizations (robust, pretty, and with numbers)

Usage (from repo root):
  py -m src_Connection.results_visualization \
      --results-dir results \
      --viz-dir results/viz \
      --features-csv test/artifacts/basic_signal_features.csv

What it does:
  1) Reads training result CSVs (non_frozen/frozen runs) if available and makes:
      - Global accuracy over rounds
      - Client-vs-global accuracy per round (if per-client columns exist)
      - Wall-time per round (if time columns exist)
      - Per-class accuracy over rounds (line & final bar)
      - Confusion matrix for the last round (if CM file exists)
      - Global & client accuracy by phase (if a 'phase' column exists)
     All figures are saved under <viz-dir>.

  2) Pretty EDA with numbers, based on your engineered features CSV:
      - Class × strat_fold grouped bars (+ CSV)
      - Class × sex counts and normalized (+ CSV)
      - Age hist, overlay by sex, age-by-class & age-by-sex boxplots (+ CSV summaries)
      - HR/HRV boxplots by class (HR_bpm, SDNN_ms, RMSSD_ms if present)
      - ANOVA (SelectKBest f_classif) top-k features bar (+ CSV)
     Figures go to <viz-dir>/eda/figs and tables to <viz-dir>/eda/tables.

This script is intentionally defensive: if a file/column is missing, it skips that plot
and keeps going, printing a short explanation instead of crashing.
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# sklearn is only needed for the EDA-ANOVA block
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# ==========================
# General helpers & styling
# ==========================

def _set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.titleweight": "semibold",
        "axes.titlesize": 15,
        "axes.labelsize": 12.5,
        "legend.frameon": False,
        "legend.fontsize": 11.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _fmt_int(x, _pos=None) -> str:
    try:
        xi = float(x)
        if xi >= 1000:
            return f"{int(xi):,}"
        if xi.is_integer():
            return str(int(xi))
        return f"{xi:.1f}"
    except Exception:
        return str(x)

def _savefig(fig, out_dir: Path, name: str):
    _ensure_dir(out_dir)
    fp = out_dir / name
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    print(f"Saved: {fp}")

def _savetab(df: pd.DataFrame | pd.Series, out_dir: Path, name: str):
    _ensure_dir(out_dir)
    fp = out_dir / name
    (df.to_frame() if isinstance(df, pd.Series) else df).to_csv(fp)
    print(f"Saved table: {fp}")

def _load_csv_or_none(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"Missing: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Failed reading {path}: {e}")
        return None

def _guess_round_col(df: pd.DataFrame) -> str:
    for cand in ["round", "epoch", "step", "r"]:
        if cand in df.columns:
            return cand
    return df.columns[0]

def _guess_global_acc_col(df: pd.DataFrame) -> Optional[str]:
    pats = [r"^global.*acc", r"^acc(_global)?$", r"accuracy_global", r"^global_accuracy$"]
    cols = [c for c in df.columns]
    for p in pats:
        for c in cols:
            if re.search(p, c, re.I):
                return c
    return None

def _client_acc_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if re.search(r"client.*acc", c, re.I)]
    # allow patterns client_0_acc, client1_accuracy, etc.
    cols += [c for c in df.columns if re.search(r"^acc_client", c, re.I)]
    # de-duplicate preserving order
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            out.append(c); seen.add(c)
    return out

def _time_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if re.search(r"(wall.*time|time.*sec|sec.*per.*round|round.*time)", c, re.I):
            return c
    return None


# ==========================
#   Training results plots
# ==========================

CLASS_COLORS = {"NORM": "#4C78A8", "MI": "#F58518", "CD": "#54A24B", "STTC": "#E45756", "HYP": "#B279A2"}

def plot_global_accuracy(df_run: pd.DataFrame, tag: str, out_dir: Path):
    _set_style()
    rcol = _guess_round_col(df_run)
    acol = _guess_global_acc_col(df_run)
    if acol is None:
        print(f"[{tag}] global accuracy column not found — skipping.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_run[rcol], df_run[acol], marker="o", linewidth=1.4)
    ax.set_title("Global accuracy over rounds")
    ax.set_xlabel(rcol); ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, out_dir, f"global_accuracy_{tag}.png")

def plot_clients_vs_global(df_run: pd.DataFrame, tag: str, out_dir: Path):
    _set_style()
    rcol = _guess_round_col(df_run)
    acol = _guess_global_acc_col(df_run)
    ccols = _client_acc_cols(df_run)
    if not ccols:
        print(f"[{tag}] per-client accuracy columns not found — skipping clients_vs_global.")
        return
    fig, ax = plt.subplots(figsize=(11, 5.4))
    for c in ccols:
        ax.plot(df_run[rcol], df_run[c], linewidth=1.0, alpha=0.8, label=c)
    if acol and acol in df_run:
        ax.plot(df_run[rcol], df_run[acol], linewidth=2.4, label="global", color="black")
    ax.set_title("Clients vs Global accuracy")
    ax.set_xlabel(rcol); ax.set_ylabel("Accuracy")
    ax.legend(ncol=2)
    plt.tight_layout()
    _savefig(fig, out_dir, f"clients_vs_global_{tag}.png")

def plot_walltime(df_run: pd.DataFrame, tag: str, out_dir: Path):
    _set_style()
    rcol = _guess_round_col(df_run)
    tcol = _time_col(df_run)
    if tcol is None:
        print(f"[{tag}] wall-time column not found — skipping walltime.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df_run[rcol], df_run[tcol], marker="o")
    ax.set_title("Wall-time per round")
    ax.set_xlabel(rcol); ax.set_ylabel(tcol)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_int))
    plt.tight_layout()
    _savefig(fig, out_dir, f"walltime_{tag}.png")

def plot_perclass_over_rounds(df_pc: pd.DataFrame, tag: str, out_dir: Path):
    _set_style()
    # Try to normalize schema: expect columns like ["round","class","acc"]
    cols = {c.lower(): c for c in df_pc.columns}
    rcol = cols.get("round") or cols.get("epoch") or list(df_pc.columns)[0]
    class_col = None
    for k in ["class", "label", "diagnosis", "category"]:
        if k in cols:
            class_col = cols[k]; break
    acc_col = None
    for k in ["acc", "accuracy", "score"]:
        if k in cols:
            acc_col = cols[k]; break
    if class_col is None or acc_col is None:
        print(f"[{tag}] per-class CSV missing class/acc columns — skipping perclass plots.")
        return

    # line plot over rounds
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    for lab, g in df_pc.groupby(class_col):
        g = g.sort_values(rcol)
        color = CLASS_COLORS.get(str(lab), None)
        ax.plot(g[rcol], g[acc_col], marker="o", linewidth=1.2, label=str(lab), color=color)
    ax.set_title("Per-class accuracy over rounds")
    ax.set_xlabel(rcol); ax.set_ylabel("Accuracy")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _savefig(fig, out_dir, "perclass_over_rounds.png")

    # final bar at last round
    last_r = df_pc[rcol].max()
    fin = df_pc[df_pc[rcol] == last_r].copy()
    order = list(fin[class_col].astype(str))
    xs = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    cols = [CLASS_COLORS.get(o, None) for o in order]
    ax.bar(xs, fin[acc_col].values, color=cols, edgecolor="white", linewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(order)
    ax.set_title(f"Per-class accuracy (round {last_r})"); ax.set_ylabel("Accuracy")
    ymax = float(fin[acc_col].max())
    for x, y in zip(xs, fin[acc_col].values):
        ax.text(x, y + max(0.005, 0.02*ymax), f"{y:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    _savefig(fig, out_dir, "perclass_final_bar.png")

def plot_confusion(df_cm: pd.DataFrame, tag: str, out_dir: Path):
    _set_style()
    # Try to support two shapes:
    # (A) long: columns [round, true, pred, count]
    # (B) wide: index=true, columns=pred, values=count (maybe last round only)
    cols_lower = [c.lower() for c in df_cm.columns]

    if set(["round","true","pred","count"]).issubset(cols_lower):
        # long form
        m = {c.lower(): c for c in df_cm.columns}
        last_r = df_cm[m["round"]].max()
        dfl = df_cm[df_cm[m["round"]] == last_r]
        pivot = dfl.pivot(index=m["true"], columns=m["pred"], values=m["count"]).fillna(0)
        classes = list(pivot.index.astype(str))
        cm = pivot.loc[classes, classes].values
        title = f"Confusion matrix (round {last_r})"
    else:
        # try wide form
        dfw = df_cm.set_index(df_cm.columns[0])
        dfw = dfw.reindex(sorted(dfw.index.astype(str))).fillna(0)
        classes = list(dfw.index.astype(str))
        cm = dfw.values
        title = "Confusion matrix"

    cm_pct = cm.astype(float)
    row_sum = cm_pct.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm_pct, np.maximum(row_sum, 1), where=row_sum>0) * 100.0

    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    im = ax.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    cb = fig.colorbar(im, ax=ax); cb.set_label("%")
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes); ax.set_yticklabels(classes)
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            v = cm_pct[i, j]; ax.text(j, i, f"{v:0.1f}", ha="center",
                                       va="center", color=("white" if v >= 50 else "black"), fontsize=10)
    plt.tight_layout()
    _savefig(fig, out_dir, f"confusion_{tag}_rLast.png")

def plot_by_phase(df_run: pd.DataFrame, tag: str, out_dir: Path):
    _set_style()
    if "phase" not in df_run.columns:
        print(f"[{tag}] 'phase' column not found — skipping phase plots.")
        return
    rcol = _guess_round_col(df_run)
    acol = _guess_global_acc_col(df_run)
    if acol is None:
        return
    # global by phase
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for ph, g in df_run.groupby("phase"):
        g = g.sort_values(rcol)
        ax.plot(g[rcol], g[acol], marker="o", linewidth=1.2, label=str(ph))
    ax.set_title("Global accuracy by phase"); ax.set_xlabel(rcol); ax.set_ylabel("Accuracy")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, out_dir, f"global_by_phase_{tag}.png")

    # clients by phase if per-client cols exist
    ccols = _client_acc_cols(df_run)
    if not ccols:
        return
    fig, ax = plt.subplots(figsize=(11.8, 5.4))
    for ph, g in df_run.groupby("phase"):
        vals = g[ccols].mean(axis=1)
        ax.plot(g[rcol], vals, linewidth=1.2, label=str(ph))
    ax.set_title("Mean client accuracy by phase")
    ax.set_xlabel(rcol); ax.set_ylabel("Mean client acc")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, out_dir, f"clients_by_phase_{tag}.png")


def handle_training_results(results_dir: Path, viz_dir: Path):
    """
    Looks for these files (each optional):
      <tag>_run.csv, <tag>_run_perclass.csv, <tag>_run_cm.csv
    for tag in {"non_frozen","frozen"}.
    """
    for tag in ["non_frozen", "frozen"]:
        run_csv   = results_dir / f"{tag}_run.csv"
        perclass  = results_dir / f"{tag}_run_perclass.csv"
        cm_csv    = results_dir / f"{tag}_run_cm.csv"

        df_run = _load_csv_or_none(run_csv)
        df_pc  = _load_csv_or_none(perclass)
        df_cm  = _load_csv_or_none(cm_csv)

        if df_run is not None:
            plot_global_accuracy(df_run, tag, viz_dir)
            plot_clients_vs_global(df_run, tag, viz_dir)
            plot_walltime(df_run, tag, viz_dir)
            plot_by_phase(df_run, tag, viz_dir)

        if df_pc is not None:
            plot_perclass_over_rounds(df_pc, tag, viz_dir)

        if df_cm is not None:
            plot_confusion(df_cm, tag, viz_dir)


# ==========================
#         Pretty EDA
# ==========================

SEX_COLORS   = {"female": "#e48ec4", "male": "#8bb1f0", "unknown": "#bdbdbd"}

def _sex_norm(series: pd.Series) -> pd.Series:
    out = pd.Series("unknown", index=series.index, dtype="string")
    s_num = pd.to_numeric(series, errors="coerce")
    out.loc[s_num == 1] = "male"; out.loc[s_num == 0] = "female"
    s = series.astype("string").str.strip().str.lower()
    out.loc[s.str.startswith("m", na=False)] = "male"
    out.loc[s.str.startswith("f", na=False)] = "female"
    out.loc[s.isin({"male","man"})] = "male"
    out.loc[s.isin({"female","woman"})] = "female"
    return out

def _annot_bars(ax, xs, ys, fmt="{:,}", dy_frac=0.02, fontsize=10):
    ymax = float(np.nanmax(ys)) if len(ys) else 1.0
    dy = max(1.0, ymax * dy_frac)
    for x, y in zip(xs, ys):
        ax.text(x, y + dy, fmt.format(int(y)) if np.isfinite(y) else "", ha="center", va="bottom", fontsize=fontsize)

def plot_class_by_fold_E(edf: pd.DataFrame, out_dir: Path):
    if "label" not in edf.columns or "strat_fold" not in edf.columns:
        print("[EDA] need 'label' and 'strat_fold' — skipping class_by_fold.")
        return
    _set_style()
    labs = edf["label"].astype(str)
    order = [c for c in ["NORM","MI","CD","STTC","HYP"] if c in set(labs)] or sorted(labs.unique())
    ct = pd.crosstab(edf["strat_fold"].astype(int), labs).reindex(columns=order).fillna(0).astype(int)
    _savetab(ct, out_dir / "tables", "table_class_by_fold_counts.csv")

    x = np.arange(len(ct.index), dtype=float)
    k = len(order); group_w = 0.84; bar_w = group_w / max(1, k); eff_w = bar_w * 0.90
    fig, ax = plt.subplots(figsize=(12, 6))
    ymax = 0
    for j, lab in enumerate(order):
        offs = -group_w/2 + j*bar_w + bar_w/2
        vals = ct[lab].values
        ymax = max(ymax, float(np.nanmax(vals)) if len(vals) else 0)
        ax.bar(x + offs, vals, width=eff_w, label=lab,
               color=CLASS_COLORS.get(lab), edgecolor="white", linewidth=0.6)
        _annot_bars(ax, x + offs, vals, dy_frac=0.018)
    ax.set_xticks(x); ax.set_xticklabels(ct.index.astype(str))
    ax.set_ylabel("Count"); ax.set_title("Class counts per strat_fold (grouped)")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_int))
    ax.set_ylim(0, ymax * 1.22)
    ax.legend(ncol=min(5, k), loc="upper center", bbox_to_anchor=(0.5, 1.12))
    plt.tight_layout()
    _savefig(fig, out_dir / "figs", "eda_fold_class_counts_grouped_pretty.png")

def plot_class_by_sex_E(edf: pd.DataFrame, out_dir: Path):
    if "label" not in edf.columns:
        print("[EDA] 'label' missing — skipping class_by_sex.")
        return
    _set_style()
    if "sex_norm" not in edf.columns:
        if "sex" in edf.columns:
            edf = edf.copy(); edf["sex_norm"] = _sex_norm(edf["sex"])
        else:
            print("[EDA] 'sex' not available — skipping class_by_sex.")
            return

    labs = edf["label"].astype(str)
    order = [c for c in ["NORM","MI","CD","STTC","HYP"] if c in set(labs)] or sorted(labs.unique())
    ct = pd.crosstab(labs, edf["sex_norm"]).reindex(index=order).fillna(0).astype(int)
    ct = ct[[c for c in ["female","male","unknown"] if c in ct.columns]]
    _savetab(ct, out_dir / "tables", "table_class_by_sex_counts.csv")

    # counts
    x = np.arange(len(ct.index)); k = ct.shape[1]
    group_w = 0.84; bar_w = group_w / max(1, k); eff_w = bar_w * 0.90
    fig, ax = plt.subplots(figsize=(12, 6))
    ymax = 0
    for j, col in enumerate(ct.columns):
        offs = -group_w/2 + j*bar_w + bar_w/2
        vals = ct[col].values
        ymax = max(ymax, float(np.nanmax(vals)) if len(vals) else 0)
        ax.bar(x + offs, vals, width=eff_w, label=col,
               color=SEX_COLORS.get(col), edgecolor="white", linewidth=0.6)
        _annot_bars(ax, x + offs, vals, dy_frac=0.018)
    ax.set_xticks(x); ax.set_xticklabels(ct.index)
    ax.set_ylabel("Count"); ax.set_title("Class × Sex (counts)")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_int))
    ax.set_ylim(0, ymax * 1.18)
    ax.legend()
    plt.tight_layout()
    _savefig(fig, out_dir / "figs", "eda_class_by_sex_counts_pretty.png")

    # percentages
    pct = (ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(2)
    _savetab(pct, out_dir / "tables", "table_class_by_sex_percent.csv")
    fig, ax = plt.subplots(figsize=(12, 6))
    for j, col in enumerate(pct.columns):
        offs = -group_w/2 + j*bar_w + bar_w/2
        vals = pct[col].values
        ax.bar(x + offs, vals, width=eff_w, label=col,
               color=SEX_COLORS.get(col), edgecolor="white", linewidth=0.6)
        _annot_bars(ax, x + offs, vals, fmt="{:.0f}%", dy_frac=0.012)
    ax.set_xticks(x); ax.set_xticklabels(pct.index)
    ax.set_ylabel("Percentage (%)"); ax.set_ylim(0, 100)
    ax.set_title("Class × Sex (normalized)")
    ax.legend()
    plt.tight_layout()
    _savefig(fig, out_dir / "figs", "eda_class_by_sex_percent_pretty.png")

def _fd_bins(x, min_bins=24, max_bins=64):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0: return min_bins
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr <= 0: return min(max_bins, max(min_bins, int(np.sqrt(x.size))))
    bw = 2 * iqr * (x.size ** (-1/3))
    if bw <= 0: return min(max_bins, max(min_bins, int(np.sqrt(x.size))))
    bins = int(np.ceil((x.max() - x.min()) / bw))
    return int(np.clip(bins, min_bins, max_bins))

def plot_age_E(edf: pd.DataFrame, out_dir: Path):
    if "age" not in edf.columns:
        print("[EDA] 'age' missing — skipping age plots.")
        return
    _set_style()
    a = pd.to_numeric(edf["age"], errors="coerce").dropna().values
    if a.size == 0:
        print("[EDA] no numeric age — skipping.")
        return
    bins = _fd_bins(a)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.hist(a, bins=bins, rwidth=0.92, edgecolor="white", linewidth=0.6)
    mu, med = float(np.mean(a)), float(np.median(a))
    ax.axvline(mu,  linestyle="--", linewidth=1.4, color="k", label=f"Mean {mu:.1f}")
    ax.axvline(med, linestyle=":",  linewidth=1.6, color="k", label=f"Median {med:.1f}")
    ax.set_title("Age distribution"); ax.set_xlabel("Age"); ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_int))
    ax.legend(ncol=2, loc="upper right")
    txt = f"n = {len(a):,}\nμ = {mu:.1f}\nmed = {med:.1f}\nσ = {np.std(a, ddof=1):.1f}"
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9))
    plt.tight_layout()
    _savefig(fig, out_dir / "figs", "eda_age_hist_overall_pretty.png")

    # overlay by sex
    if "sex_norm" not in edf.columns and "sex" in edf.columns:
        edf = edf.copy(); edf["sex_norm"] = _sex_norm(edf["sex"])
    if "sex_norm" in edf.columns:
        af = pd.to_numeric(edf.loc[edf["sex_norm"]=="female","age"], errors="coerce").dropna().values
        am = pd.to_numeric(edf.loc[edf["sex_norm"]=="male","age"], errors="coerce").dropna().values
        if af.size or am.size:
            base = a if a.size else (np.concatenate([af, am]) if (af.size and am.size) else (af if af.size else am))
            bins = _fd_bins(base)
            fig, ax = plt.subplots(figsize=(14, 6))
            if af.size: ax.hist(af, bins=bins, rwidth=0.88, alpha=0.55, label="female",
                                 color=SEX_COLORS["female"], edgecolor="white", linewidth=0.6)
            if am.size: ax.hist(am, bins=bins, rwidth=0.88, alpha=0.55, label="male",
                                 color=SEX_COLORS["male"], edgecolor="white", linewidth=0.6)
            ax.set_title("Age distribution by sex (overlay)")
            ax.set_xlabel("Age"); ax.set_ylabel("Count"); ax.legend(ncol=2)
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt_int))
            plt.tight_layout()
            _savefig(fig, out_dir / "figs", "eda_age_hist_by_sex_overlay_pretty.png")

    # box by class
    if "label" in edf.columns:
        labs = edf["label"].astype(str)
        order = [c for c in ["NORM","MI","CD","STTC","HYP"] if c in set(labs)] or sorted(labs.unique())
        data_c, labels = [], []
        for c in order:
            v = pd.to_numeric(edf.loc[labs==c, "age"], errors="coerce").dropna().values
            if v.size >= 5: data_c.append(v); labels.append(c)
        if data_c:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.boxplot(data_c, labels=labels, showfliers=False)
            ax.set_title("Age by diagnostic class"); ax.set_ylabel("Age")
            ymin, ymax = ax.get_ylim(); dy = 0.03*(ymax-ymin)
            for i, arr in enumerate(data_c, start=1):
                med = float(np.median(arr)); q3 = float(np.percentile(arr, 75))
                ax.text(i, q3 + dy, f"n={len(arr):,}\nmed={med:.1f}",
                        ha="center", va="bottom", fontsize=9)
            plt.tight_layout()
            _savefig(fig, out_dir / "figs", "eda_age_by_class_box_pretty.png")

    # box by sex
    if "sex_norm" in edf.columns:
        groups, labels = [], []
        for s in ["female","male","unknown"]:
            v = pd.to_numeric(edf.loc[edf["sex_norm"]==s,"age"], errors="coerce").dropna().values
            if v.size >= 5: groups.append(v); labels.append(s)
        if groups:
            fig, ax = plt.subplots(figsize=(10, 6))
            bp = ax.boxplot(groups, labels=labels, showfliers=False, patch_artist=True)
            for i, s in enumerate(labels):
                c = SEX_COLORS.get(s, "#999")
                bp["boxes"][i].set_facecolor(c); bp["boxes"][i].set_alpha(0.25)
                bp["boxes"][i].set_edgecolor(c)
            ax.set_title("Age by sex"); ax.set_ylabel("Age")
            ymin, ymax = ax.get_ylim(); dy = 0.03*(ymax-ymin)
            for i, arr in enumerate(groups, start=1):
                med = float(np.median(arr)); q3 = float(np.percentile(arr, 75))
                ax.text(i, q3 + dy, f"n={len(arr):,}\nmed={med:.1f}",
                        ha="center", va="bottom", fontsize=9)
            plt.tight_layout()
            _savefig(fig, out_dir / "figs", "eda_age_by_sex_box_pretty.png")

    # tables of numbers
    labs = edf["label"].astype(str) if "label" in edf.columns else pd.Series(index=edf.index, dtype=str)
    order = sorted(labs.dropna().unique())
    by_class = {
        c: pd.to_numeric(edf.loc[labs==c, "age"], errors="coerce").dropna().values for c in order
    }
    age_stats = pd.DataFrame({
        c: pd.Series({"n": len(v),
                      "mean": np.mean(v) if len(v) else np.nan,
                      "std": np.std(v, ddof=1) if len(v)>1 else np.nan,
                      "median": np.median(v) if len(v) else np.nan})
        for c, v in by_class.items()
    }).T
    if len(age_stats):
        _savetab(age_stats, out_dir / "tables", "table_age_summary_by_class.csv")

    if "sex_norm" in edf.columns:
        by_sex = {
            s: pd.to_numeric(edf.loc[edf["sex_norm"]==s, "age"], errors="coerce").dropna().values
            for s in ["female","male","unknown"] if s in set(edf["sex_norm"])
        }
        age_sex = pd.DataFrame({
            s: pd.Series({"n": len(v),
                          "mean": np.mean(v) if len(v) else np.nan,
                          "std": np.std(v, ddof=1) if len(v)>1 else np.nan,
                          "median": np.median(v) if len(v) else np.nan})
            for s, v in by_sex.items()
        }).T
        if len(age_sex):
            _savetab(age_sex, out_dir / "tables", "table_age_summary_by_sex.csv")


def compute_anova_scores(feature_df: pd.DataFrame, top_k: int = 25) -> Optional[pd.Series]:
    if "label" not in feature_df.columns:
        print("[EDA-ANOVA] 'label' column missing — skipping ANOVA.")
        return None

    X_num = feature_df.drop(columns=["label"], errors="ignore").select_dtypes(include=[np.number]).copy()
    if X_num.shape[1] == 0:
        print("[EDA-ANOVA] no numeric features — skipping.")
        return None

    # Clean inf to NaN
    X_num = X_num.replace([np.inf, -np.inf], np.nan)

    # Remove near-constant
    try:
        vt = VarianceThreshold(threshold=1e-12)
        X_vt = vt.fit_transform(X_num)
        cols_vt = X_num.columns[vt.get_support()]
    except Exception:
        cols_vt = X_num.columns
        X_vt = X_num.values

    if X_vt.shape[1] == 0:
        print("[EDA-ANOVA] all features constant — skipping.")
        return None

    # Impute
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_vt)

    # y encode
    y_all = feature_df["label"].astype(str).values
    le = LabelEncoder()
    y_enc = le.fit_transform(y_all)

    # ANOVA
    k = int(min(max(1, top_k), X_imp.shape[1]))
    try:
        selector = SelectKBest(score_func=f_classif, k=k).fit(X_imp, y_enc)
    except Exception as e:
        print(f"[EDA-ANOVA] f_classif failed: {e}")
        return None

    scores = pd.Series(selector.scores_, index=cols_vt).sort_values(ascending=False)
    return scores


def plot_anova_top(scores: pd.Series, out_dir: Path, top_k: int = 25):
    if scores is None or not len(scores):
        return
    _set_style()
    s = scores.sort_values(ascending=False).head(top_k)
    _savetab(s.rename("F_score"), out_dir / "tables", "table_anova_top.csv")

    xs = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(xs, s.values, edgecolor="white", linewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(s.index, rotation=65, ha="right")
    ax.set_ylabel("F-score"); ax.set_title(f"ANOVA F-scores (top {top_k})")
    for x, y in zip(xs, s.values):
        ax.text(x, y + max(1, 0.01*s.max()), f"{y:.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    _savefig(fig, out_dir / "figs", "eda_anova_top_bar.png")


def handle_pretty_eda(features_csv: Optional[Path], viz_dir: Path):
    """
    Loads engineered features (CSV) and produces prettier EDA with tables.
    If 'features_csv' is None or not found, tries a few common paths and skips silently if missing.
    """
    # Try a few sensible defaults if not provided
    candidates = []
    if features_csv:
        candidates.append(features_csv)
    candidates += [
        Path("test/artifacts/basic_signal_features.csv"),
        Path("artifacts/basic_signal_features.csv"),
        Path("results/basic_signal_features.csv"),
    ]

    df = None
    for p in candidates:
        if p.exists():
            try:
                df = pd.read_csv(p)
                print(f"[EDA] Loaded features: {p} {df.shape}")
                break
            except Exception:
                continue

    if df is None:
        print("[EDA] features CSV not found — skipping pretty EDA.")
        return

    # Ensure key columns are strings
    if "label" in df.columns:
        df["label"] = df["label"].astype(str)
    if "sex" in df.columns and "sex_norm" not in df.columns:
        df["sex_norm"] = _sex_norm(df["sex"])

    eda_root = viz_dir / "eda"
    # Global class counts table
    if "label" in df.columns:
        counts = df["label"].value_counts().sort_index()
        _savetab(pd.DataFrame({"count": counts, "percent": (counts/counts.sum()*100).round(2)}),
                 eda_root / "tables", "table_class_counts.csv")

    # Plots + tables
    plot_class_by_fold_E(df, eda_root)
    plot_class_by_sex_E(df, eda_root)
    plot_age_E(df, eda_root)

    # HR/HRV boxplots if present
    for col in ["HR_bpm","SDNN_ms","RMSSD_ms"]:
        if col in df.columns and "label" in df.columns:
            _set_style()
            labs = df["label"].astype(str)
            order = [c for c in ["CD","HYP","MI","NORM","STTC"] if c in set(labs)] or sorted(labs.unique())
            groups, labels = [], []
            for lab in order:
                v = pd.to_numeric(df.loc[labs==lab, col], errors="coerce").dropna().values
                if v.size >= 5: groups.append(v); labels.append(lab)
            if groups:
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.boxplot(groups, labels=labels, showfliers=False)
                ax.set_title(f"{col} by diagnostic class"); ax.set_ylabel(col)
                ymin, ymax = ax.get_ylim(); dy = 0.03*(ymax-ymin)
                for i, arr in enumerate(groups, start=1):
                    med = float(np.median(arr)); q3 = float(np.percentile(arr, 75))
                    ax.text(i, q3 + dy, f"n={len(arr):,}\nmed={med:.1f}",
                            ha="center", va="bottom", fontsize=9)
                plt.tight_layout()
                _savefig(fig, eda_root / "figs", f"box_{col}_by_class_pretty.png")

    # ANOVA top-k (SelectKBest) — plus numbers table
    scores = compute_anova_scores(df, top_k=25)
    plot_anova_top(scores, eda_root)


# ==========================
#              Main
# ==========================

def main():
    ap = argparse.ArgumentParser(description="Training results + Pretty EDA visualizations")
    ap.add_argument("--results-dir", type=str, default="results", help="Directory containing *_run.csv etc.")
    ap.add_argument("--viz-dir",     type=str, default="results/viz", help="Output directory for figures.")
    ap.add_argument("--features-csv", type=str, default=None, help="Engineered features CSV for EDA.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    viz_dir     = _ensure_dir(Path(args.viz_dir))
    feat_csv    = Path(args.features_csv) if args.features_csv else None

    # 1) Training results (robust to partial availability)
    handle_training_results(results_dir, viz_dir)

    # 2) Pretty EDA with tables (+ ANOVA on engineered features)
    handle_pretty_eda(feat_csv, viz_dir)

    print(f"Saved plots and ANOVA tables (where possible) to: {viz_dir}")

if __name__ == "__main__":
    main()