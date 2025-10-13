# results_visualization.py
"""
Visualization utilities for FL runs and centralized evaluation.

What this module does
---------------------
• Loads CSV logs produced by federated and centralized runs.
• Produces publication-ready plots (matplotlib/Agg headless backend).
• Runs optional ANOVAs to compare:
  - Global accuracy between runs
  - Per-class accuracy between runs
  - Client accuracy by phase (within run)

Where this plugs into the project
---------------------------------
• src_Connection.Centralized:
    - Uses TorchAdapter, evaluate_models, plot_confusion, plot_learning_curves.
• src_Connection.Experiments:
    - Uses centralized pipeline → the CSVs loaded here.
• src_Connection.Client / Server:
    - Their CSV logging is consumed here for time/accuracy/phase plots.

Outputs
-------
• Figures saved to RESULTS_DIR / "viz"
• ANOVA tables (CSV) saved to the same viz directory

Notes
-----
• All plotting uses the "Agg" backend for headless environments.
• The optional stats dependencies (scipy, statsmodels) are guarded at call sites.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless-safe backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src_Connection.config import RESULTS_DIR, SUPERCLASSES
from src_Connection.utils import ensure_dir

# Optional stats deps (guarded below)
try:
    import scipy.stats as ss
except Exception:  # pragma: no cover
    ss = None  # we'll guard at call sites

try:
    import statsmodels.api as sm                     # type: ignore
    from statsmodels.formula.api import ols         # type: ignore
    _HAS_SM = True
except Exception:  # pragma: no cover
    _HAS_SM = False

# -------------------------------------------------------------------------
# I/O - loading results
# -------------------------------------------------------------------------
FILES = {
    "non_frozen": RESULTS_DIR / "non_frozen_run.csv",
    "frozen":     RESULTS_DIR / "frozen_run.csv",
}
PERCLASS = {
    "non_frozen": RESULTS_DIR / "non_frozen_run_perclass.csv",
    "frozen":     RESULTS_DIR / "frozen_run_perclass.csv",
}
CONFUSION = {
    "non_frozen": RESULTS_DIR / "non_frozen_run_cm.csv",
    "frozen":     RESULTS_DIR / "frozen_run_cm.csv",
}
OUT = RESULTS_DIR / "viz"
ensure_dir(OUT)

PHASE_ENABLED  = "post_cv"
PHASE_DISABLED = "no_cv"

# --- small helper: moving average (rounds) -----------------------------------
MA_WINDOW = 3  # change if you want a smoother/rougher overlay

def _ma_xy(x, y, k: int = MA_WINDOW):
    """Return (x_ma, y_ma) for a simple moving average over rounds.

    Parameters
    ----------
    x, y : array-like
        Original x/y arrays (e.g., rounds, accuracies).
    k : int
        Window length for the moving average.

    Returns
    -------
    (x_ma, y_ma) : np.ndarray, np.ndarray
        x truncated to match the valid convolution window, y averaged.
    """
    try:
        x = np.asarray(list(x), dtype=float)
        y = np.asarray(list(y), dtype=float)
    except Exception:
        return x, y
    if k is None or k <= 1 or y.size < k:
        return x, y
    y_ma = np.convolve(y, np.ones(k, dtype=float) / k, mode="valid")
    x_ma = x[k - 1 :]
    return x_ma, y_ma


def _print_missing(path: Path):
    """Tiny helper to consistently report missing CSVs."""
    print(f"Missing: {path}")


# --- Load CSVs with numeric columns coerced to numbers (NaN if invalid) ---
def load_numeric_csv(path: Path):
    """Load a CSV, coerce common numeric columns, and normalize client_id type.

    Used by:
        • This module's top-level loaders for run/per-class/confusion CSVs.

    Returns
    -------
    pd.DataFrame | None
        A DataFrame with numeric columns parsed, or None on failure/missing.
    """
    if not path.exists():
        _print_missing(path)
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return None
    if df is None or df.empty:
        return df

    # Normalize common columns if present
    for col in ["round", "accuracy", "loss", "wall_time_sec", "trainable_params", "true", "pred", "count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Normalize client_id to string (GLOBAL vs numeric client ids)
    if "client_id" in df.columns:
        df["client_id"] = df["client_id"].astype(str)

    return df


# --- Load all dataframes (calling load_numeric_csv) ---------------------
dfs       = {name: load_numeric_csv(p) for name, p in FILES.items()}
percls    = {name: load_numeric_csv(p) for name, p in PERCLASS.items()}
conf_long = {name: load_numeric_csv(p) for name, p in CONFUSION.items()}

# Keep only non-empty frames
dfs       = {k: v for k, v in dfs.items() if v is not None and not v.empty}
percls    = {k: v for k, v in percls.items() if v is not None and not v.empty}
conf_long = {k: v for k, v in conf_long.items() if v is not None and not v.empty}


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def perclass_columns(df: pd.DataFrame):
    """Extract mapping of label → per-class accuracy Series from a long CSV.

    Accepts both:
        • acc_<label>  (named by SUPERCLASSES)
        • acc_<index>  (integer index mapping to SUPERCLASSES)

    Returns
    -------
    dict[str, pd.Series]
        Keys are class labels compatible with SUPERCLASSES.
    """
    cols = [c for c in df.columns if str(c).startswith("acc_")]
    mapping = {}
    for c in cols:
        key = str(c)[4:]
        if key.isdigit():
            idx = int(key)
            if 0 <= idx < len(SUPERCLASSES):
                mapping[SUPERCLASSES[idx]] = df[c]
        else:
            mapping[key] = df[c]
    return mapping


def _safe_save(fig, name: str):
    """Save a figure into OUT/name with tight layout; always close."""
    out = OUT / name
    try:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print("Saved:", out)
    finally:
        plt.close(fig)

def _safe_save_table(df: pd.DataFrame, name: str):
    """Save a DataFrame as CSV into OUT/name."""
    out = OUT / name
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Saved:", out)


# --------------------------- ANOVA SECTION -----------------------------------
def _effect_sizes_from_groups(groups: list[np.ndarray]) -> tuple[float, float]:
    """Compute eta-squared and omega-squared effect sizes for one-way ANOVA.

    Parameters
    ----------
    groups : list[np.ndarray]
        Each array contains the values for one group.

    Returns
    -------
    (eta^2, omega^2) : tuple[float, float]
        Standard effect size estimates for ANOVA.
    """
    ns   = [len(g) for g in groups if len(g)]
    means= [np.mean(g) for g in groups if len(g)]
    if not ns:
        return (np.nan, np.nan)
    N    = sum(ns); k = len(ns)
    grand= np.average(means, weights=ns)
    ss_between = float(sum(n * (m - grand)**2 for n, m in zip(ns, means)))
    ss_within  = float(sum(((g - np.mean(g))**2).sum() for g in groups if len(g)))
    ss_total   = ss_between + ss_within
    eta2   = (ss_between / ss_total) if ss_total > 0 else np.nan
    dfb = k - 1
    dfw = N - k
    msw = (ss_within / dfw) if dfw > 0 else np.nan
    omega2 = ((ss_between - dfb * msw) / (ss_total + msw)) if (ss_total + msw) > 0 else np.nan
    return float(eta2), float(omega2)

def anova_global_between_runs(dfs_dict: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """One-way ANOVA comparing GLOBAL accuracy distributions across runs.

    Inputs
    ------
    dfs_dict : dict[str, DataFrame]
        Mapping of run name → run CSV DataFrame.

    Output
    ------
    • Saves group stats and ANOVA summary CSVs.
    • Returns the ANOVA summary DataFrame (or None if not enough data).

    Connected to
    ------------
    • Non-frozen vs Frozen FL runs (see config.EXPERIMENT 'run_name').
    """
    if not ss or not dfs_dict:
        print("ANOVA skipped: SciPy not available or no data.")
        return None
    groups = []
    labels = []
    desc = []
    for name, df in dfs_dict.items():
        if df is None or df.empty or "client_id" not in df.columns:
            continue
        g = df[df["client_id"].astype(str) == "GLOBAL"]
        if g.empty or "accuracy" not in g.columns:
            continue
        vals = g["accuracy"].dropna().values.astype(float)
        if vals.size < 2:
            continue
        groups.append(vals); labels.append(name)
        desc.append({"run": name, "n": int(vals.size), "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))})
    if len(groups) < 2:
        print("ANOVA (global between runs): need ≥2 runs with data.")
        return None
    F, p = ss.f_oneway(*groups)
    eta2, omega2 = _effect_sizes_from_groups(groups)
    summary = pd.DataFrame([{"test": "oneway_ANOVA", "F": float(F), "p": float(p),
                             "eta_sq": eta2, "omega_sq": omega2, "k_groups": len(groups),
                             "labels": ",".join(labels)}])
    _safe_save_table(pd.DataFrame(desc), "anova_global_between_runs_groups.csv")
    _safe_save_table(summary, "anova_global_between_runs_summary.csv")

    # If just 2 groups, also report Welch t-test
    if len(groups) == 2:
        t, pw = ss.ttest_ind(groups[0], groups[1], equal_var=False)
        pair = pd.DataFrame([{"comparison": f"{labels[0]} vs {labels[1]}", "t_welch": float(t), "p": float(pw)}])
        _safe_save_table(pair, "anova_global_between_runs_pairwise_welch.csv")
    print("ANOVA (global between runs):", summary.to_dict("records")[0])
    return summary

def anova_perclass_between_runs(perclass_dict: dict[str, pd.DataFrame]) -> pd.DataFrame | None:
    """One-way ANOVA for each superclass comparing per-class accuracy across runs.

    Inputs
    ------
    perclass_dict : dict[str, DataFrame]
        Mapping run → per-class CSV (with columns acc_<label> or acc_<index>).

    Output
    ------
    • CSV saved to viz/ 'anova_perclass_between_runs.csv' with F, p, effect sizes.
    • Returns the DataFrame or None if insufficient data.
    """
    if not ss or not perclass_dict:
        print("Per-class ANOVA skipped: SciPy not available or no data.")
        return None
    rows = []
    for cls in SUPERCLASSES:
        groups = []
        labs   = []
        for name, df in perclass_dict.items():
            if df is None or df.empty:
                continue
            df = df.sort_values("round")
            mapping = perclass_columns(df)
            if cls not in mapping:
                continue
            vals = pd.to_numeric(mapping[cls], errors="coerce").dropna().values.astype(float)
            if vals.size >= 2:
                groups.append(vals); labs.append(name)
        if len(groups) >= 2:
            F, p = ss.f_oneway(*groups)
            eta2, omega2 = _effect_sizes_from_groups(groups)
            rows.append({"class": cls, "k_runs": len(groups), "F": float(F), "p": float(p),
                         "eta_sq": eta2, "omega_sq": omega2, "labels": ",".join(labs)})
    if not rows:
        print("Per-class ANOVA: insufficient data.")
        return None
    df_out = pd.DataFrame(rows).sort_values("p")
    _safe_save_table(df_out, "anova_perclass_between_runs.csv")
    print("Per-class ANOVA — saved CSV.")
    return df_out

def anova_client_accuracy_by_phase(dfs_dict: dict[str, pd.DataFrame]) -> None:
    """Within each run: compare client accuracy between phases (no_cv vs post_cv).

    Behavior
    --------
    • Computes pooled one-way ANOVA across clients.
    • Computes per-client one-way ANOVAs.
    • Optionally runs a two-way ANOVA (Client × Phase) when statsmodels is present.

    Outputs
    -------
    • Saves multiple CSVs under viz/ with per-run results.
    """
    if not ss:
        print("Client-by-phase ANOVA skipped: SciPy not available.")
        return
    for run, df in dfs_dict.items():
        if df is None or df.empty or "phase" not in df.columns or "client_id" not in df.columns:
            continue
        sub = df[df["client_id"].astype(str) != "GLOBAL"].copy()
        if sub.empty:
            continue

        # Pooled across clients
        g1 = pd.to_numeric(sub.loc[sub["phase"].astype(str).eq(PHASE_DISABLED), "accuracy"], errors="coerce").dropna().values
        g2 = pd.to_numeric(sub.loc[sub["phase"].astype(str).eq(PHASE_ENABLED),  "accuracy"], errors="coerce").dropna().values
        if g1.size >= 2 and g2.size >= 2:
            F, p = ss.f_oneway(g1, g2)
            eta2, omega2 = _effect_sizes_from_groups([g1, g2])
            overall = pd.DataFrame([{"run": run, "groups": "pooled_clients", "F": float(F), "p": float(p),
                                     "eta_sq": eta2, "omega_sq": omega2, "n_no_cv": int(g1.size), "n_post_cv": int(g2.size)}])
            _safe_save_table(overall, f"anova_client_phase_{run}_overall.csv")

        # Per-client one-way (two groups) + optional two-way with statsmodels
        rows = []
        for cid, grp in sub.groupby("client_id"):
            a = pd.to_numeric(grp.loc[grp["phase"].astype(str).eq(PHASE_DISABLED), "accuracy"], errors="coerce").dropna().values
            b = pd.to_numeric(grp.loc[grp["phase"].astype[str].eq(PHASE_ENABLED),  "accuracy"], errors="coerce").dropna().values
            if a.size >= 2 and b.size >= 2:
                F, p = ss.f_oneway(a, b)
                eta2, omega2 = _effect_sizes_from_groups([a, b])
                rows.append({"run": run, "client_id": str(cid), "F": float(F), "p": float(p),
                             "eta_sq": eta2, "omega_sq": omega2, "n_no_cv": int(a.size), "n_post_cv": int(b.size),
                             "mean_no_cv": float(np.mean(a)), "mean_post_cv": float(np.mean(b))})
        if rows:
            dfc = pd.DataFrame(rows).sort_values("p")
            _safe_save_table(dfc, f"anova_client_phase_{run}_perclient.csv")

        # Optional two-way (Client × Phase)
        if _HAS_SM:
            try:
                tmp = sub.dropna(subset=["accuracy", "phase", "client_id"]).copy()
                tmp["phase"] = tmp["phase"].astype(str)
                tmp["client_id"] = tmp["client_id"].astype(str)
                model = ols("accuracy ~ C(client_id) + C(phase) + C(client_id):C(phase)", data=tmp).fit()
                anova_tbl = sm.stats.anova_lm(model, typ=2)
                out = anova_tbl.reset_index().rename(columns={"index": "term"})
                _safe_save_table(out, f"anova_two_way_client_phase_{run}.csv")
            except Exception as e:
                print(f"[ANOVA two-way] {run}: {e}")


# -------------------------------------------------------------------------
# Plotting + ANOVA runner (no side effects on import)
# -------------------------------------------------------------------------
def _run_plots_and_anova():
    """Generate standard plots from CSVs and run ANOVAs if possible.

    Produces
    --------
    • Global accuracy per round
    • Client vs Global accuracy per run
    • Per-round total client wall-time
    • Per-client trajectories (frozen vs non-frozen)
    • Per-class accuracy trajectories and final-round bars
    • Confusion matrices for last round of each run
    • Global and per-client accuracy by phase
    • ANOVA CSVs (global between runs, per-class between runs, client-by-phase)
    """
    # -- Global accuracy vs round --------------------------------------------
    if dfs:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for name, df in dfs.items():
            if df is None or df.empty or "client_id" not in df.columns:
                continue
            g = df.loc[df["client_id"].astype(str) == "GLOBAL"].copy()
            if g.empty or "round" not in g.columns or "accuracy" not in g.columns:
                continue
            g = g.sort_values("round")
            ax.plot(g["round"], g["accuracy"], label=f"Global ({name})")
        ax.set_xlabel("Round"); ax.set_ylabel("Accuracy"); ax.set_title("Global Accuracy per Round")
        ax.legend(); ax.grid(True); fig.tight_layout()
        _safe_save(fig, "global_accuracy.png")

    # -- Client accuracies vs round ------------------------------------------
    for name, df in dfs.items():
        if df is None or df.empty or "client_id" not in df.columns:
            continue
        sub = df[df["client_id"].astype(str) != "GLOBAL"].copy()
        if sub.empty or "round" not in sub.columns or "accuracy" not in sub.columns:
            continue
        sub = sub.sort_values(["client_id", "round"])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for cid, grp in sub.groupby("client_id"):
            if grp.empty:
                continue
            ax.plot(grp["round"], grp["accuracy"], alpha=0.6, label=f"Client {cid}")
        g = df[df["client_id"].astype(str) == "GLOBAL"].copy()
        if not g.empty and "round" in g.columns and "accuracy" in g.columns:
            g = g.sort_values("round")
            ax.plot(g["round"], g["accuracy"], linewidth=3, linestyle="--", label="GLOBAL")
        ax.set_xlabel("Round"); ax.set_ylabel("Accuracy")
        ax.set_title(f"Client vs Global Accuracy — {name}")
        ax.legend(ncol=2); ax.grid(True); fig.tight_layout()
        _safe_save(fig, f"clients_vs_global_{name}.png")

    # -- Total wall-time per round (sum across clients) ----------------------
    for name, df in dfs.items():
        if df is None or df.empty or "wall_time_sec" not in df.columns or "round" not in df.columns:
            continue
        sub = df[df["client_id"].astype(str) != "GLOBAL"].copy()
        if sub.empty:
            continue
        agg = (sub.groupby("round", as_index=False)
                  .agg(wall_time_sec=("wall_time_sec", "sum"))
                  .sort_values(by="round"))
        if agg.empty:
            continue
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(agg["round"], agg["wall_time_sec"])
        ax.set_xlabel("Round"); ax.set_ylabel("Total client wall-time (s)")
        ax.set_title(f"Total Wall-Time per Round — {name}")
        ax.grid(True); fig.tight_layout()
        _safe_save(fig, f"walltime_{name}.png")

    # -- Per-client trajectories (frozen vs non-frozen, one figure) ----------
    if all(k in dfs for k in ("frozen", "non_frozen")):
        nf = dfs.get("non_frozen"); fr = dfs.get("frozen")
        if nf is not None and fr is not None and not nf.empty and not fr.empty:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            cids = set(nf.query("client_id!='GLOBAL'")["client_id"]).union(
                   set(fr.query("client_id!='GLOBAL'")["client_id"]))
            for cid in sorted(cids):
                for run_name, dfi, style in [("non_frozen", nf, "-"), ("frozen", fr, "--")]:
                    sub_df = dfi.loc[dfi["client_id"] == cid].copy()
                    if sub_df.empty or "round" not in sub_df.columns or "accuracy" not in sub_df.columns:
                        continue
                    sub_df.sort_values(by="round", inplace=True)
                    lbl = f"Client {cid} ({'unfrozen' if run_name=='non_frozen' else 'frozen'})"
                    ax.plot(sub_df["round"], sub_df["accuracy"], linestyle=style, label=lbl, alpha=0.8)
            ax.set_xlabel("Round"); ax.set_ylabel("Local Accuracy")
            ax.set_title("Per-Client Accuracy Trajectories (Frozen vs Unfrozen)")
            ax.legend(ncol=2); ax.grid(True); fig.tight_layout()
            _safe_save(fig, "per_client_trajectories.png")

    # -- Per-class accuracy over rounds (line plot) --------------------------
    if percls:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if "non_frozen" in percls and not percls["non_frozen"].empty:
            nf = percls["non_frozen"].sort_values(by="round")
            for label, series in perclass_columns(nf).items():
                try:
                    ax.plot(nf["round"], series, label=f"{label} (unfrozen)")
                except Exception:
                    pass
        if "frozen" in percls and not percls["frozen"].empty:
            fr = percls["frozen"].sort_values(by="round")
            for label, series in perclass_columns(fr).items():
                try:
                    ax.plot(fr["round"], series, linestyle="--", label=f"{label} (frozen)")
                except Exception:
                    pass
        ax.set_xlabel("Round"); ax.set_ylabel("Per-class Accuracy")
        ax.set_title("Per-class Accuracy per Round")
        ax.legend(ncol=2); ax.grid(True); fig.tight_layout()
        _safe_save(fig, "perclass_over_rounds.png")

    # -- Final-round per-class accuracy (bar) --------------------------------
    if percls:
        labels = SUPERCLASSES
        x = np.arange(len(labels)); width = 0.35
        fig = plt.figure()
        ax = fig.add_subplot(111)
        drew_any = False
        if "non_frozen" in percls and not percls["non_frozen"].empty:
            nf_last = percls["non_frozen"].sort_values(by="round").tail(1)
            nf_map  = perclass_columns(nf_last)
            y_nf    = [float(nf_map.get(lbl, pd.Series([0])).iloc[0]) for lbl in labels]
            ax.bar(x - width/2, y_nf, width, label="Unfrozen"); drew_any = True
        if "frozen" in percls and not percls["frozen"].empty:
            fr_last = percls["frozen"].sort_values(by="round").tail(1)
            fr_map  = perclass_columns(fr_last)
            y_fr    = [float(fr_map.get(lbl, pd.Series([0])).iloc[0]) for lbl in labels]
            ax.bar(x + width/2, y_fr, width, label="Frozen"); drew_any = True
        if drew_any:
            ax.set_xticks(x); ax.set_xticklabels(labels)
            ax.set_ylabel("Accuracy"); ax.set_title("Final-Round Per-class Accuracy")
            ax.legend(); fig.tight_layout()
            _safe_save(fig, "perclass_final_bar.png")
        else:
            plt.close(fig)

    # -- Confusion heatmaps (last round of each run) -------------------------
    for name, dfc in conf_long.items():
        if dfc is None or dfc.empty:
            continue
        if "round" not in dfc.columns or "true" not in dfc.columns or "pred" not in dfc.columns or "count" not in dfc.columns:
            continue
        r = int(dfc["round"].max())
        sub = dfc[dfc["round"] == r].copy()
        if sub.empty:
            continue
        n = len(SUPERCLASSES)
        try:
            piv = sub.pivot_table(index="true", columns="pred", values="count", aggfunc="sum", fill_value=0)
            piv.index = pd.to_numeric(piv.index, errors="coerce")
            piv.columns = pd.to_numeric(piv.columns, errors="coerce")
            cm = (piv.reindex(index=range(n), columns=range(n), fill_value=0).to_numpy(dtype=float))
        except Exception as e:
            print(f"Confusion pivot failed for {name}: {e}")
            continue

        with np.errstate(invalid="ignore", divide="ignore"):
            row_sum = cm.sum(axis=1, keepdims=True)
            cmn = np.nan_to_num(cm / np.maximum(row_sum, 1), nan=0.0, posinf=0.0, neginf=0.0)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        im = ax.imshow(cmn, aspect="auto", vmin=0, vmax=1.0)
        ax.set_title(f"Confusion Matrix — {name} (round {r})")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_xticks(range(n)); ax.set_xticklabels(SUPERCLASSES, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(SUPERCLASSES)
        cbar = fig.colorbar(im, ax=ax); cbar.set_label("Row-normalized")
        for i in range(n):
            for j in range(n):
                val = cmn[i, j] * 100.0
                ax.text(j, i, f"{val:0.2f}", ha="center", va="center",
                        color=("white" if val >= 50 else "black"), fontsize=9)
        fig.tight_layout()
        _safe_save(fig, f"confusion_{name}_r{r:02d}.png")

    # -- Global accuracy by phase (within each run) --------------------------
    for name, df in dfs.items():
        if df is None or df.empty or "phase" not in df.columns:
            continue
        g = df[df["client_id"].astype(str) == "GLOBAL"].copy()
        if g.empty or "round" not in g.columns or "accuracy" not in g.columns:
            continue
        pre  = g[g["phase"].astype(str).eq(PHASE_DISABLED)].sort_values(by="round")
        post = g[g["phase"].astype(str).eq(PHASE_ENABLED)].sort_values(by="round")
        if pre.empty and post.empty:
            continue
        fig = plt.figure()
        ax = fig.add_subplot(111)
        if not pre.empty:  ax.plot(pre["round"], pre["accuracy"], label=f"{name} (no-CV)")
        if not post.empty: ax.plot(post["round"], post["accuracy"], label=f"{name} (post-CV)")
        ax.set_xlabel("Round"); ax.set_ylabel("Global Accuracy")
        ax.set_title(f"Global Accuracy by Phase — {name}")
        ax.legend(); ax.grid(True); fig.tight_layout()
        _safe_save(fig, f"global_by_phase_{name}.png")

    # -- Per-client accuracy by phase (overlay pre/post + MA overlays) -------
    for name, df in dfs.items():
        if df is None or df.empty or "phase" not in df.columns or "client_id" not in df.columns:
            continue
        sub = df[df["client_id"].astype(str) != "GLOBAL"].copy()
        if sub.empty or "round" not in sub.columns or "accuracy" not in sub.columns:
            continue
        fig = plt.figure()
        ax = fig.add_subplot(111)
        drew_any = False

        for phase, style in [(PHASE_DISABLED, "-"), (PHASE_ENABLED, "--")]:
            grp = sub[sub["phase"].astype(str).eq(phase)]
            for cid, g in grp.groupby("client_id"):
                g = g.sort_values(by="round")
                if g.empty:
                    continue
                line, = ax.plot(g["round"], g["accuracy"], linestyle=style, alpha=0.35, zorder=1)
                xm, ym = _ma_xy(g["round"].values, g["accuracy"].values)
                ax.plot(
                    xm, ym,
                    linestyle=style, linewidth=2.0, color=line.get_color(), alpha=0.95,
                    label=f"Client {cid} ({name}, {phase})", zorder=2
                )
                drew_any = True

        g_global = df[df["client_id"].astype(str) == "GLOBAL"].copy()
        if not g_global.empty and "round" in g_global.columns and "accuracy" in g_global.columns:
            g_global = g_global.sort_values("round")
            xg, yg = _ma_xy(g_global["round"].values, g_global["accuracy"].values)
            if len(yg):
                ax.plot(
                    xg, yg,
                    linestyle="-.", linewidth=3.0, color="black", alpha=0.8,
                    label=f"GLOBAL ({name}, MA{MA_WINDOW})", zorder=3
                )

        if drew_any:
            ax.set_xlabel("Round"); ax.set_ylabel("Local Accuracy")
            ax.set_title(f"Per-Client Accuracy by Phase — {name}")
            ax.legend(ncol=2); ax.grid(True); fig.tight_layout()
            _safe_save(fig, f"clients_by_phase_{name}.png")
        else:
            plt.close(fig)

    # Run ANOVAs when data present
    anova_global_between_runs(dfs)
    anova_perclass_between_runs(percls)
    anova_client_accuracy_by_phase(dfs)

    print(f"Saved plots and ANOVA tables (where possible) to: {OUT.resolve()}")


# =============================================================================
# The remainder provides unified evaluation utilities used by Centralized.py.
# =============================================================================

import time as _time, re as _re
import torch as _torch
from sklearn.metrics import (classification_report as _clsrep, accuracy_score as _acc,
                             f1_score as _f1, precision_score as _prec, recall_score as _rec,
                             roc_auc_score as _rocauc, confusion_matrix as _sk_cm)

# Avoid circular imports at module load time
try:
    from . import config as CFG
    from .utils import torch_loader_kwargs as _torch_loader_kwargs
except Exception:
    from src_Connection import config as CFG  # type: ignore
    from src_Connection.utils import torch_loader_kwargs as _torch_loader_kwargs  # type: ignore


class TorchAdapter:
    """Unified inference wrapper for PyTorch models used by evaluation/visualization.

    Responsibilities
    ----------------
    • Builds a predict/predict_proba API over waveform paths (WFDB base paths).
    • Handles device placement and batch loading consistently with config.
    • Converts model logits into predicted labels using the provided LabelEncoder.

    Used by
    -------
    • Centralized.run_deep_models → to get y_pred / proba for metrics and plots.
    """

    def __init__(self, model, label_encoder, is_binary: bool, batch=32, device=None):
        self.model = model.to(device or _torch.device("cpu")).eval()
        self.le = label_encoder; self.is_binary = is_binary
        self.batch = int(batch); self.device = device or _torch.device("cpu")
        self.classes_ = getattr(self.le, "classes_", None)

    @_torch.no_grad()
    def _loader(self, paths):
        """Internal: build a predict-only DataLoader from WFDB base paths."""
        from torch.utils.data import Dataset, DataLoader
        from .data_loader import load_waveform_np
        class _PredictOnly(Dataset):
            def __init__(self, p):
                self.paths=list(p)
                self.T = max(1, CFG.SEQ_LEN // max(1, CFG.DOWNSAMPLE_FACTOR))
            def __len__(self): return len(self.paths)
            def __getitem__(self, i):
                return _torch.from_numpy(load_waveform_np(self.paths[i], T=self.T, factor=CFG.DOWNSAMPLE_FACTOR))
        return DataLoader(_PredictOnly(paths), **_torch_loader_kwargs(False, self.batch, self.device.type))

    @_torch.no_grad()
    def predict(self, paths):
        """Return predicted class labels (str) for a list/array of WFDB base paths."""
        preds=[]
        for xb in self._loader(paths):
            xb=xb.to(self.device); logits=self.model(xb)
            y_idx=((_torch.sigmoid(logits.view(-1))>=0.5).long().cpu().numpy()
                   if self.is_binary else logits.argmax(1).cpu().numpy())
            preds.extend(self.le.inverse_transform(y_idx))
        return np.asarray(preds)

    @_torch.no_grad()
    def predict_proba(self, paths):
        """Return class probability estimates aligned to self.classes_ order."""
        out=[]
        for xb in self._loader(paths):
            xb=xb.to(self.device); logits=self.model(xb)
            if self.is_binary:
                p1=_torch.sigmoid(logits.view(-1)).unsqueeze(1); p0=1.0-p1; p=_torch.cat([p0,p1], dim=1)
            else:
                p=_torch.softmax(logits, dim=1)
            out.append(p.cpu().numpy().astype("float32"))
        return np.vstack(out)


def evaluate_models(models_dict, paths, y_true, label_encoder, is_binary, device):
    """Evaluate multiple models/adapters and return a unified metrics table.

    Parameters
    ----------
    models_dict : dict[str, TorchAdapter]
        Mapping of model name → TorchAdapter instance.
    paths : array-like[str]
        WFDB base paths (test set).
    y_true : array-like[str]
        Ground-truth labels in string form (for readability).
    label_encoder : LabelEncoder
        Used to map indices ↔ labels for ROC-AUC multi-class alignment.
    is_binary : bool
        Binary classification flag (affects ROC-AUC handling).
    device : torch.device
        Passed through to adapter if needed (already applied at construction).

    Returns
    -------
    pd.DataFrame
        Sorted by weighted F1, with accuracy/precision/recall/F1 and OVR ROC-AUC (when possible).

    Connected to
    ------------
    • Centralized.run_deep_models: prints classification report and uses the returned DataFrame
      to pick a “best” model for confusion plots and learning curves.
    """
    rows=[]
    for name, adapter in models_dict.items():
        y_pred = adapter.predict(paths)
        try:
            print(f"\n=== {name} ===")
            print(_clsrep(y_true, y_pred, digits=3, zero_division=0))
        except Exception:
            pass
        row = {
            "model": name,
            "accuracy": _acc(y_true, y_pred),
            "precision_macro": _prec(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": _rec(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": _f1(y_true, y_pred, average="macro", zero_division=0),
            "f1_weighted": _f1(y_true, y_pred, average="weighted", zero_division=0),
            "roc_auc_ovr_macro": np.nan
        }
        try:
            classes = np.unique(y_true)
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y_true, classes=classes)
            scores = adapter.predict_proba(paths)
            est = np.asarray(adapter.classes_)
            pos = {c:i for i,c in enumerate(est)}
            idx = [pos[c] for c in classes]
            scores = scores[:, idx]
            row["roc_auc_ovr_macro"] = float(_rocauc(y_bin, scores, average="macro", multi_class="ovr"))
        except Exception:
            pass
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("f1_weighted", ascending=False)
    return df


def _slug(txt): return _re.sub(r"[^a-z0-9]+", "_", str(txt).lower()).strip("_")

def plot_confusion(y_true, y_pred, title="Confusion Matrix", classes=None, save_path=None, collapse_to_3=False):
    """Plot a (row-normalized %) confusion matrix and optionally save to file.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted labels (strings or ints).
    title : str
        Figure title.
    classes : Optional[list[str]]
        Class label order; if None, inferred from union of y_true/y_pred.
    save_path : Optional[str | Path]
        If provided, saves PNG at this path.
    collapse_to_3 : bool
        If True, maps classes to {'MI', 'NORM', 'OTHER'} for a coarse view.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure (also saved if save_path provided).

    Used by
    -------
    • Centralized.run_deep_models: to visualize best model confusion.
    """
    if collapse_to_3:
        def _map3(a):
            a=str(a)
            if a=="MI": return "MI"
            if a=="NORM": return "NORM"
            return "OTHER"
        y_true = np.vectorize(_map3)(np.array(y_true)); y_pred = np.vectorize(_map3)(np.array(y_pred))
        classes = ["MI","NORM","OTHER"]
    else:
        y_true = np.array(y_true); y_pred = np.array(y_pred)
        classes = classes or sorted(list(set(y_true)|set(y_pred)))
    cm = _sk_cm(y_true, y_pred, labels=classes)
    cm_pct = cm.astype(float); row_sum = cm_pct.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm_pct, np.maximum(row_sum, 1), where=row_sum>0) * 100.0
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    im = ax.imshow(cm_pct, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
    fig.colorbar(im, ax=ax).set_label("%")
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            val = cm_pct[i, j]
            ax.text(j, i, f"{val:0.2f}", ha="center", va="center",
                    color=("white" if val >= 50 else "black"), fontsize=10)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_learning_curves(histories: dict[str, dict], out_dir: str | Path | None = None):
    """Plot and save loss/accuracy curves for multiple models.

    Parameters
    ----------
    histories : dict[str, dict]
        For each model name, a dict with keys 'loss', 'val_loss', 'accuracy', 'val_accuracy'.
    out_dir : Optional[str | Path]
        Destination directory; defaults to CFG.ART_DIR/figs when None.

    Outputs
    -------
    • 'loss_curve_<model>.png' and, when accuracy present, 'accuracy_curve_<model>.png'
    """
    if not histories:
        print("No histories to plot.")
        return
    out_dir = Path(out_dir or (Path(CFG.ART_DIR) / "figs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, hist in histories.items():
        ep = range(1, len(hist.get('loss', [])) + 1)
        plt.figure(figsize=(7,4))
        plt.plot(ep, hist.get('loss', []), marker='o', linewidth=1, label='Training Loss')
        plt.plot(ep, hist.get('val_loss', []), marker='o', linewidth=1, label='Validation Loss')
        plt.title(f'Loss — {name}'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(alpha=0.3)
        fp1 = out_dir / f"loss_curve_{_slug(name)}.png"
        plt.tight_layout(); plt.savefig(fp1, dpi=300, bbox_inches='tight'); print('Saved:', fp1); plt.close()

        acc = hist.get('accuracy', []); val_acc = hist.get('val_accuracy', [])
        if len(acc):
            plt.figure(figsize=(7,4))
            plt.plot(ep, acc, marker='o', linewidth=1, label='Training Acc')
            plt.plot(ep, val_acc, marker='o', linewidth=1, label='Validation Acc')
            plt.title(f'Accuracy — {name}'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(alpha=0.3)
            fp2 = out_dir / f"accuracy_curve_{_slug(name)}.png"
            plt.tight_layout(); plt.savefig(fp2, dpi=300, bbox_inches='tight'); print('Saved:', fp2); plt.close()


from sklearn.metrics import f1_score as _f1_score

def fairness_macro_f1_by_groups(best_adapter, y_true, test_index, meta_df: pd.DataFrame):
    """Compute macro-F1 per demographic group (sex, age bins) on the test subset.

    Parameters
    ----------
    best_adapter : TorchAdapter
        Adapter used to make predictions.
    y_true : array-like[str]
        Ground-truth labels for the test subset (aligned by test_index).
    test_index : array-like
        Indices into meta_df defining the test subset.
    meta_df : pd.DataFrame
        Metadata table with at least 'record_path' and optionally 'sex', 'age'.

    Returns
    -------
    pd.DataFrame
        Columns: group, value, n, macro_F1
    """
    paths = meta_df.loc[test_index, "record_path"].astype(str).values
    y_pred = np.asarray(best_adapter.predict(paths))
    y_true = np.asarray(list(y_true))
    meta_test = meta_df.loc[test_index].copy()

    if "sex_norm" not in meta_test.columns and "sex" in meta_test.columns:
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
        meta_test["sex_norm"] = _sex_norm(meta_test["sex"])

    meta_test["age_num"] = pd.to_numeric(meta_test.get("age", np.nan), errors="coerce")
    meta_test["age_bin"] = pd.cut(meta_test["age_num"], bins=[0,40,60,200], labels=["<=40","41-60","61+"])

    rows = []
    def _macro_f1_mask(mask):
        a, b = y_true[mask], y_pred[mask]
        if len(np.unique(a)) < 2:
            return np.nan
        return _f1_score(a, b, average="macro", zero_division=0)

    for grp_name, series in [("sex_norm", meta_test["sex_norm"]), ("age_bin", meta_test["age_bin"])]:
        for g, idxs in series.groupby(series).groups.items():
            mask = meta_test.index.isin(idxs)
            rows.append({"group": grp_name, "value": str(g), "n": int(mask.sum()), "macro_F1": _macro_f1_mask(mask)})
    fair_df = pd.DataFrame(rows).sort_values(["group","value"])
    return fair_df


def saliency_grad_times_input(model: _torch.nn.Module, path: str, device=None):
    """Compute a simple saliency signal (|grad * input|) for one record.

    Steps
    -----
    • Loads a single waveform from path.
    • Forwards through model and backprops the winning logit.
    • Returns the raw (T, C) array and the per-time averaged saliency over leads.

    Returns
    -------
    (x, sal_mean) : np.ndarray, np.ndarray
        x is (T, C) waveform; sal_mean is (T,) saliency to overlay on a lead.
    """
    device = device or _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    from .data_loader import load_waveform_np
    T = max(1, CFG.SEQ_LEN // max(1, CFG.DOWNSAMPLE_FACTOR))
    x = load_waveform_np(path, T=T, factor=CFG.DOWNSAMPLE_FACTOR)
    xt = _torch.from_numpy(x).unsqueeze(0).to(device)
    xt.requires_grad_(True)

    with _torch.enable_grad():
        logits = model(xt)
        if logits.ndim == 2 and logits.size(1) > 1:
            target = logits[0, logits.argmax(1).item()]
        else:
            target = logits.view(-1)[0]
        model.zero_grad(set_to_none=True)
        if xt.grad is not None: xt.grad.zero_()
        target.backward()
        grad = xt.grad.detach(); xt_det = xt.detach()
        sal = (grad * xt_det).abs().squeeze(0).cpu().numpy()
        sal_mean = sal.mean(axis=1)
    return x, sal_mean

def plot_saliency_overlay(model, sample_paths: list[str], out_dir: str | Path | None = None, device=None):
    """Save saliency overlays (Lead I + saliency curve) for a list of records."""
    out_dir = Path(out_dir or (Path(CFG.ART_DIR) / "figs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sample_paths:
        x, s = saliency_grad_times_input(model, p, device=device)
        fig, ax1 = plt.subplots(figsize=(10, 3.0))
        ax1.plot(x[:, 0], linewidth=0.8); ax1.set_title("ECG (Lead I) with saliency overlay"); ax1.set_ylabel("mV"); ax1.grid(alpha=0.3)
        ax2 = ax1.twinx(); ax2.plot(s, linewidth=0.8, alpha=0.85); ax2.set_ylabel("saliency")
        fp = out_dir / f"saliency_{Path(p).name.replace('/', '_')}.png"
        plt.tight_layout(); plt.savefig(fp, dpi=220, bbox_inches="tight"); print("Saved:", fp); plt.close()


def count_params(m):
    """Count trainable parameters of a torch.nn.Module."""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def benchmark_inference(adapter, paths, device=None, warmup=1):
    """Time adapter.predict over provided paths (with a small warmup)."""
    device = device or _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    model = adapter.model.to(device).eval()
    if warmup:
        _ = adapter.predict(paths[:min(8, len(paths))])
    t0 = _time.time()
    _ = adapter.predict(paths)
    return _time.time() - t0

def summarize_efficiency(adapters: dict[str, object], test_paths: list[str]):
    """Return a table of params, total inference time, and throughput per model."""
    rows = []
    for name, adapter in adapters.items():
        try:
            secs = benchmark_inference(adapter, test_paths)
            params = count_params(adapter.model)
            rows.append({"model": name, "params": int(params), "test_infer_sec": float(secs),
                         "samples": int(len(test_paths)),
                         "samples_per_sec": float(len(test_paths))/max(secs, 1e-6)})
        except Exception:
            rows.append({"model": name, "params": np.nan, "test_infer_sec": np.nan,
                         "samples": len(test_paths), "samples_per_sec": np.nan})
    return pd.DataFrame(rows).sort_values("samples_per_sec", ascending=False)


# Run only when executed as a script (no side effects on import)
if __name__ == "__main__":
    _run_plots_and_anova()