import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, SUPERCLASSES
from src.utils import ensure_dir

# -------------------------------------------------------------------------
# I/O - loading results
# -------------------------------------------------------------------------

# --- Auto-discover runs per model (cnn/lstm) -----------------------------
def _discover_runs():
    runs, percls, conf = {}, {}, {}
    for p in (RESULTS_DIR).glob("*_frozen_run.csv"):
        model = p.stem.replace("_frozen_run", "")
        runs.setdefault(model, {})["frozen"] = p
    for p in (RESULTS_DIR).glob("*_non_frozen_run.csv"):
        model = p.stem.replace("_non_frozen_run", "")
        runs.setdefault(model, {})["non_frozen"] = p
    for model, vs in runs.items():
        percls[model] = {}
        conf[model] = {}
        if "frozen" in vs:
            percls[model]["frozen"] = RESULTS_DIR / f"{model}_frozen_run_perclass.csv"
            conf[model]["frozen"]   = RESULTS_DIR / f"{model}_frozen_run_cm.csv"
        if "non_frozen" in vs:
            percls[model]["non_frozen"] = RESULTS_DIR / f"{model}_non_frozen_run_perclass.csv"
            conf[model]["non_frozen"]   = RESULTS_DIR / f"{model}_non_frozen_run_cm.csv"
    return runs, percls, conf

RUNS, PERCLS_MAP, CONF_MAP = _discover_runs()
OUT = RESULTS_DIR / "viz"
ensure_dir(OUT)

PHASE_ENABLED  = "post_cv"
PHASE_DISABLED = "no_cv"
PHASE_CACHED   = "cached_cv"  # if you log a cached phase

# --- Load CSVs with numeric columns coerced to numbers (NaN if invalid) ---
def load_numeric_csv(path: Path):
    if not path.exists():
        print(f"Missing: {path}")
        return None
    df = pd.read_csv(path)
    for col in ["round","accuracy","loss","wall_time_sec","trainable_params","true","pred","count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# --- Load all dataframes per model --------------------------------------
dfs_by_model       = {m: {k: load_numeric_csv(p) for k, p in d.items()} for m, d in RUNS.items()}
percls_by_model    = {m: {k: load_numeric_csv(p) for k, p in d.items()} for m, d in PERCLS_MAP.items()}
conf_long_by_model = {m: {k: load_numeric_csv(p) for k, p in d.items()} for m, d in CONF_MAP.items()}

# Drop missing
for m in list(dfs_by_model.keys()):
    dfs_by_model[m]       = {k:v for k,v in dfs_by_model[m].items() if v is not None}
    percls_by_model[m]    = {k:v for k,v in percls_by_model.get(m, {}).items() if v is not None}
    conf_long_by_model[m] = {k:v for k,v in conf_long_by_model.get(m, {}).items() if v is not None}

# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def perclass_columns(df):
    """Return mapping label->Series, accepting acc_<name> or acc_<index>."""
    cols = [c for c in df.columns if c.startswith("acc_")]
    mapping = {}
    for c in cols:
        key = c[4:]
        if key.isdigit():
            idx = int(key)
            if 0 <= idx < len(SUPERCLASSES):
                mapping[SUPERCLASSES[idx]] = df[c]
        else:
            mapping[key] = df[c]
    return mapping

# --- Smoothing function (rolling mean) -----------------------------------
SMOOTH = int(os.getenv("SMOOTH", "1"))  # 1 = off; 3/5 for presentation
def smooth(y):
    if SMOOTH <= 1:
        return y
    s = pd.Series(y, dtype=float)
    return s.rolling(SMOOTH, min_periods=1).mean().to_numpy()

# --- Phase de-dup helpers ------------------------------------------------
PHASE_PRIORITY = {PHASE_DISABLED: 0, PHASE_CACHED: 1, PHASE_ENABLED: 2}

def _collapse_rows(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """
    Keep a single row per 'keys', preferring later tuning phases.
    Falls back to 'keep=last' if 'phase' column does not exist.
    """
    if df is None or df.empty:
        return df
    if "phase" in df.columns:
        return (df.assign(_p=df["phase"].map(PHASE_PRIORITY).fillna(-1))
                  .sort_values([*keys, "_p"])
                  .drop_duplicates(keys, keep="last")
                  .drop(columns="_p"))
    # Per-class / confusion CSVs usually have no 'phase'; keep the last written row.
    return df.sort_values(keys).drop_duplicates(keys, keep="last")

# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

for model in dfs_by_model.keys():
    dfs       = dfs_by_model[model]
    percls    = percls_by_model.get(model, {})
    conf_long = conf_long_by_model.get(model, {})
    SFX = f"_{model}"

    # -- Global accuracy vs round ------------------------------------------
    plt.figure()
    for name, df in dfs.items():
        g = _collapse_rows(df[df["client_id"] == "GLOBAL"].copy(), ["client_id", "round"])
        if g is not None and len(g):
            plt.plot(g["round"], smooth(g["accuracy"]), label=f"Global ({name})")
    plt.xlabel("Round"); plt.ylabel("Accuracy"); plt.title(f"Global Accuracy per Round — {model}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"global_accuracy{SFX}.png", dpi=150); plt.close()

    # -- Client accuracies vs round (with GLOBAL overlay) -------------------
    for name, df in dfs.items():
        plt.figure()
        sub = _collapse_rows(df[df["client_id"] != "GLOBAL"].copy(), ["client_id", "round"])
        if sub is not None and not sub.empty:
            for cid, grp in sub.groupby("client_id"):
                grp = grp.sort_values("round")
                plt.plot(grp["round"], smooth(grp["accuracy"]), alpha=0.6, label=f"Client {cid}")
        g = _collapse_rows(df[df["client_id"] == "GLOBAL"].copy(), ["client_id", "round"])
        if g is not None and len(g):
            plt.plot(g["round"], smooth(g["accuracy"]), linewidth=3, linestyle="--", label="GLOBAL")
        plt.xlabel("Round"); plt.ylabel("Accuracy")
        plt.title(f"Client vs Global Accuracy — {name} — {model}")
        plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
        plt.savefig(OUT / f"clients_vs_global_{name}{SFX}.png", dpi=150); plt.close()


    # -- Mean client accuracy per round (not GLOBAL) -----------------------
    plt.figure()
    for name, df in dfs.items():
        sub = _collapse_rows(df[df["client_id"] != "GLOBAL"].copy(), ["client_id", "round"])
        if sub is None or sub.empty:
            continue
        agg = (sub.groupby("round", as_index=False)
                  .agg(mean_acc=("accuracy","mean"))
                  .sort_values("round"))
        plt.plot(agg["round"], smooth(agg["mean_acc"]), label=f"{name} (mean client)")
    plt.xlabel("Round"); plt.ylabel("Mean Client Accuracy")
    plt.title(f"Mean Client Accuracy per Round - {model}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"global_mean_accuracy{SFX}.png", dpi=150); plt.close()


    # -- Wall-time per round (total across clients) ------------------------
    for name, df in dfs.items():
        if "wall_time_sec" not in df.columns:
            continue
        sub = _collapse_rows(df[df["client_id"] != "GLOBAL"].copy(), ["client_id", "round"])
        if sub is None or sub.empty:
            continue
        agg = (
            sub.groupby("round", as_index=False)
               .agg(wall_time_sec=("wall_time_sec", "sum"))
               .sort_values(by="round")
        )
        plt.figure()
        plt.plot(agg["round"], agg["wall_time_sec"])
        plt.xlabel("Round"); plt.ylabel("Total client wall-time (s)")
        plt.title(f"Total Wall-Time per Round — {name} — {model}")
        plt.grid(True); plt.tight_layout()
        plt.savefig(OUT / f"walltime_{name}{SFX}.png", dpi=150); plt.close()


    # -- Per-client trajectories (frozen vs non-frozen, one figure) ---
    if all(k in dfs for k in ("frozen", "non_frozen")):
        plt.figure()
        # normalize client ids to strings to avoid mixed-type sorting/comparison
        cids_nf = set(
            dfs["non_frozen"]
            .loc[dfs["non_frozen"]["client_id"] != "GLOBAL", "client_id"]
            .astype(str)
        )
        cids_fr = set(
            dfs["frozen"]
            .loc[dfs["frozen"]["client_id"] != "GLOBAL", "client_id"]
            .astype(str)
        )
        # sort numerically when possible, otherwise lexicographically
        cids = sorted(
            cids_nf | cids_fr,
            key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x),
        )

        for cid in cids:
            for name, style in [("non_frozen", "-"), ("frozen", "--")]:
                sub_df = _collapse_rows(
                    dfs[name].loc[dfs[name]["client_id"].astype(str) == cid].copy(),
                    ["client_id", "round"],
                )
                if sub_df is not None and not sub_df.empty:
                    sub_df = sub_df.sort_values("round")
                    lbl = f"Client {cid} ({'unfrozen' if name == 'non_frozen' else 'frozen'})"
                    plt.plot(
                        sub_df["round"],
                        smooth(sub_df["accuracy"]),
                        linestyle=style,
                        label=lbl,
                        alpha=0.8,
                    )
        plt.xlabel("Round");
        plt.ylabel("Local Accuracy")
        plt.title(f"Per-Client Accuracy Trajectories (Frozen vs Unfrozen) — {model}")
        plt.legend(ncol=2);
        plt.grid(True);
        plt.tight_layout()
        plt.savefig(OUT / f"per_client_trajectories{SFX}.png", dpi=150);
        plt.close()

    # -- Per-client accuracy lines per run (no GLOBAL) ---------------------
    for name, df in dfs.items():
        plt.figure()
        sub = _collapse_rows(df[df["client_id"] != "GLOBAL"].copy(), ["client_id", "round"])
        if sub is not None and not sub.empty:
            sub = sub.sort_values(["client_id","round"])
            for cid, grp in sub.groupby("client_id"):
                plt.plot(grp["round"], smooth(grp["accuracy"]), alpha=0.8, label=f"Client {cid}")
        plt.xlabel("Round"); plt.ylabel("Accuracy")
        plt.title(f"Per-Client Accuracy — {name} - {model}")
        plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
        plt.savefig(OUT / f"per_client_accuracy_{name}{SFX}.png", dpi=150); plt.close()

    # -- Per-class accuracy over rounds (line plot) ------------------------
    if percls:
        plt.figure()
        if "non_frozen" in percls:
            nf = _collapse_rows(percls["non_frozen"].copy(), ["round"]).sort_values("round")
            for label, series in perclass_columns(nf).items():
                plt.plot(nf["round"], smooth(series), label=f"{label} (unfrozen)")
        if "frozen" in percls:
            fr = _collapse_rows(percls["frozen"].copy(), ["round"]).sort_values("round")
            for label, series in perclass_columns(fr).items():
                plt.plot(fr["round"], smooth(series), linestyle="--", label=f"{label} (frozen)")
        plt.xlabel("Round"); plt.ylabel("Per-class Accuracy")
        plt.title(f"Per-class Accuracy per Round — {model}")
        plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
        plt.savefig(OUT / f"perclass_over_rounds{SFX}.png", dpi=150); plt.close()

    # -- Final-round per-class accuracy (bar) -------------------------------
    if percls:
        labels = SUPERCLASSES
        x = np.arange(len(labels)); width = 0.35
        plt.figure()
        if "non_frozen" in percls:
            nf_last = _collapse_rows(percls["non_frozen"].copy(), ["round"]).sort_values("round").tail(1)
            nf_map  = perclass_columns(nf_last)
            y_nf    = [float(nf_map.get(lbl, pd.Series([0])).iloc[0]) for lbl in labels]
            plt.bar(x - width/2, y_nf, width, label="Unfrozen")
        if "frozen" in percls:
            fr_last = _collapse_rows(percls["frozen"].copy(), ["round"]).sort_values("round").tail(1)
            fr_map  = perclass_columns(fr_last)
            y_fr    = [float(fr_map.get(lbl, pd.Series([0])).iloc[0]) for lbl in labels]
            plt.bar(x + width/2, y_fr, width, label="Frozen")
        plt.xticks(x, labels); plt.ylabel("Accuracy")
        plt.title(f"Final-Round Per-class Accuracy — {model}")
        plt.legend(); plt.tight_layout()
        plt.savefig(OUT / f"perclass_final_bar{SFX}.png", dpi=150); plt.close()

    # -- Confusion heatmaps (last round of each run) ------------------------
    for name, dfc in conf_long.items():
        if dfc is None or dfc.empty:
            continue
        r = int(dfc["round"].max())
        sub = dfc[dfc["round"] == r].copy()
        # remove duplicates for (true,pred) at this round
        sub = _collapse_rows(sub, ["true", "pred"])
        n = len(SUPERCLASSES)
        cm = (sub.pivot_table(index="true", columns="pred", values="count", aggfunc="sum", fill_value=0)
                .reindex(index=range(n), columns=range(n), fill_value=0)
                .to_numpy(dtype=float))
        with np.errstate(invalid="ignore", divide="ignore"):
            cmn = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
        plt.figure(figsize=(6,5))
        plt.imshow(cmn, aspect="auto")
        plt.title(f"Confusion Matrix — {name} — {model} (round {r})")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.xticks(range(n), SUPERCLASSES, rotation=45, ha="right")
        plt.yticks(range(n), SUPERCLASSES)
        plt.colorbar(label="Row-normalized")
        plt.tight_layout()
        plt.savefig(OUT / f"confusion_{name}{SFX}_r{r:02d}.png", dpi=150); plt.close()

    # -- Global accuracy by phase (within each run) ------------------------
    for name, df in dfs.items():
        if "phase" not in df.columns:
            continue
        plt.figure()
        g = df[df["client_id"] == "GLOBAL"].copy()
        lines = []
        for ph, style in [(PHASE_DISABLED,"-"), (PHASE_CACHED,":"), (PHASE_ENABLED,"--")]:
            sub = g[g["phase"].eq(ph)].sort_values(by="round")
            if not sub.empty:
                plt.plot(sub["round"], smooth(sub["accuracy"]), linestyle=style, label=f"{name} ({ph})")
                lines.append(ph)
        if not lines:
            plt.close()
            continue
        plt.xlabel("Round"); plt.ylabel("Global Accuracy")
        plt.title(f"Global Accuracy by Phase — {name} — {model}")
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(OUT / f"global_by_phase_{name}{SFX}.png", dpi=150); plt.close()

    # -- Per-client accuracy by phase (overlay pre/cached/post) ------------
    for name, df in dfs.items():
        if "phase" not in df.columns:
            continue
        plt.figure()
        base = df[df["client_id"] != "GLOBAL"].copy()
        for phase, style in [(PHASE_DISABLED,"-"), (PHASE_CACHED,":"), (PHASE_ENABLED,"--")]:
            grp = _collapse_rows(base[base["phase"].eq(phase)].copy(), ["client_id", "round"])
            if grp is None or grp.empty:
                continue
            for cid, g in grp.groupby("client_id"):
                g = g.sort_values(by="round")
                plt.plot(g["round"], smooth(g["accuracy"]), linestyle=style, alpha=0.7,
                         label=f"Client {cid} ({name}, {phase})")
        plt.xlabel("Round"); plt.ylabel("Local Accuracy")
        plt.title(f"Per-Client Accuracy by Phase — {name} — {model}")
        plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
        plt.savefig(OUT / f"clients_by_phase_{name}{SFX}.png", dpi=150); plt.close()

print(f"Saved plots to: {OUT.resolve()}")