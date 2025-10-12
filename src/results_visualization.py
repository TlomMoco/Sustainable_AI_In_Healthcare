import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import RESULTS_DIR, SUPERCLASSES
from src.utils import ensure_dir

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
PHASE_CACHED   = "cached_cv"




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

# --- Load all dataframes (calling load_numeric_csv) ---------------------
dfs      = {name: load_numeric_csv(p) for name, p in FILES.items()}
percls   = {name: load_numeric_csv(p) for name, p in PERCLASS.items()}
conf_long= {name: load_numeric_csv(p) for name, p in CONFUSION.items()}

dfs      = {k:v for k,v in dfs.items() if v is not None}
percls   = {k:v for k,v in percls.items() if v is not None}
conf_long= {k:v for k,v in conf_long.items() if v is not None}


# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------
def perclass_columns(df):
    """Return mapping label-> Series, accepting acc_<name> or acc_<index>."""
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


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

# -- Global accuracy vs round ----------------------------------
plt.figure()
for name, df in dfs.items():
    g = df[df["client_id"] == "GLOBAL"].sort_values("round")
    if len(g):
        plt.plot(g["round"], smooth(g["accuracy"]), label=f"Global ({name})")
plt.xlabel("Round"); plt.ylabel("Accuracy"); plt.title("Global Accuracy per Round")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(OUT / "global_accuracy.png", dpi=150); plt.close()


# -- Client accuracies vs round ---------------------------------
for name, df in dfs.items():
    plt.figure()
    sub = df[df["client_id"] != "GLOBAL"].sort_values(["client_id", "round"])
    for cid, grp in sub.groupby("client_id"):
        plt.plot(grp["round"], smooth(grp["accuracy"]), alpha=0.6, label=f"Client {cid}")
    g = df[df["client_id"] == "GLOBAL"].sort_values("round")
    if len(g):
        plt.plot(g["round"], smooth(g["accuracy"]), linewidth=3, linestyle="--", label="GLOBAL")
    plt.xlabel("Round"); plt.ylabel("Accuracy")
    plt.title(f"Client vs Global Accuracy — {name}")
    plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"clients_vs_global_{name}.png", dpi=150); plt.close()


# -- Mean client accuracy per round (not GLOBAL) -------------------------
plt.figure()
for name, df in dfs.items():
    sub = df[df["client_id"] != "GLOBAL"]
    if sub.empty:
        continue
    agg = (sub.groupby("round", as_index=False)
              .agg(mean_acc=("accuracy","mean"))
              .sort_values("round"))
    plt.plot(agg["round"], smooth(agg["mean_acc"]), label=f"{name} (mean client)")
plt.xlabel("Round"); plt.ylabel("Mean Client Accuracy")
plt.title("Mean Client Accuracy per Round")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(OUT / "global_mean_accuracy.png", dpi=150); plt.close()


# -- Wall-time per round (total across clients) -------------------
for name, df in dfs.items():
    if "wall_time_sec" not in df.columns:
        continue
    sub = df[df["client_id"] != "GLOBAL"]
    agg = (
        sub.groupby("round", as_index=False)
           .agg(wall_time_sec=("wall_time_sec", "sum"))
           .sort_values(by="round")
    )
    plt.figure()
    plt.plot(agg["round"], agg["wall_time_sec"])
    plt.xlabel("Round"); plt.ylabel("Total client wall-time (s)")
    plt.title(f"Total Wall-Time per Round — {name}")
    plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"walltime_{name}.png", dpi=150); plt.close()


# -- Per-client trajectories (frozen vs non-frozen, one figure) ---
if all(k in dfs for k in ("frozen", "non_frozen")):
    plt.figure()
    cids = set(dfs["non_frozen"].query("client_id!='GLOBAL'")["client_id"]).union(
           set(dfs["frozen"].query("client_id!='GLOBAL'")["client_id"]))
    for cid in sorted(cids):
        for name, style in [("non_frozen","-"), ("frozen","--")]:
            sub_df = dfs[name].loc[dfs[name]["client_id"] == cid].copy()  # DataFrame
            sub_df.sort_values(by="round", inplace=True)                  # sort DataFrame
            if not sub_df.empty:
                lbl = f"Client {cid} ({'unfrozen' if name=='non_frozen' else 'frozen'})"
                plt.plot(sub_df["round"], smooth(sub_df["accuracy"]), linestyle=style, label=lbl, alpha=0.8)
    plt.xlabel("Round"); plt.ylabel("Local Accuracy")
    plt.title("Per-Client Accuracy Trajectories (Frozen vs Unfrozen)")
    plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / "per_client_trajectories.png", dpi=150); plt.close()


# -- Per-client accuracy lines per run (no GLOBAL) -----------------------
for name, df in dfs.items():
    plt.figure()
    sub = df[df["client_id"] != "GLOBAL"].sort_values(["client_id","round"])
    for cid, grp in sub.groupby("client_id"):
        plt.plot(grp["round"], smooth(grp["accuracy"]), alpha=0.8, label=f"Client {cid}")
    plt.xlabel("Round"); plt.ylabel("Accuracy")
    plt.title(f"Per-Client Accuracy — {name}")
    plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"per_client_accuracy_{name}.png", dpi=150); plt.close()


# -- Per-class accuracy over rounds (line plot) --
if percls:
    plt.figure()
    if "non_frozen" in percls:
        nf = percls["non_frozen"].sort_values(by="round")
        for label, series in perclass_columns(nf).items():
            plt.plot(nf["round"], smooth(series), label=f"{label} (unfrozen)")
    if "frozen" in percls:
        fr = percls["frozen"].sort_values(by="round")
        for label, series in perclass_columns(fr).items():
            plt.plot(fr["round"], smooth(series), linestyle="--", label=f"{label} (frozen)")
    plt.xlabel("Round"); plt.ylabel("Per-class Accuracy")
    plt.title("Per-class Accuracy per Round")
    plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / "perclass_over_rounds.png", dpi=150); plt.close()


# -- Final-round per-class accuracy (bar) --
if percls:
    labels = SUPERCLASSES
    x = np.arange(len(labels)); width = 0.35
    plt.figure()
    if "non_frozen" in percls:
        nf_last = percls["non_frozen"].sort_values(by="round").tail(1)
        nf_map  = perclass_columns(nf_last)
        y_nf    = [float(nf_map.get(lbl, pd.Series([0])).iloc[0]) for lbl in labels]
        plt.bar(x - width/2, y_nf, width, label="Unfrozen")
    if "frozen" in percls:
        fr_last = percls["frozen"].sort_values(by="round").tail(1)
        fr_map  = perclass_columns(fr_last)
        y_fr    = [float(fr_map.get(lbl, pd.Series([0])).iloc[0]) for lbl in labels]
        plt.bar(x + width/2, y_fr, width, label="Frozen")
    plt.xticks(x, labels); plt.ylabel("Accuracy")
    plt.title("Final-Round Per-class Accuracy")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "perclass_final_bar.png", dpi=150); plt.close()


# -- Confusion heatmaps (last round of each run) ------------------
for name, dfc in conf_long.items():
    if dfc is None or dfc.empty:
        continue
    r = int(dfc["round"].max())
    sub = dfc[dfc["round"] == r]
    n = len(SUPERCLASSES)
    cm = (sub.pivot_table(index="true", columns="pred", values="count", aggfunc="sum", fill_value=0)
              .reindex(index=range(n), columns=range(n), fill_value=0)
              .to_numpy(dtype=float))
    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
    plt.figure(figsize=(6,5))
    plt.imshow(cmn, aspect="auto")
    plt.title(f"Confusion Matrix — {name} (round {r})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(n), SUPERCLASSES, rotation=45, ha="right")
    plt.yticks(range(n), SUPERCLASSES)
    plt.colorbar(label="Row-normalized")
    plt.tight_layout()
    plt.savefig(OUT / f"confusion_{name}_r{r:02d}.png", dpi=150); plt.close()


# -- Global accuracy by phase (within each run) ---------------------------
for name, df in dfs.items():
    if "phase" not in df.columns:
        continue
    g = df[df["client_id"] == "GLOBAL"].copy()
    lines = []
    for ph, style in [(PHASE_DISABLED,"-"), (PHASE_CACHED,":"), (PHASE_ENABLED,"--")]:
        sub = g[g["phase"].eq(ph)].sort_values(by="round")
        if not sub.empty:
            plt.plot(sub["round"], smooth(sub["accuracy"]), linestyle=style, label=f"{name} ({ph})")
            lines.append(ph)
    if not lines:
        continue
    plt.xlabel("Round"); plt.ylabel("Global Accuracy")
    plt.title(f"Global Accuracy by Phase — {name}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"global_by_phase_{name}.png", dpi=150); plt.close()


# -- Per-client accuracy by phase (overlay pre/cached/post) ---------------
for name, df in dfs.items():
    if "phase" not in df.columns:
        continue
    plt.figure()
    sub = df[df["client_id"] != "GLOBAL"].copy()
    for phase, style in [(PHASE_DISABLED,"-"), (PHASE_CACHED,":"), (PHASE_ENABLED,"--")]:
        grp = sub[sub["phase"].eq(phase)]
        for cid, g in grp.groupby("client_id"):
            g = g.sort_values(by="round")
            plt.plot(g["round"], smooth(g["accuracy"]), linestyle=style, alpha=0.7,
                     label=f"Client {cid} ({name}, {phase})")
    plt.xlabel("Round"); plt.ylabel("Local Accuracy")
    plt.title(f"Per-Client Accuracy by Phase — {name}")
    plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"clients_by_phase_{name}.png", dpi=150); plt.close()


print(f"Saved plots to: {OUT.resolve()}")