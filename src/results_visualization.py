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

def load_numeric_csv(path: Path):
    if not path.exists():
        print(f"Missing: {path}")
        return None
    df = pd.read_csv(path)
    for col in ["round","accuracy","loss","wall_time_sec","trainable_params","true","pred","count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

dfs      = {name: load_numeric_csv(p) for name, p in FILES.items()}
percls   = {name: load_numeric_csv(p) for name, p in PERCLASS.items()}
conf_long= {name: load_numeric_csv(p) for name, p in CONFUSION.items()}

dfs      = {k:v for k,v in dfs.items() if v is not None}
percls   = {k:v for k,v in percls.items() if v is not None}
conf_long= {k:v for k,v in conf_long.items() if v is not None}

# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

# -- Global accuracy vs round ----------------------------------
plt.figure()
for name, df in dfs.items():
    g = df[df["client_id"] == "GLOBAL"].sort_values("round")
    if len(g):
        plt.plot(g["round"], g["accuracy"], label=f"Global ({name})")
plt.xlabel("Round"); plt.ylabel("Accuracy"); plt.title("Global Accuracy per Round")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(OUT / "global_accuracy.png", dpi=150); plt.close()

# -- Client accuracies vs round ---------------------------------
for name, df in dfs.items():
    plt.figure()
    sub = df[df["client_id"] != "GLOBAL"].sort_values(["client_id", "round"])
    for cid, grp in sub.groupby("client_id"):
        plt.plot(grp["round"], grp["accuracy"], alpha=0.6, label=f"Client {cid}")
    g = df[df["client_id"] == "GLOBAL"].sort_values("round")
    if len(g):
        plt.plot(g["round"], g["accuracy"], linewidth=3, linestyle="--", label="GLOBAL")
    plt.xlabel("Round"); plt.ylabel("Accuracy")
    plt.title(f"Client vs Global Accuracy — {name}")
    plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"clients_vs_global_{name}.png", dpi=150); plt.close()

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
                plt.plot(sub_df["round"], sub_df["accuracy"], linestyle=style, label=lbl, alpha=0.8)
    plt.xlabel("Round"); plt.ylabel("Local Accuracy")
    plt.title("Per-Client Accuracy Trajectories (Frozen vs Unfrozen)")
    plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / "per_client_trajectories.png", dpi=150); plt.close()

# -- Per-class accuracy over rounds (line plot) -------------------
if percls:
    plt.figure()
    if "non_frozen" in percls:
        nf = percls["non_frozen"].sort_values("round")
        for c in SUPERCLASSES:
            plt.plot(nf["round"], nf[f"acc_{c}"], label=f"{c} (unfrozen)")
    if "frozen" in percls:
        fr = percls["frozen"].sort_values("round")
        for c in SUPERCLASSES:
            plt.plot(fr["round"], fr[f"acc_{c}"], linestyle="--", label=f"{c} (frozen)")
    plt.xlabel("Round"); plt.ylabel("Per-class Accuracy")
    plt.title("Per-class Accuracy per Round")
    plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / "perclass_over_rounds.png", dpi=150); plt.close()

# -- Final-round per-class accuracy (bar) -------------------------
if percls:
    labels = SUPERCLASSES; x = np.arange(len(labels)); width = 0.35
    plt.figure()
    if "non_frozen" in percls:
        nf_last = percls["non_frozen"].sort_values("round").tail(1)
        y_nf = [float(nf_last[f"acc_{c}"].values[0]) for c in labels]
        plt.bar(x - width/2, y_nf, width, label="Unfrozen")
    if "frozen" in percls:
        fr_last = percls["frozen"].sort_values("round").tail(1)
        y_fr = [float(fr_last[f"acc_{c}"].values[0]) for c in labels]
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

    # pivot to matrix (rows=true, cols=pred), fill missing with 0
    cm = (
        sub.pivot_table(index="true", columns="pred", values="count",
                        aggfunc="sum", fill_value=0)
           .reindex(index=range(n), columns=range(n), fill_value=0)
           .to_numpy(dtype=float)
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = cm / cm.sum(axis=1, keepdims=True)
        cmn = np.nan_to_num(cmn)
    plt.figure(figsize=(6, 5))
    plt.imshow(cmn, aspect="auto")
    plt.title(f"Confusion Matrix — {name} (round {r})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(range(n), SUPERCLASSES, rotation=45, ha="right")
    plt.yticks(range(n), SUPERCLASSES)
    plt.colorbar(label="Row-normalized")
    plt.tight_layout()
    plt.savefig(OUT / f"confusion_{name}_r{r:02d}.png", dpi=150)
    plt.close()

print(f"Saved plots to: {OUT.resolve()}")