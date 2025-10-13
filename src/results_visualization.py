import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from src.config import RESULTS_DIR, SUPERCLASSES, TUNING_DIR  # TUNING_DIR ok if unused
from src.utils import ensure_dir

# -------------------------------------------------------------------------
# Run discovery + naming
# -------------------------------------------------------------------------

# --- Auto-discover runs per model (cnn/lstm) -----------------------------
def _discover_runs():
    runs, percls, conf = {}, {}, {}
    # model = any chars not including underscore, right before _non_frozen/_frozen
    pat = re.compile(r"^(?P<model>[^_]+)_(?P<kind>non_frozen|frozen)_run$")
    for p in RESULTS_DIR.glob("*_*_run.csv"):
        m = pat.match(p.stem)
        if not m:
            continue
        model = m.group("model")            # "cnn" or "lstm"
        kind  = m.group("kind")             # "frozen" or "non_frozen"
        runs.setdefault(model, {})[kind] = p
    for model, vs in runs.items():
        percls[model], conf[model] = {}, {}
        if "frozen" in vs:
            percls[model]["frozen"] = RESULTS_DIR / f"{model}_frozen_run_perclass.csv"
            conf[model]["frozen"]   = RESULTS_DIR / f"{model}_frozen_run_cm.csv"
        if "non_frozen" in vs:
            percls[model]["non_frozen"] = RESULTS_DIR / f"{model}_non_frozen_run_perclass.csv"
            conf[model]["non_frozen"]   = RESULTS_DIR / f"{model}_non_frozen_run_cm.csv"
    return runs, percls, conf

PRETTY_RUN   = {"frozen": "Frozen", "non_frozen": "Unfrozen"}
PRETTY_MODEL = {"cnn": "CNN", "lstm": "LSTM"}
PRETTY_PHASE = {"no_cv": "No CV", "cached_cv": "Cached CV", "post_cv": "Post-CV"}

def title_of(base: str, run_name: str, model_key: str) -> str:
    return f"{base} — {PRETTY_RUN.get(run_name, run_name)} — {PRETTY_MODEL.get(model_key, model_key.upper())}"

RUNS, PERCLS_MAP, CONF_MAP = _discover_runs()
OUT = RESULTS_DIR / "viz"
ensure_dir(OUT)

# Phase labels written by training
PHASE_ENABLED  = "post_cv"
PHASE_DISABLED = "no_cv"
PHASE_CACHED   = "cached_cv"  # if you log cached phase

# -------------------------------------------------------------------------
# CSV loaders
# -------------------------------------------------------------------------
def load_numeric_csv(path: Path):
    if not path.exists():
        print(f"Missing: {path}")
        return None
    df = pd.read_csv(path)
    for col in ["round","accuracy","loss","wall_time_sec","trainable_params","true","pred","count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

dfs_by_model       = {m: {k: load_numeric_csv(p) for k, p in d.items()} for m, d in RUNS.items()}
percls_by_model    = {m: {k: load_numeric_csv(p) for k, p in d.items()} for m, d in PERCLS_MAP.items()}
conf_long_by_model = {m: {k: load_numeric_csv(p) for k, p in d.items()} for m, d in CONF_MAP.items()}

# Drop missing
for m in list(dfs_by_model.keys()):
    dfs_by_model[m]       = {k:v for k,v in dfs_by_model[m].items() if v is not None}
    percls_by_model[m]    = {k:v for k,v in percls_by_model.get(m, {}).items() if v is not None}
    conf_long_by_model[m] = {k:v for k,v in conf_long_by_model.get(m, {}).items() if v is not None}

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
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

# Smoothing (rolling mean)
SMOOTH = int(os.getenv("SMOOTH", "1"))  # 1 = off; 3/5 for presentation
def smooth(y):
    if SMOOTH <= 1:
        return y
    s = pd.Series(y, dtype=float)
    return s.rolling(SMOOTH, min_periods=1).mean().to_numpy()

# Phase de-dup: prefer later phase for same (client,round)
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
    return df.sort_values(keys).drop_duplicates(keys, keep="last")

# --------------------------------------------------------------------------------------
# ANOVA F-test on engineered features (SelectKBest) — robust to NaNs & constants
# --------------------------------------------------------------------------------------
def _load_features_csv_or_inline():
    """
    Return a DataFrame with numeric features + 'label'.
    1) Prefer RESULTS_DIR/'features.csv' if it exists.
    2) Otherwise, build features inline from PTB-XL using your data_loader.
    """
    path = RESULTS_DIR / "features.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            if "label" in df.columns:
                return df
            else:
                print("[anova] features.csv missing 'label' column; building inline.")
        except Exception as e:
            print(f"[anova] failed reading features.csv ({e}); building inline.")

    # ---- Inline build (uses your repo's helpers) ----
    try:
        from src.data_loader import (
            load_metadata, map_superclasses, filter_single_label,
            stratified_patient_split, make_feature_table
        )
        from src.config import SEED, SUPERCLASSES

        ptb = load_metadata()
        df_meta = filter_single_label(map_superclasses(ptb))
        # Use train split to avoid leakage into your test if you later compare
        train_df, _ = stratified_patient_split(df_meta, test_size=0.2, seed=SEED)

        # limit=None uses all; set a number for speed if you like
        X, y, _ = make_feature_table(train_df, limit=None)

        cols = (
            [f"mean_{i}" for i in range(12)] +
            [f"std_{i}"  for i in range(12)] +
            [f"rms_{i}"  for i in range(12)]
        )
        feat = pd.DataFrame(X, columns=cols)
        feat["label"] = y
        feat["y_name"] = [SUPERCLASSES[int(i)] for i in y]
        return feat
    except Exception as e:
        print(f"[anova] inline feature build failed: {e}")
        return pd.DataFrame()

def run_anova_selectkbest(outdir: Path):
    """
    Loads/creates a feature table and runs ANOVA (f_classif) with SelectKBest.
    Saves a top-25 bar chart and a CSV with all features ranked by F-score.
    """
    try:
        from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import LabelEncoder
    except Exception:
        print("[anova] scikit-learn not available — skipping.")
        return

    feature_df = _load_features_csv_or_inline()
    if feature_df.empty or "label" not in feature_df.columns:
        print("[anova] no features available — skipping.")
        return

    # numeric features only; handle inf/NaN
    X_num = (
        feature_df.drop(columns=["label"], errors="ignore")
                  .select_dtypes(include=[np.number])
                  .replace([np.inf, -np.inf], np.nan)
                  .copy()
    )
    y_all = feature_df["label"].astype(str).copy()

    # drop constant columns
    try:
        vt = VarianceThreshold(threshold=1e-12)
        X_vt = vt.fit_transform(X_num)
        cols_vt = X_num.columns[vt.get_support()]
    except Exception:
        print("[anova] no numeric features after preprocessing.")
        return
    if X_vt.shape[1] == 0:
        print("[anova] all numeric features were constant or removed.")
        return

    # impute missing values
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X_vt)

    # encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_all.values)

    # SelectKBest with ANOVA F
    k = int(min( max(1, X_imp.shape[1]), 20 ))  # up to 20 for the bar chart
    selector = SelectKBest(score_func=f_classif, k=k).fit(X_imp, y_enc)

    # Full ranking (not just top-k)
    # We need scores for all cols_vt; selector.scores_ is length == n_features after VT
    scores_all = pd.Series(selector.scores_, index=cols_vt).sort_values(ascending=False)
    scores_all = scores_all.replace({np.nan: 0.0})

    # Save full CSV
    scores_csv = RESULTS_DIR / "anova_fscores.csv"
    scores_all.to_frame("F_score").to_csv(scores_csv)
    print(f"[anova] wrote {scores_csv} (n={len(scores_all)})")

    # Plot top-25
    top = scores_all.head(25)
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    xs = np.arange(len(top))
    ax.bar(xs, top.values)
    ax.set_xticks(xs)
    ax.set_xticklabels(top.index, rotation=65, ha="right")
    ax.set_title("ANOVA F-scores (top 25)")
    ax.set_ylabel("F-score")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "anova_selectkbest_top25.png", dpi=150)
    plt.close(fig)
    print(f"[anova] saved {outdir / 'anova_selectkbest_top25.png'}")

# ---- Run it (will execute once when you call the viz script) ----
try:
    run_anova_selectkbest(OUT)
except Exception as _e:
    print(f"[anova] step skipped due to error: {_e}")

# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

for model in dfs_by_model.keys():
    dfs       = dfs_by_model[model]
    percls    = percls_by_model.get(model, {})
    conf_long = conf_long_by_model.get(model, {})
    SFX = f"_{model}"

    # Global accuracy vs round
    plt.figure()
    for name, df in dfs.items():
        g = _collapse_rows(df[df["client_id"] == "GLOBAL"].copy(), ["client_id", "round"])
        if g is not None and len(g):
            plt.plot(g["round"], smooth(g["accuracy"]), label=f"Global ({PRETTY_RUN.get(name,name)})")
    plt.xlabel("Round"); plt.ylabel("Accuracy")
    plt.title(title_of("Global Accuracy per Round", "non_frozen", model))
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"global_accuracy{SFX}.png", dpi=150); plt.close()

    # Client accuracies vs round (with GLOBAL overlay)
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
        plt.title(title_of("Client vs Global Accuracy", name, model))
        plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
        plt.savefig(OUT / f"clients_vs_global_{name}{SFX}.png", dpi=150); plt.close()

    # Mean client accuracy per round (not GLOBAL)
    plt.figure()
    for name, df in dfs.items():
        sub = _collapse_rows(df[df["client_id"] != "GLOBAL"].copy(), ["client_id", "round"])
        if sub is None or sub.empty:
            continue
        agg = (sub.groupby("round", as_index=False)
                  .agg(mean_acc=("accuracy","mean"))
                  .sort_values("round"))
        plt.plot(agg["round"], smooth(agg["mean_acc"]), label=f"{PRETTY_RUN.get(name,name)}")
    plt.xlabel("Round"); plt.ylabel("Mean Client Accuracy")
    plt.title(title_of("Mean Client Accuracy per Round", "non_frozen", model))
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(OUT / f"global_mean_accuracy{SFX}.png", dpi=150); plt.close()

    # Wall-time per round (total across clients)
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
        plt.title(title_of("Total Wall-Time per Round", name, model))
        plt.grid(True); plt.tight_layout()
        plt.savefig(OUT / f"walltime_{name}{SFX}.png", dpi=150); plt.close()

    # Per-client trajectories (frozen vs non-frozen, one figure)
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
                    lbl = f"Client {cid} ({'Unfrozen' if name == 'non_frozen' else 'Frozen'})"
                    plt.plot(
                        sub_df["round"],
                        smooth(sub_df["accuracy"]),
                        linestyle=style,
                        label=lbl,
                        alpha=0.8,
                    )
        plt.xlabel("Round"); plt.ylabel("Local Accuracy")
        plt.title(title_of("Per-Client Accuracy Trajectories (Frozen vs Unfrozen)", "non_frozen", model))
        plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
        plt.savefig(OUT / f"per_client_trajectories{SFX}.png", dpi=150); plt.close()

    # Per-client accuracy lines per run (no GLOBAL)
    for name, df in dfs.items():
        plt.figure()
        sub = _collapse_rows(df[df["client_id"] != "GLOBAL"].copy(), ["client_id", "round"])
        if sub is not None and not sub.empty:
            sub = sub.sort_values(["client_id","round"])
            for cid, grp in sub.groupby("client_id"):
                plt.plot(grp["round"], smooth(grp["accuracy"]), alpha=0.8, label=f"Client {cid}")
        plt.xlabel("Round"); plt.ylabel("Accuracy")
        plt.title(title_of("Per-Client Accuracy", name, model))
        plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
        plt.savefig(OUT / f"per_client_accuracy_{name}{SFX}.png", dpi=150); plt.close()

    # Per-class accuracy over rounds (line plot)
    if percls:
        plt.figure()
        if "non_frozen" in percls:
            nf = _collapse_rows(percls["non_frozen"].copy(), ["round"]).sort_values("round")
            for label, series in perclass_columns(nf).items():
                plt.plot(nf["round"], smooth(series), label=f"{label} (Unfrozen)")
        if "frozen" in percls:
            fr = _collapse_rows(percls["frozen"].copy(), ["round"]).sort_values("round")
            for label, series in perclass_columns(fr).items():
                plt.plot(fr["round"], smooth(series), linestyle="--", label=f"{label} (Frozen)")
        plt.xlabel("Round"); plt.ylabel("Per-class Accuracy")
        plt.title(title_of("Per-class Accuracy per Round", "non_frozen", model))
        plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
        plt.savefig(OUT / f"perclass_over_rounds{SFX}.png", dpi=150); plt.close()

    # Final-round per-class accuracy (bar)
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
        plt.title(title_of("Final-Round Per-class Accuracy", "non_frozen", model))
        plt.legend(); plt.tight_layout()
        plt.savefig(OUT / f"perclass_final_bar{SFX}.png", dpi=150); plt.close()

    # Confusion heatmaps (last round of each run)
    for name, dfc in conf_long.items():
        if dfc is None or dfc.empty:
            continue
        r = int(dfc["round"].max())
        sub = dfc[dfc["round"] == r].copy()
        sub = _collapse_rows(sub, ["true", "pred"])  # de-dup pairs
        n = len(SUPERCLASSES)
        cm = (sub.pivot_table(index="true", columns="pred", values="count", aggfunc="sum", fill_value=0)
                .reindex(index=range(n), columns=range(n), fill_value=0)
                .to_numpy(dtype=float))
        with np.errstate(invalid="ignore", divide="ignore"):
            cmn = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True))
        plt.figure(figsize=(6,5))
        plt.imshow(cmn, aspect="auto")
        plt.title(title_of(f"Confusion Matrix (round {r})", name, model))
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.xticks(range(n), SUPERCLASSES, rotation=45, ha="right")
        plt.yticks(range(n), SUPERCLASSES)
        plt.colorbar(label="Row-normalized")
        plt.tight_layout()
        plt.savefig(OUT / f"confusion_{name}{SFX}_r{r:02d}.png", dpi=150); plt.close()

    # Global accuracy by phase (within each run)
    for name, df in dfs.items():
        if "phase" not in df.columns:
            continue
        plt.figure()
        g = df[df["client_id"] == "GLOBAL"].copy()
        lines = []
        for ph, style in [(PHASE_DISABLED,"-"), (PHASE_CACHED,":"), (PHASE_ENABLED,"--")]:
            sub = g[g["phase"].eq(ph)].sort_values(by="round")
            if not sub.empty:
                plt.plot(sub["round"], smooth(sub["accuracy"]), linestyle=style,
                         label=f"{PRETTY_RUN.get(name,name)} ({PRETTY_PHASE.get(ph,ph)})")
                lines.append(ph)
        if not lines:
            plt.close(); continue
        plt.xlabel("Round"); plt.ylabel("Global Accuracy")
        plt.title(title_of("Global Accuracy by Phase", name, model))
        plt.legend(); plt.grid(True); plt.tight_layout()
        plt.savefig(OUT / f"global_by_phase_{name}{SFX}.png", dpi=150); plt.close()

    # Per-client accuracy by phase (overlay pre/cached/post)
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
                         label=f"Client {cid} ({PRETTY_RUN.get(name,name)}, {PRETTY_PHASE.get(phase,phase)})")
        plt.xlabel("Round"); plt.ylabel("Local Accuracy")
        plt.title(title_of("Per-Client Accuracy by Phase", name, model))
        plt.legend(ncol=2); plt.grid(True); plt.tight_layout()
        plt.savefig(OUT / f"clients_by_phase_{name}{SFX}.png", dpi=150); plt.close()

print(f"Saved plots to: {OUT.resolve()}")
