import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, SUPERCLASSES, N_CLASSES
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
        model = m.group("model")  # "cnn" or "lstm"
        kind = m.group("kind")  # "frozen" or "non_frozen"
        runs.setdefault(model, {})[kind] = p
    for model, vs in runs.items():
        percls[model], conf[model] = {}, {}
        if "frozen" in vs:
            percls[model]["frozen"] = RESULTS_DIR / f"{model}_frozen_run_perclass.csv"
            conf[model]["frozen"] = RESULTS_DIR / f"{model}_frozen_run_cm.csv"
        if "non_frozen" in vs:
            percls[model]["non_frozen"] = RESULTS_DIR / f"{model}_non_frozen_run_perclass.csv"
            conf[model]["non_frozen"] = RESULTS_DIR / f"{model}_non_frozen_run_cm.csv"
    return runs, percls, conf


PRETTY_RUN = {"frozen": "Frozen", "non_frozen": "Unfrozen"}
PRETTY_MODEL = {"cnn":"CNN","lstm":"LSTM","gru":"GRU","cnn_lstm":"CNN+LSTM","mlp":"MLP"}
# Show nice labels for both old and new phase names (labels only; filtering uses constants below)
PRETTY_PHASE = {
    "no_cv": "No tuning/CV",
    "cached": "Tuned/CV",
    "enabled": "Tuned/CV",
    "default": "Default",
    "tuned": "Tuned",
    "cached_cv": "Tuned (cached)",
    "post_cv": "Tuned",
}


def title_of(base: str, run_name: str, model_key: str) -> str:
    return f"{base} — {PRETTY_RUN.get(run_name, run_name)} — {PRETTY_MODEL.get(model_key, model_key.upper())}"


RUNS, PERCLS_MAP, CONF_MAP = _discover_runs()

# -------------------------------------------------------------------------
# Phase labels written by training (FILTER VALUES)
# Your logs use "default"/"tuned"; treat cached best as tuned.
# -------------------------------------------------------------------------
PHASE_ENABLED = "tuned"
PHASE_DISABLED = "default"
PHASE_CACHED = "tuned"  # cached best → treated as tuned


# -------------------------------------------------------------------------
# CSV loaders
# -------------------------------------------------------------------------
def load_numeric_csv(path: Path):
    if not path.exists():
        print(f"Missing: {path}")
        return None
    df = pd.read_csv(path)
    for col in ["round", "accuracy", "loss", "wall_time_sec", "trainable_params", "true", "pred", "count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


dfs_by_model = {m: {k: load_numeric_csv(p) for k, p in d.items()} for m, d in RUNS.items()}
percls_by_model = {m: {k: load_numeric_csv(p) for k, p in d.items()} for m, d in PERCLS_MAP.items()}
conf_long_by_model = {m: {k: load_numeric_csv(p) for k, p in d.items()} for m, d in CONF_MAP.items()}

# Drop missing
for m in list(dfs_by_model.keys()):
    dfs_by_model[m] = {k: v for k, v in dfs_by_model[m].items() if v is not None}
    percls_by_model[m] = {k: v for k, v in percls_by_model.get(m, {}).items() if v is not None}
    conf_long_by_model[m] = {k: v for k, v in conf_long_by_model.get(m, {}).items() if v is not None}


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

# Phase styles (for line plots)
def _unique_phase_styles():
    """Return (phase, linestyle) pairs with duplicates removed, in order."""
    seen = set()
    out = []
    for ph, style in [(PHASE_DISABLED, "-"), (PHASE_CACHED, ":"), (PHASE_ENABLED, "--")]:
        if ph and ph not in seen:
            out.append((ph, style))
            seen.add(ph)
    return out

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


# Per-model OUT helper
def _out_dir_for(model_key: str) -> Path:
    """Return and ensure the output directory for a given model (CNN/LSTM)."""
    pretty = PRETTY_MODEL.get(model_key, model_key.upper())
    d = RESULTS_DIR / "viz" / f"FL_Training_{pretty}"
    ensure_dir(d)
    return d


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------

_saved_dirs = set()

for model in dfs_by_model.keys():
    dfs = dfs_by_model[model]
    percls = percls_by_model.get(model, {})
    conf_long = conf_long_by_model.get(model, {})
    SFX = f"_{model}"

    # Create per-model OUT directory
    outdir = _out_dir_for(model)
    _saved_dirs.add(str(outdir.resolve()))

    # Global accuracy vs round
    plt.figure()
    for name, df in dfs.items():
        g = _collapse_rows(df[df["client_id"] == "GLOBAL"].copy(), ["client_id", "round"])
        if g is not None and len(g):
            plt.plot(g["round"], smooth(g["accuracy"]), label=f"Global ({PRETTY_RUN.get(name, name)})")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title(title_of("Global Accuracy per Round", "non_frozen", model))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / f"global_accuracy{SFX}.png", dpi=150)
    plt.close()

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
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title(title_of("Client vs Global Accuracy", name, model))
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / f"clients_vs_global_{name}{SFX}.png", dpi=150)
        plt.close()

    # Mean client accuracy per round (not GLOBAL)
    plt.figure()
    for name, df in dfs.items():
        sub = _collapse_rows(df[df["client_id"] != "GLOBAL"].copy(), ["client_id", "round"])
        if sub is None or sub.empty:
            continue
        agg = (sub.groupby("round", as_index=False)
               .agg(mean_acc=("accuracy", "mean"))
               .sort_values("round"))
        plt.plot(agg["round"], smooth(agg["mean_acc"]), label=f"{PRETTY_RUN.get(name, name)}")
    plt.xlabel("Round")
    plt.ylabel("Mean Client Accuracy")
    plt.title(title_of("Mean Client Accuracy per Round", "non_frozen", model))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outdir / f"global_mean_accuracy{SFX}.png", dpi=150)
    plt.close()

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
        plt.xlabel("Round")
        plt.ylabel("Total client wall-time (s)")
        plt.title(title_of("Total Wall-Time per Round", name, model))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / f"walltime_{name}{SFX}.png", dpi=150)
        plt.close()

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
        plt.xlabel("Round")
        plt.ylabel("Local Accuracy")
        plt.title(title_of("Per-Client Accuracy Trajectories (Frozen vs Unfrozen)", "non_frozen", model))
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / f"per_client_trajectories{SFX}.png", dpi=150)
        plt.close()

    # Per-client accuracy lines per run (no GLOBAL)
    for name, df in dfs.items():
        plt.figure()
        sub = _collapse_rows(df[df["client_id"] != "GLOBAL"].copy(), ["client_id", "round"])
        if sub is not None and not sub.empty:
            sub = sub.sort_values(["client_id", "round"])
            for cid, grp in sub.groupby("client_id"):
                plt.plot(grp["round"], smooth(grp["accuracy"]), alpha=0.8, label=f"Client {cid}")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")
        plt.title(title_of("Per-Client Accuracy", name, model))
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / f"per_client_accuracy_{name}{SFX}.png", dpi=150)
        plt.close()

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
        plt.xlabel("Round")
        plt.ylabel("Per-class Accuracy")
        plt.title(title_of("Per-class Accuracy per Round", "non_frozen", model))
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / f"perclass_over_rounds{SFX}.png", dpi=150)
        plt.close()

    # Final-round per-class accuracy (bar)
    if percls:
        labels = SUPERCLASSES
        x = np.arange(len(labels))
        width = 0.35
        plt.figure()
        if "non_frozen" in percls:
            nf_last = _collapse_rows(percls["non_frozen"].copy(), ["round"]).sort_values("round").tail(1)
            nf_map = perclass_columns(nf_last)
            y_nf = [float(nf_map.get(lbl, pd.Series([0])).iloc[0]) for lbl in labels]
            plt.bar(x - width / 2, y_nf, width, label="Unfrozen")
        if "frozen" in percls:
            fr_last = _collapse_rows(percls["frozen"].copy(), ["round"]).sort_values("round").tail(1)
            fr_map = perclass_columns(fr_last)
            y_fr = [float(fr_map.get(lbl, pd.Series([0])).iloc[0]) for lbl in labels]
            plt.bar(x + width / 2, y_fr, width, label="Frozen")
        plt.xticks(x, labels)
        plt.ylabel("Accuracy")
        plt.title(title_of("Final-Round Per-class Accuracy", "non_frozen", model))
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"perclass_final_bar{SFX}.png", dpi=150)
        plt.close()

    # =====================================================================
    # CONFUSION HEATMAPS - PROFESSIONAL STYLE WITH MODEL NAME IN TITLE
    # =====================================================================
    for name, dfc in conf_long.items():
        if dfc is None or dfc.empty:
            continue
        r = int(dfc["round"].max())
        sub = dfc[dfc["round"] == r].copy()
        sub = _collapse_rows(sub, ["true", "pred"])
        n = len(SUPERCLASSES)

        # Build confusion matrix
        cm = (sub.pivot_table(index="true", columns="pred", values="count", aggfunc="sum", fill_value=0)
              .reindex(index=range(n), columns=range(n), fill_value=0)
              .to_numpy(dtype=float))

        # Normalize to percentages
        with np.errstate(invalid="ignore", divide="ignore"):
            cmn = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True)) * 100

        # Create professional heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cmn, aspect="auto", cmap="Blues", vmin=0, vmax=100)

        # Add text annotations
        for i in range(n):
            for j in range(n):
                val = cmn[i, j]
                text_color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        color=text_color, fontsize=12, weight="bold")

        # Styling with MODEL NAME in title
        model_name = PRETTY_MODEL.get(model, model.upper())
        run_type = PRETTY_RUN.get(name, name)
        ax.set_title(
            f"Confusion Matrix - Prediction Percentages\n{model_name} - {run_type} (Round {r})",
            fontsize=16, pad=20, weight="bold"
        )
        ax.set_xlabel("Predicted Labels", fontsize=14, labelpad=10)
        ax.set_ylabel("True Labels", fontsize=14, labelpad=10)

        # Set ticks
        ax.set_xticks(range(n))
        ax.set_xticklabels(SUPERCLASSES, fontsize=12)
        ax.set_yticks(range(n))
        ax.set_yticklabels(SUPERCLASSES, fontsize=12)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("%", fontsize=14, rotation=0, labelpad=20)
        cbar.ax.tick_params(labelsize=11)

        # Save
        fig.tight_layout()
        plt.savefig(outdir / f"confusion_{name}{SFX}_r{r:02d}.png",
                    dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # Global accuracy by phase (within each run)
    for name, df in dfs.items():
        if "phase" not in df.columns:
            continue
        plt.figure()
        g = df[df["client_id"] == "GLOBAL"].copy()
        lines = []
        for ph, style in _unique_phase_styles():
            sub = g[g["phase"].eq(ph)].sort_values(by="round")
            if not sub.empty:
                plt.plot(sub["round"], smooth(sub["accuracy"]), linestyle=style,
                         label=f"{PRETTY_RUN.get(name, name)} ({PRETTY_PHASE.get(ph, ph)})")
                lines.append(ph)
        if not lines:
            plt.close()
            continue
        plt.xlabel("Round")
        plt.ylabel("Global Accuracy")
        plt.title(title_of("Global Accuracy by Phase", name, model))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / f"global_by_phase_{name}{SFX}.png", dpi=150)
        plt.close()

    # Per-client accuracy by phase (overlay pre/cached/post)
    for name, df in dfs.items():
        if "phase" not in df.columns:
            continue
        plt.figure()
        base = df[df["client_id"] != "GLOBAL"].copy()
        for phase, style in _unique_phase_styles():
            grp = _collapse_rows(base[base["phase"].eq(phase)].copy(), ["client_id", "round"])
            if grp is None or grp.empty:
                continue
            for cid, g in grp.groupby("client_id"):
                g = g.sort_values(by="round")
                plt.plot(g["round"], smooth(g["accuracy"]), linestyle=style, alpha=0.7,
                         label=f"Client {cid} ({PRETTY_RUN.get(name, name)}, {PRETTY_PHASE.get(phase, phase)})")
        plt.xlabel("Round")
        plt.ylabel("Local Accuracy")
        plt.title(title_of("Per-Client Accuracy by Phase", name, model))
        plt.legend(ncol=2)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outdir / f"clients_by_phase_{name}{SFX}.png", dpi=150)
        plt.close()


# =====================================================================
# ADDITIONAL: Generate confusion matrices for all phase combinations
# =====================================================================
def generate_all_phase_confusion_matrices():
    """Generate confusion matrices for default and tuned phases."""
    print("\n" + "=" * 70)
    print("GENERATING CONFUSION MATRICES FOR ALL PHASES")
    print("=" * 70 + "\n")

    configs = [
        ("cnn", "frozen", "default"),
        ("cnn", "frozen", "tuned"),
        ("cnn", "non_frozen", "default"),
        ("cnn", "non_frozen", "tuned"),
        ("lstm", "frozen", "default"),
        ("lstm", "frozen", "tuned"),
        ("lstm", "non_frozen", "default"),
        ("lstm", "non_frozen", "tuned"),
    ]

    generated = 0
    missing = []

    for model, freeze, phase in configs:
        # Look for CM CSV file with phase suffix
        cm_csv = RESULTS_DIR / f"{model}_{freeze}_run_cm_{phase}.csv"

        if not cm_csv.exists():
            missing.append(str(cm_csv.name))
            continue

        # Load and process
        df = pd.read_csv(cm_csv)
        df.columns = df.columns.str.lower()

        if not {"round", "true", "pred", "count"}.issubset(set(df.columns)):
            print(f"✗ Skipping {cm_csv.name} - missing required columns")
            continue

        # Get last round
        last_round = int(df["round"].max())
        df_round = df[df["round"] == last_round]

        # Build confusion matrix
        n = N_CLASSES
        cm = np.zeros((n, n), dtype=np.int64)
        for _, row in df_round.iterrows():
            i, j = int(row["true"]), int(row["pred"])
            cm[i, j] = int(row["count"])

        # Normalize to percentages
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = np.nan_to_num(cm / cm.sum(axis=1, keepdims=True)) * 100

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm_norm, aspect="auto", cmap="Blues", vmin=0, vmax=100)

        # Add text annotations
        for i in range(n):
            for j in range(n):
                val = cm_norm[i, j]
                text_color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center",
                        color=text_color, fontsize=12, weight="bold")

        # Styling
        model_name = PRETTY_MODEL.get(model, model.upper())
        freeze_label = PRETTY_RUN.get(freeze, freeze.title())
        phase_label = PRETTY_PHASE.get(phase, phase.title())

        ax.set_title(
            f"Confusion Matrix - Prediction Percentages\n{model_name} - {freeze_label} - {phase_label} (Round {last_round})",
            fontsize=16, pad=20, weight="bold"
        )
        ax.set_xlabel("Predicted Labels", fontsize=14, labelpad=10)
        ax.set_ylabel("True Labels", fontsize=14, labelpad=10)

        # Set ticks
        ax.set_xticks(range(n))
        ax.set_xticklabels(SUPERCLASSES, fontsize=12)
        ax.set_yticks(range(n))
        ax.set_yticklabels(SUPERCLASSES, fontsize=12)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("%", fontsize=14, rotation=0, labelpad=20)
        cbar.ax.tick_params(labelsize=11)

        # Save to model-specific directory
        outdir = _out_dir_for(model)
        filename = f"confusion_{freeze}_{phase}_r{last_round:02d}.png"
        save_path = outdir / filename

        plt.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"✓ Saved: {save_path}")
        generated += 1

    print(f"\n✓ Generated {generated} confusion matrices across all phases")

    if missing:
        print(f"\n⚠  Missing {len(missing)} CM CSV files:")
        for m in missing:
            print(f"  - {m}")

    print("=" * 70 + "\n")


# Call the additional function
generate_all_phase_confusion_matrices()

print("Saved plots to:")
for d in sorted(_saved_dirs):
    print(f" - {d}")