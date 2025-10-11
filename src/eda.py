# from __future__ import annotations
# import pandas as pd
# import numpy as np
# from rich import print
# 
# from src.config import SAMPLE_RATE
# from src.data_loader import load_metadata, map_superclasses, filter_single_label, load_waveform
# from src.utils import plot_signal, summarize_dataset
# 
# 
# if __name__ == "__main__":
#     ptb = load_metadata()
#     df = map_superclasses(ptb)
#     one = filter_single_label(df)
# 
#     summarize_dataset(one, sample_rate=SAMPLE_RATE, title="PTB-XL (Filtered Single-Label)")
# 
# 
#     print(f"Total records: {len(df):,}")
#     print(f"Single‑label records: {len(one):,}")
# 
# 
#     # Inspect distributions
#     print(one["y"].value_counts())
#     by_patient = one.groupby("patient_id").size()
#     print("Records per patient — median:", by_patient.median())
# 
# 
#     # Age/Sex distribution (age 300 is 90+ per PTB‑XL privacy; treat specially in analysis)
#     ages = one["age"].replace({300: np.nan})
#     print("Age — mean (excluding 300):", ages.mean())
#     print(one["sex"].value_counts())
# 
# """
#     # Plot a few example signals
#     for i, (_, row) in enumerate(one.sample(3, random_state=7).iterrows()):
#         sig = load_waveform(row, sampling_rate=SAMPLE_RATE)
#         plot_signal(sig, title=f"ecg_id={row.ecg_id} class={row.y}", save=f"example_{i+1}.png")
#         print("Saved example plots to results/.")"""
#         
# 



# src/eda.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Project imports (robust fallbacks) ──────────────────────────────────────
try:
    from src.config import RESULTS_DIR, SAMPLE_RATE, SUPERCLASSES
except Exception:
    RESULTS_DIR = Path("results")
    SAMPLE_RATE = 100
    SUPERCLASSES = ["NORM", "MI", "CD", "STTC", "HYP"]

try:
    from src.data_loader import (
        load_metadata,
        map_superclasses,
        filter_single_label,
        load_waveform,        # optional
        make_feature_table,   # optional
    )
except Exception:
    # Minimal placeholders so the file can still import
    def load_metadata():
        raise RuntimeError("src.data_loader.load_metadata not found")
    def map_superclasses(df):
        raise RuntimeError("src.data_loader.map_superclasses not found")
    def filter_single_label(df):
        return df
    def load_waveform(*_, **__):
        raise RuntimeError("src.data_loader.load_waveform not found")
    def make_feature_table(*_, **__):
        raise RuntimeError("src.data_loader.make_feature_table not found")

try:
    from src.utils import ensure_dir
except Exception:
    def ensure_dir(p: Path) -> None:
        Path(p).mkdir(parents=True, exist_ok=True)

# ── Paths ──────────────────────────────────────────────────────────────────
FIG_DIR = RESULTS_DIR / "figs"
EDA_DIR = RESULTS_DIR / "eda"
ensure_dir(FIG_DIR)
ensure_dir(EDA_DIR)

# ── Small helpers ──────────────────────────────────────────────────────────
def _sex_normalize(s: pd.Series) -> pd.Series:
    out = pd.Series("unknown", index=s.index, dtype="string")
    s_num = pd.to_numeric(s, errors="coerce")
    out.loc[s_num == 1] = "male"
    out.loc[s_num == 0] = "female"
    st = s.astype("string").str.strip().str.lower()
    out.loc[st.str.startswith("m", na=False)] = "male"
    out.loc[st.str.startswith("f", na=False)] = "female"
    out.loc[st.isin({"male", "man"})] = "male"
    out.loc[st.isin({"female", "woman"})] = "female"
    return out

def _fd_bins(x: np.ndarray, min_bins: int = 24, max_bins: int = 64) -> int:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return min_bins
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    if iqr <= 0:
        return min(max_bins, max(min_bins, int(np.sqrt(x.size))))
    bw = 2 * iqr * (x.size ** (-1 / 3))
    if bw <= 0:
        return min(max_bins, max(min_bins, int(np.sqrt(x.size))))
    bins = int(np.ceil((x.max() - x.min()) / bw))
    return int(np.clip(bins, min_bins, max_bins))

def _savefig(fig: plt.Figure, name: str) -> Path:
    fp = FIG_DIR / name
    fig.savefig(fp, dpi=300, bbox_inches="tight")
    print(f"[EDA] Saved: {fp.resolve()}")
    return fp

# ── Data container ─────────────────────────────────────────────────────────
@dataclass
class EDAInputs:
    df: pd.DataFrame
    classes: Sequence[str]

def prepare_df() -> EDAInputs:
    """Load PTB-XL meta, map to superclasses, keep single-label rows; add light cleanups."""
    ptb = load_metadata()              # your loader
    df = map_superclasses(ptb)         # must add 'y' column (superclass)
    df = filter_single_label(df).copy()

    if "sex" in df.columns and "sex_norm" not in df.columns:
        df["sex_norm"] = _sex_normalize(df["sex"])
    if "recording_date" in df.columns:
        df["recording_date"] = pd.to_datetime(df["recording_date"], errors="coerce")
        df["year"] = df["recording_date"].dt.year

    classes = [c for c in SUPERCLASSES if c in set(df["y"].astype(str))] or \
              sorted(df["y"].astype(str).unique())
    return EDAInputs(df=df, classes=classes)

# ── Plots ──────────────────────────────────────────────────────────────────
def plot_class_counts(inp: EDAInputs) -> None:
    vc = inp.df["y"].astype(str).value_counts().reindex(inp.classes).fillna(0).astype(int)
    vc.rename("count").to_frame().to_csv(EDA_DIR / "class_counts.csv")
    xs = np.arange(len(vc))
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.bar(xs, vc.values, width=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(inp.classes)
    ax.set_title("Class counts (superclasses)"); ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    ymax = float(vc.max()) if len(vc) else 1.0
    for x, y in zip(xs, vc.values):
        ax.text(x, y + max(1, 0.02 * ymax), f"{int(y):,}", ha="center", va="bottom", fontsize=9)
    _savefig(fig, "eda_class_counts.png"); plt.close(fig)

def plot_sex_overall(inp: EDAInputs) -> None:
    if "sex_norm" not in inp.df.columns:
        print("[EDA] sex column not available; skipping sex plots.")
        return
    vc = inp.df["sex_norm"].value_counts().reindex(["female", "male", "unknown"]).dropna()
    vc.rename("count").to_frame().to_csv(EDA_DIR / "sex_distribution.csv")
    labels, vals = vc.index.tolist(), vc.values.astype(int)
    k = max(1, len(labels)); group_w = 0.80; bar_w = group_w / k
    xs = [(-group_w / 2) + j * bar_w + bar_w / 2 for j in range(k)]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.bar(xs, vals, width=bar_w * 0.9)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_title("Sex distribution (overall)"); ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)
    ymax = float(vals.max()) if vals.size else 1.0
    for x, y in zip(xs, vals):
        ax.text(x, y + max(1, 0.02 * ymax), f"{int(y):,}", ha="center", va="bottom", fontsize=9)
    _savefig(fig, "eda_sex_overall.png"); plt.close(fig)

def plot_class_by_sex(inp: EDAInputs) -> None:
    if "sex_norm" not in inp.df.columns:
        return
    ct = pd.crosstab(inp.df["y"].astype(str), inp.df["sex_norm"]) \
           .reindex(inp.classes).fillna(0).astype(int)
    sex_order = [s for s in ["female", "male", "unknown"] if s in ct.columns] or list(ct.columns)

    # counts
    xs = np.arange(len(ct.index)); K = len(sex_order); group_w = 0.84; bar_w = group_w / max(1, K)
    fig, ax = plt.subplots(figsize=(10.8, 5.6)); ymax = 0
    for j, s in enumerate(sex_order):
        offs = -group_w / 2 + j * bar_w + bar_w / 2
        vals = ct[s].values; ymax = max(ymax, float(vals.max() if vals.size else 0.0))
        ax.bar(xs + offs, vals, width=bar_w * 0.88, label=s)
    ax.set_title("Class × Sex (counts)"); ax.set_ylabel("Count")
    ax.set_xticks(xs); ax.set_xticklabels(inp.classes)
    ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1.0)
    ax.legend(frameon=False, ncol=min(3, K), loc="upper center", bbox_to_anchor=(0.5, 1.10))
    _savefig(fig, "eda_class_by_sex_counts.png"); plt.close(fig)

    # normalized %
    pct = (ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100).fillna(0)
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    for j, s in enumerate(sex_order):
        offs = -group_w / 2 + j * bar_w + bar_w / 2
        ax.bar(xs + offs, pct[s].values, width=bar_w * 0.88, label=s)
    ax.set_title("Class × Sex (normalized)"); ax.set_ylabel("Percentage (%)")
    ax.set_xticks(xs); ax.set_xticklabels(inp.classes)
    ax.set_ylim(0, 100); ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=min(3, K), loc="upper center", bbox_to_anchor=(0.5, 1.10))
    _savefig(fig, "eda_class_by_sex_percent.png"); plt.close(fig)

def plot_age(inp: EDAInputs) -> None:
    if "age" not in inp.df.columns:
        print("[EDA] age not found; skipping age plots.")
        return

    # PTB-XL: 300 = “90+”; exclude from mean/median calc
    a_all = pd.to_numeric(inp.df["age"], errors="coerce").replace({300: np.nan}).dropna().values
    if a_all.size:
        bins = _fd_bins(a_all, 24, 64)
        fig, ax = plt.subplots(figsize=(11.2, 5.2))
        ax.hist(a_all, bins=bins, rwidth=0.94, edgecolor="white", linewidth=0.6, alpha=0.9)
        mu, med = float(np.mean(a_all)), float(np.median(a_all))
        ax.axvline(mu, linestyle="--", linewidth=1.4, color="k", label=f"Mean {mu:.1f}")
        ax.axvline(med, linestyle=":",  linewidth=1.6, color="k", label=f"Median {med:.1f}")
        ax.set_title("Age distribution"); ax.set_xlabel("Age"); ax.set_ylabel("Count")
        ax.grid(alpha=0.3, axis="y"); ax.legend(frameon=False, ncol=2, loc="upper center")
        _savefig(fig, "eda_age_hist_overall.png"); plt.close(fig)

    # by class
    data_c, labels_c = [], []
    for c in inp.classes:
        v = pd.to_numeric(inp.df.loc[inp.df["y"].astype(str) == c, "age"], errors="coerce") \
                .replace({300: np.nan}).dropna().values
        if v.size >= 5:
            data_c.append(v); labels_c.append(c)
    if data_c:
        fig, ax = plt.subplots(figsize=(10.4, 5.0))
        ax.boxplot(data_c, labels=labels_c, showfliers=False)
        ax.set_title("Age by diagnostic class"); ax.set_ylabel("Age")
        ax.grid(alpha=0.3, axis="y")
        _savefig(fig, "eda_age_by_class_box.png"); plt.close(fig)

    # by sex
    if "sex_norm" in inp.df.columns:
        af = pd.to_numeric(inp.df.loc[inp.df["sex_norm"] == "female", "age"], errors="coerce") \
                .replace({300: np.nan}).dropna().values
        am = pd.to_numeric(inp.df.loc[inp.df["sex_norm"] == "male", "age"], errors="coerce") \
                .replace({300: np.nan}).dropna().values
        base = a_all if a_all.size else (np.concatenate([af, am]) if (af.size and am.size) else (af if af.size else am))
        if base.size:
            bins = _fd_bins(base, 24, 64)
            fig, ax = plt.subplots(figsize=(11.2, 5.2))
            if af.size:
                ax.hist(af, bins=bins, rwidth=0.90, alpha=0.55, label="female",
                        edgecolor="white", linewidth=0.6)
            if am.size:
                ax.hist(am, bins=bins, rwidth=0.90, alpha=0.55, label="male",
                        edgecolor="white", linewidth=0.6)
            ax.set_title("Age distribution by sex (overlay)")
            ax.set_xlabel("Age"); ax.set_ylabel("Count")
            ax.grid(alpha=0.3, axis="y"); ax.legend(frameon=False, ncol=2, loc="upper center")
            _savefig(fig, "eda_age_hist_by_sex_overlay.png"); plt.close(fig)

            if af.size and am.size:
                fig, ax = plt.subplots(figsize=(8.0, 4.8))
                bp = ax.boxplot([af, am], labels=["female", "male"], showfliers=False, patch_artist=True)
                for i, c in enumerate(["#E66BB5", "#4A90E2"]):
                    bp["boxes"][i].set_facecolor(c); bp["boxes"][i].set_alpha(0.25); bp["boxes"][i].set_edgecolor(c)
                    bp["medians"][i].set_color(c)
                ax.set_title("Age by sex"); ax.set_ylabel("Age"); ax.grid(alpha=0.3, axis="y")
                _savefig(fig, "eda_age_by_sex_box.png"); plt.close(fig)

def plot_year_counts(inp: EDAInputs) -> None:
    if "year" not in inp.df.columns:
        return
    yrs = inp.df["year"].dropna().astype(int).value_counts().sort_index()
    if not len(yrs):
        return
    xs = np.arange(len(yrs))
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.bar(xs, yrs.values, width=0.72, edgecolor="white", linewidth=0.6)
    ax.set_xticks(xs); ax.set_xticklabels(yrs.index.astype(str), rotation=45, ha="right")
    ax.set_title("Records per year"); ax.set_ylabel("Count"); ax.grid(alpha=0.3, axis="y")
    ymax = float(yrs.max()); ax.set_ylim(0, ymax * 1.14)
    for x, y in zip(xs, yrs.values):
        ax.text(x, y + max(1, 0.02 * ymax), f"{int(y):,}", ha="center", va="bottom", fontsize=9)
    _savefig(fig, "eda_records_per_year.png"); plt.close(fig)

def plot_fold_class_counts(inp: EDAInputs) -> None:
    if "strat_fold" not in inp.df.columns:
        return
    ct = pd.crosstab(inp.df["strat_fold"].astype(int), inp.df["y"].astype(str)) \
           .reindex(columns=inp.classes).fillna(0).astype(int)
    folds = ct.index.astype(str).tolist()
    xs = np.arange(len(folds)); K = len(ct.columns); group_w = 0.84; bar_w = group_w / max(1, K)
    fig, ax = plt.subplots(figsize=(12.0, 5.8)); ymax = 0.0
    for j, c in enumerate(ct.columns):
        offs = -group_w / 2 + j * bar_w + bar_w / 2
        vals = ct[c].values; ymax = max(ymax, float(vals.max()) if vals.size else 0.0)
        ax.bar(xs + offs, vals, width=bar_w * 0.88, label=str(c))
    ax.set_title("Class counts per strat_fold (grouped)")
    ax.set_xlabel("strat_fold"); ax.set_ylabel("Count")
    ax.set_xticks(xs); ax.set_xticklabels(folds)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(ncol=min(5, K), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    ax.set_ylim(0, ymax * 1.25 if ymax > 0 else 1.0)
    _savefig(fig, "eda_fold_class_counts_grouped.png"); plt.close(fig)

def plot_feature_correlation_if_available(inp: EDAInputs) -> None:
    """Optional: build numeric feature table and plot |r| heatmap (skipped if helper missing)."""
    try:
        X, y, _ = make_feature_table(inp.df)     # your helper returns (X, y, classes)
        num = pd.DataFrame(X).select_dtypes(include=[np.number])
        if num.shape[1] < 2:
            print("[EDA] Not enough numeric features for correlation plot.")
            return
        corr = num.corr().abs()
        fig, ax = plt.subplots(figsize=(9.5, 7.8))
        im = ax.imshow(corr.values, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        cb = fig.colorbar(im, ax=ax); cb.set_label("|r|")
        ax.set_title("Feature correlation heatmap (|r|)")
        ax.set_xticks([]); ax.set_yticks([])
        _savefig(fig, "eda_feature_corr_heatmap.png"); plt.close(fig)
    except Exception as e:
        print(f"[EDA] Feature-table correlation skipped: {e}")

def plot_example_waveforms(inp: EDAInputs, n: int = 3) -> None:
    """Optional: save a few single-lead examples (requires load_waveform)."""
    try:
        sample = inp.df.sample(min(n, len(inp.df)), random_state=7)
        for i, (_, row) in enumerate(sample.iterrows(), 1):
            sig = load_waveform(row, sampling_rate=SAMPLE_RATE)  # expected shape (12, T)
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(sig[0], linewidth=0.8)
            ax.set_title(f"ECG example — id={row.get('ecg_id', 'NA')} class={row['y']}")
            ax.set_xlabel("Time (samples)"); ax.set_ylabel("mV"); ax.grid(alpha=0.3)
            _savefig(fig, f"eda_example_{i}.png"); plt.close(fig)
    except Exception as e:
        print(f"[EDA] Example waveform plots skipped: {e}")

# ── Orchestration ──────────────────────────────────────────────────────────
def run_all() -> None:
    inp = prepare_df()

    # Core tables/figures
    plot_class_counts(inp)
    plot_sex_overall(inp)
    plot_class_by_sex(inp)
    plot_age(inp)
    plot_year_counts(inp)
    plot_fold_class_counts(inp)

    # Optional extras
    plot_feature_correlation_if_available(inp)
    plot_example_waveforms(inp, n=3)

    print(f"\n[EDA] Done. Figures → {FIG_DIR.resolve()} | CSVs → {EDA_DIR.resolve()}")

# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all()
