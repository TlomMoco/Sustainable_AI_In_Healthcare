"""
data_loader.py — PTB-XL Dataset Utilities
-----------------------------------------

Handles all data operations for the Sustainable AI in Healthcare project:
  • Load and parse the PTB-XL metadata (CSV files)
  • Map diagnostic SCP codes to 5 diagnostic superclasses
  • Patient-aware train/val/test splits (no leakage)
  • ECG waveform loading via WFDB
  • Per-lead normalization (z-score)
  • Feature extraction for baseline ML models
  • ANOVA artifacts + lead selection (threshold or top-K)

All functions are deterministic and use only patient IDs for stratification.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import re

import numpy as np
import pandas as pd
import wfdb
import ast

from src.config import SAMPLE_RATE as _FS
from src.config import (
    PTBXL_CSV, SCP_CSV, DATA_ROOT, SAMPLE_RATE, SUPERCLASSES
)

# -------------------------------------------------------------------------
# Data structures
# -------------------------------------------------------------------------
@dataclass
class PTBXL:
    """Container for PTB-XL metadata and SCP aggregation table."""
    df: pd.DataFrame
    agg: pd.DataFrame


# -------------------------------------------------------------------------
# Metadata cleaning
# -------------------------------------------------------------------------
def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Dtypes
    for col in ["age", "patient_id", "ecg_id"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Age cleaning
    if "age" in out.columns:
        out.loc[out["age"] == 300, "age"] = np.nan
        out.loc[(out["age"] < 0) | (out["age"] > 120), "age"] = np.nan

    # Sex normalization
    if "sex" in out.columns:
        s = out["sex"].astype("string")
        s = s.str.strip()
        s = s.replace({"0": "Female", "1": "Male", "2": "Unknown"})
        txt = s.str.lower()
        txt_map = {
            "female": "Female", "f": "Female", "woman": "Female",
            "male": "Male", "m": "Male", "man": "Male",
        }
        s2 = txt.map(txt_map)
        out["sex"] = s2.fillna(s).fillna("Unknown")

    # Ensure waveform paths exist
    need_cols = ["filename_lr", "filename_hr"]
    have = [c for c in need_cols if c in out.columns]
    if have:
        mask_paths = out[have].notna().any(axis=1)
        out = out[mask_paths].copy()

    # Drop exact duplicated ecg_id rows
    if "ecg_id" in out.columns:
        out = out.drop_duplicates(subset=["ecg_id"])

    return out


# -------------------------------------------------------------------------
# Metadata loading
# -------------------------------------------------------------------------
def load_metadata() -> PTBXL:
    df = pd.read_csv(PTBXL_CSV)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    df = clean_metadata(df)

    agg = pd.read_csv(SCP_CSV, index_col=0)
    agg = agg[agg["diagnostic"] == 1][["diagnostic_class"]]
    return PTBXL(df=df, agg=agg)


# -------------------------------------------------------------------------
# Label mapping and filtering
# -------------------------------------------------------------------------
def map_superclasses(ptb: PTBXL) -> pd.DataFrame:
    df = ptb.df.copy()

    def to_superclasses(code_dict: Dict[str, float]) -> List[str]:
        labels = set()
        for code in code_dict.keys():
            if code in ptb.agg.index:
                labels.add(ptb.agg.loc[code, "diagnostic_class"])
        return sorted(labels)

    df["diagnostic_superclasses"] = df["scp_codes"].apply(to_superclasses)
    return df


def filter_single_label(df: pd.DataFrame) -> pd.DataFrame:
    one = df[df["diagnostic_superclasses"].apply(lambda x: len(x) == 1)].copy()
    one["y"] = one["diagnostic_superclasses"].str[0]
    one = one[one["y"].isin(SUPERCLASSES)]
    return one


# -------------------------------------------------------------------------
# Patient-aware splitting
# -------------------------------------------------------------------------
def stratified_patient_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    patient_labels = df.groupby("patient_id")["y"].agg(lambda s: s.value_counts().idxmax())
    patients = patient_labels.index.to_numpy()

    test_patients: List[int] = []
    for c in patient_labels.unique():
        p_cls = patients[patient_labels.values == c]
        rng.shuffle(p_cls)
        n_test = max(1, int(len(p_cls) * test_size))
        test_patients.extend(list(p_cls[:n_test]))

    train = df[~df["patient_id"].isin(test_patients)].copy()
    test = df[df["patient_id"].isin(test_patients)].copy()
    return train, test


def stratified_patient_split_3way(
    df: pd.DataFrame, splits: Tuple[float, float, float] = (0.70, 0.15, 0.15), seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    train_frac, val_frac, test_frac = splits

    train_df, temp_df = stratified_patient_split(df, test_size=(1.0 - train_frac), seed=seed)

    if len(temp_df) == 0:
        raise ValueError("Temp split is empty; dataset too small for requested ratios.")

    val_prop_within_temp = val_frac / (val_frac + test_frac)
    val_df, test_df = stratified_patient_split(temp_df, test_size=(1.0 - val_prop_within_temp), seed=seed + 1)
    return train_df, val_df, test_df


# -------------------------------------------------------------------------
# Waveform loading and preprocessing
# -------------------------------------------------------------------------
def load_waveform(row: pd.Series, sampling_rate: int = SAMPLE_RATE) -> np.ndarray:
    rel = row["filename_lr"] if sampling_rate == 100 else row["filename_hr"]
    rec_path = (DATA_ROOT / f"{rel}").as_posix()
    sig, _ = wfdb.rdsamp(rec_path)
    return np.asarray(sig).T  # (T, 12) → (12, T)


# -------------------------------------------------------------------------
# Features for ANOVA (and ML baselines)
# -------------------------------------------------------------------------
def basic_features(signal: np.ndarray) -> np.ndarray:
    means = signal.mean(axis=1)
    stds = signal.std(axis=1)
    rms = np.sqrt((signal ** 2).mean(axis=1))
    return np.concatenate([means, stds, rms], axis=0)


def basic_feature_names() -> List[str]:
    names: List[str] = []
    for j in range(12):
        names += [f"L{j+1}_mean", f"L{j+1}_std", f"L{j+1}_rms"]
    return names


def engineered_features(signal: np.ndarray) -> np.ndarray:
    X = np.asarray(signal, dtype=np.float32)
    assert X.ndim == 2 and X.shape[0] == 12, "engineered_features expects (12, T)"

    means = X.mean(axis=1)
    stds = X.std(axis=1)
    rms = np.sqrt((X ** 2).mean(axis=1))
    ptp = np.ptp(X, axis=1)
    stats_vec = np.concatenate([means, stds, rms, ptp], axis=0)  # (48,)

    def _bandpower(sig1d: np.ndarray, fs: float, f_lo: float, f_hi: float) -> float:
        x = np.asarray(sig1d, dtype=np.float32)
        n = int(x.size)
        if n == 0:
            return 0.0
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spec = np.abs(np.fft.rfft(x)) ** 2
        m = (freqs >= f_lo) & (freqs < f_hi)
        cnt = int(m.sum())
        return float(spec[m].sum() / max(1, cnt))

    bands = [(0.0, 5.0), (5.0, 15.0), (15.0, 40.0)]
    bp_list: List[float] = []
    for lead in range(12):
        s = X[lead]
        for (a, b) in bands:
            bp_list.append(_bandpower(s, float(_FS), a, b))
    bp_vec = np.array(bp_list, dtype=np.float32)  # (36,)

    best_lead = int(np.argmax(stds))
    s = X[best_lead].astype(np.float32, copy=False)
    y = s - np.median(s)
    y = y ** 2
    win = max(1, int(0.150 * _FS))
    env = np.convolve(y, np.ones(win, dtype=np.float32) / win, mode="same")
    thr = float(env.mean() + 1.0 * env.std())
    refractory = max(1, int(0.250 * _FS))
    cand = np.where((env[1:-1] > env[:-2]) & (env[1:-1] > env[2:]) & (env[1:-1] >= thr))[0] + 1
    peaks_list: List[int] = []
    last = -10 ** 9
    for p in cand:
        if (p - last) >= refractory:
            peaks_list.append(int(p))
            last = p
    peaks = np.array(peaks_list, dtype=int)
    if peaks.size >= 3:
        rr = np.diff(peaks) / float(_FS)
        hr_bpm = 60.0 / rr.mean()
        sdnn_ms = rr.std(ddof=1) * 1000.0
        rmssd_ms = np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000.0
    else:
        hr_bpm, sdnn_ms, rmssd_ms = np.nan, np.nan, np.nan
    hrv_vec = np.array([hr_bpm, sdnn_ms, rmssd_ms], dtype=np.float32)

    return np.concatenate([stats_vec, bp_vec, hrv_vec], axis=0)  # (87,)


def engineered_feature_names() -> List[str]:
    names: List[str] = []
    for j in range(12):
        names += [f"L{j+1}_mean", f"L{j+1}_std", f"L{j+1}_rms", f"L{j+1}_ptp"]
    for j in range(12):
        names += [f"L{j+1}_bp_0_5Hz", f"L{j+1}_bp_5_15Hz", f"L{j+1}_bp_15_40Hz"]
    names += ["HR_bpm", "SDNN_ms", "RMSSD_ms"]
    return names


def make_feature_table(
    df: pd.DataFrame,
    limit: Optional[int] = None,
    feature_set: str = "basic",   # "basic" or "engineered"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    sel_df = df if limit is None else df.iloc[:limit]
    feat_fn = engineered_features if feature_set == "engineered" else basic_features

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for _, row in sel_df.iterrows():
        sig = load_waveform(row)                   # (12, T)
        X_list.append(feat_fn(sig))                # 1D vector
        y_list.append(SUPERCLASSES.index(row["y"]))

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=int)
    return X, y, SUPERCLASSES


# -------------------------------------------------------------------------
# Normalization utilities
# -------------------------------------------------------------------------
def compute_perlead_norm_stats(df: pd.DataFrame, sampling_rate: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    means, stds = [], []
    for _, row in df.iterrows():
        sig = load_waveform(row, sampling_rate)
        means.append(sig.mean(axis=1))
        stds.append(sig.std(axis=1))
    mu = np.stack(means).mean(axis=0) if means else np.zeros(12)
    sigma = np.stack(stds).mean(axis=0) if stds else np.ones(12)
    sigma = np.maximum(sigma, 1e-6)
    return mu, sigma


def normalize_signal(sig: np.ndarray, mu: np.ndarray, sigma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return (sig - mu[:, None]) / (sigma[:, None] + eps)


# -------------------------------------------------------------------------
# ANOVA artifacts (top-K) — optional utility (still used by EDA)
# -------------------------------------------------------------------------
def export_selectkbest_artifacts(
    train_df: pd.DataFrame,
    outdir: Path,
    feature_set: str = "engineered",
    k: int = 50,
) -> Tuple[List[int], List[str]]:
    """
    Fits VarianceThreshold + SelectKBest(f_classif) on TRAIN features.
    Exports:
      - outdir/tables/table_anova_fscores_full.csv
      - outdir/tables/table_anova_fscores_topK.csv
    Returns:
      (top_indices_in_vt_space, top_feature_names)
    """
    outdir = Path(outdir)
    tables_dir = outdir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = make_feature_table(train_df, feature_set=feature_set)
    if feature_set == "engineered":
        all_names = engineered_feature_names()
    else:
        all_names = basic_feature_names()

    d = X.shape[1]
    if len(all_names) != d:
        all_names = [f"f{i}" for i in range(d)]

    try:
        from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
        from sklearn.impute import SimpleImputer

        X_num = np.asarray(X, dtype=np.float64)
        X_num[~np.isfinite(X_num)] = np.nan

        vt = VarianceThreshold(threshold=1e-12)
        X_vt = vt.fit_transform(X_num)

        # ---- explicit array & python list for strict type checkers ----
        cols_idx_arr: np.ndarray = np.array(vt.get_support(indices=True), dtype=int).reshape(-1)
        cols_idx_list: List[int] = [int(i) for i in cols_idx_arr]

        cols_vt = [all_names[i] for i in cols_idx_list]
        if X_vt.shape[1] == 0:
            raise RuntimeError("VarianceThreshold removed all features")

        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X_vt)

        k_eff = int(min(max(1, k), X_imp.shape[1]))
        selector = SelectKBest(score_func=f_classif, k=k_eff)
        selector.fit(X_imp, y)
        F = np.clip(np.asarray(selector.scores_, dtype=float), 0, None)
        p = np.asarray(selector.pvalues_, dtype=float)

        full = (
            pd.DataFrame({"feature": cols_vt, "F_score": F, "p_value": p})
            .sort_values("F_score", ascending=False)
            .reset_index(drop=True)
        )
        full.to_csv(tables_dir / "table_anova_fscores_full.csv", index=False)

        top = full.head(k_eff).copy()
        top.to_csv(tables_dir / "table_anova_fscores_topK.csv", index=False)

        top_features = top["feature"].astype(str).tolist()
        top_indices_list: List[int] = [cols_vt.index(f) for f in top_features]
        return top_indices_list, top_features

    except Exception as e:
        full = pd.DataFrame({
            "feature": all_names,
            "F_score": np.ones(d, dtype=float),
            "p_value": np.ones(d, dtype=float),
        })
        (outdir / "tables").mkdir(parents=True, exist_ok=True)
        full.to_csv(outdir / "tables" / "table_anova_fscores_full.csv", index=False)

        k_eff = int(min(max(1, k), d))
        top = full.head(k_eff)
        top.to_csv(outdir / "tables" / "table_anova_fscores_topK.csv", index=False)

        top_indices_list = list(range(k_eff))
        top_features = top["feature"].astype(str).tolist()
        print(f"(ANOVA fallback) {e} — wrote uniform weights; continuing.")
        return top_indices_list, top_features


# -------------------------------------------------------------------------
# NEW: ANOVA threshold → per-lead selection → lead mask for CNN/LSTM
# -------------------------------------------------------------------------
def _lead_index_from_feature_name(name: str) -> Optional[int]:
    """
    Parse names like 'L3_mean', 'L10_bp_5_15Hz' → 0-based lead index (2, 9).
    Returns None for non-lead features (e.g., HRV).
    """
    m = re.match(r"^L(\d+)_", str(name))
    if not m:
        return None
    li = int(m.group(1))
    if 1 <= li <= 12:
        return li - 1
    return None


def compute_anova_lead_mask_by_threshold(
    train_df: pd.DataFrame,
    outdir: Path,
    feature_set: str = "engineered",
    fscore_threshold: float = 200.0,
    fallback_k: int = 8,
) -> Tuple[np.ndarray, List[int], pd.DataFrame]:
    """
    1) Run VarianceThreshold + f_classif on TRAIN features (exports full table)
    2) Select features with F_score >= fscore_threshold
    3) Map selected features to leads; keep all leads that have >=1 selected feature
    4) If none selected, fall back to top `fallback_k` leads ranked by summed F_score (over ALL features)
    Returns:
        lead_mask: np.ndarray shape (12,) with 1.0 for kept leads else 0.0
        kept_leads: list of kept 0-based lead indices
        df_selected: DataFrame of selected features (thresholded) with F_scores
    """
    outdir = Path(outdir)
    tables_dir = outdir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Ensure full table exists
    export_selectkbest_artifacts(train_df, outdir, feature_set=feature_set, k=99999)

    full_csv = tables_dir / "table_anova_fscores_full.csv"
    if not full_csv.exists():
        return np.ones(12, dtype=np.float32), list(range(12)), pd.DataFrame()

    df_full = pd.read_csv(full_csv)
    if "feature" not in df_full.columns or "F_score" not in df_full.columns:
        return np.ones(12, dtype=np.float32), list(range(12)), pd.DataFrame()

    # Selected features above threshold
    df_sel = df_full[df_full["F_score"] >= float(fscore_threshold)].copy()

    # Aggregate to leads
    def _agg_to_leads(df_in: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        sumF = np.zeros(12, dtype=float)
        cnt = np.zeros(12, dtype=int)
        for _, r in df_in.iterrows():
            li = _lead_index_from_feature_name(str(r.at["feature"]))
            if li is None:
                continue
            fs = float(r.at["F_score"]) if pd.notna(r.at["F_score"]) else 0.0
            sumF[li] += fs
            cnt[li] += 1
        return sumF, cnt

    sumF_all, cnt_all = _agg_to_leads(df_full)
    sumF_sel, cnt_sel = _agg_to_leads(df_sel)

    # Leads that have >=1 feature above the threshold
    kept = [i for i in range(12) if cnt_sel[i] > 0]

    # Enforce a minimum number of leads (fallback_k) by topping up
    min_leads = max(1, min(12, int(fallback_k)))
    if len(kept) < min_leads:
        # Rank all leads by total F (over full table), highest first
        order = list(np.argsort(-sumF_all))
        for li in order:
            if li not in kept:
                kept.append(int(li))
            if len(kept) >= min_leads:
                break

    # Build mask
    lead_mask = np.zeros(12, dtype=np.float32)
    for i in kept:
        lead_mask[int(i)] = 1.0

    # Summary table
    sel_out = pd.DataFrame({
        "lead": [i + 1 for i in range(12)],
        "kept": [int(i in kept) for i in range(12)],
        "sumF_selected": sumF_sel.tolist(),
        "cnt_selected": cnt_sel.tolist(),
        "sumF_all": sumF_all.tolist(),
        "cnt_all": cnt_all.tolist(),
        "threshold": [float(fscore_threshold)] * 12,
    })
    sel_out.to_csv(tables_dir / "table_anova_selected_leads_threshold.csv", index=False)

    return lead_mask, kept, df_sel