from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif, VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from src.config import RESULTS_DIR, SAMPLE_RATE, SUPERCLASSES, SEED
from src.data_loader import (
    load_metadata, map_superclasses, filter_single_label,
    make_feature_table, stratified_patient_split
)
from src.utils import summarize_dataset

# --- output folders --------------------------------------------------------
EDA_OUT = RESULTS_DIR / "viz" / "EDA"
EDA_OUT.mkdir(parents=True, exist_ok=True)

def _savefig(name: str):
    plt.tight_layout()
    plt.savefig(EDA_OUT / name, dpi=150)
    plt.close()

def _build_or_load_features() -> pd.DataFrame:
    """
    Load cached features.csv if present; else build inline from the train split.
    Prefers engineered features if your make_feature_table supports it.
    Returns a DataFrame with numeric columns + 'label'.
    """
    feat_csv = RESULTS_DIR / "features.csv"
    if feat_csv.exists():
        try:
            df = pd.read_csv(feat_csv)
            if "label" in df.columns:
                return df
        except Exception:
            pass

    # build inline from metadata
    ptb = load_metadata()                  # cleaning (age==300->NaN, etc.) happens inside
    df_meta = filter_single_label(map_superclasses(ptb))
    # use the same helper your pipeline uses to materialize features
    try:
        train_df, _ = stratified_patient_split(df_meta, test_size=0.2, seed=SEED)
        X, y, _ = make_feature_table(train_df, feature_set="engineered")
    except TypeError:
        train_df, _ = stratified_patient_split(df_meta, test_size=0.2, seed=SEED)
        X, y, _ = make_feature_table(train_df)

    # feature names (if you defined them; otherwise f0..fN-1)
    try:
        from src.data_loader import engineered_feature_names
        names = engineered_feature_names()
        if len(names) != X.shape[1]:
            names = [f"f{i}" for i in range(X.shape[1])]
    except Exception:
        names = [f"f{i}" for i in range(X.shape[1])]

    feat = pd.DataFrame(X, columns=names)
    feat["label"] = y.astype(int)

    # cache for reuse
    try:
        feat.to_csv(feat_csv, index=False)
        print(f"[EDA] cached features -> {feat_csv} shape={feat.shape}")
    except Exception:
        pass
    return feat

# ------------------------------------------------------------------------------
# ANOVA (label-feature correlation)
# ------------------------------------------------------------------------------
def run_anova_selectkbest_to_eda(top_k: int = 25):
    """
    Produces in results/viz/EDA/:
      - anova_fscores_full.csv     (all features after VT with F-score & p-value)
      - anova_fscores_topK.csv     (top-K ranking)
      - anova_topK_by_class_means.csv (top-K features' class-wise means)
      - anova_selectkbest_topK.png (bar chart of F-scores)
    """
    feat = _build_or_load_features()
    if feat.empty or "label" not in feat.columns:
        print("[EDA/ANOVA] no features available — skipped.")
        return

    # numeric X + y
    X = (feat.drop(columns=["label"], errors="ignore")
            .select_dtypes(include=[np.number])
            .replace([np.inf, -np.inf], np.nan))
    if X.shape[1] == 0:
        print("[EDA/ANOVA] no numeric features — skipped.")
        return
    y = feat["label"].astype(str).values

    # drop constant columns
    vt = VarianceThreshold(threshold=1e-12)
    try:
        X_vt = vt.fit_transform(X)
    except Exception:
        print("[EDA/ANOVA] variance thresholding failed — skipped.")
        return
    cols_vt = X.columns[vt.get_support()]
    if X_vt.shape[1] == 0:
        print("[EDA/ANOVA] all features constant — skipped.")
        return

    # impute; label-encode
    X_imp = SimpleImputer(strategy="median").fit_transform(X_vt)
    y_enc = LabelEncoder().fit_transform(y)

    # ANOVA F-test (univariate)
    F, p = f_classif(X_imp, y_enc)
    F = np.clip(F, 0, None)
    full = pd.DataFrame({"feature": cols_vt, "F_score": F, "p_value": p})
    full = full.sort_values("F_score", ascending=False).reset_index(drop=True)

    # save full + top-K tables
    EDA_OUT.mkdir(parents=True, exist_ok=True)
    full_path = EDA_OUT / "anova_fscores_full.csv"
    full.to_csv(full_path, index=False)
    top = full.head(int(min(top_k, len(full)))).copy()
    top_path = EDA_OUT / "anova_fscores_topK.csv"
    top.to_csv(top_path, index=False)

    # class-wise means for the top-K features (using imputed X for fairness)
    X_imp_df = pd.DataFrame(X_imp, columns=cols_vt)
    df_for_means = pd.concat([pd.Series(y, name="label_str"), X_imp_df], axis=1)
    class_means = (
        df_for_means.groupby("label_str")[top["feature"].tolist()]
        .mean()
        .reset_index()
        .rename(columns={"label_str": "class"})
    )
    class_means_path = EDA_OUT / "anova_topK_by_class_means.csv"
    class_means.to_csv(class_means_path, index=False)

    # bar plot of top-K F-scores
    plt.figure(figsize=(float(max(8.0, min(18.0, 0.35 * len(top)))), 4.0))
    plt.bar(top["feature"], top["F_score"])
    plt.ylabel("F-score (ANOVA)")
    plt.title(f"ANOVA F-scores (top {len(top)})")
    plt.xticks(rotation=65, ha="right")
    plt.tight_layout()
    png_path = EDA_OUT / "anova_selectkbest_topK.png"
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"[EDA/ANOVA] saved:\n  - {full_path}\n  - {top_path}\n  - {class_means_path}\n  - {png_path}")


# ------------------------------------------------------------------------------
# Main EDA routine
# ------------------------------------------------------------------------------
def main():
    # ---------- metadata EDA ----------
    ptb = load_metadata()                      # centralized cleaning
    df  = map_superclasses(ptb)
    one = filter_single_label(df)

    # summary to console
    summarize_dataset(one, sample_rate=SAMPLE_RATE, title="PTB-XL (Single-label, cleaned)")

    # class distribution
    ax = one["y"].value_counts().reindex(SUPERCLASSES).plot(kind="bar")
    ax.set_title("Class distribution (single-label)")
    ax.set_ylabel("# records")
    _savefig("class_distribution.png")


    # sex distribution (robust re-normalization just for plotting)
    sex_order = ["Male", "Female", "Unknown"]

    s = one.get("sex")
    if s is None:
        print("[EDA] 'sex' column missing — skipping sex distribution.")
    else:
        s = s.astype("string")  # keep <NA>
        s = s.str.strip()

        # numeric fallbacks
        s = s.replace({"0": "Female", "1": "Male", "2": "Unknown"})

        # textual normalization
        txt = s.str.lower()
        txt_map = {
            "female": "Female", "f": "Female", "woman": "Female",
            "male": "Male", "m": "Male", "man": "Male",
        }
        s_norm = txt.map(txt_map).fillna(s).fillna("Unknown")

        # plot with fixed order so bars always appear
        counts = pd.Series(pd.Categorical(s_norm, categories=sex_order)).value_counts().reindex(sex_order, fill_value=0)

        ax = counts.plot(kind="bar")
        ax.set_title("Sex distribution")
        ax.set_ylabel("# records")
        plt.xticks(rotation=0)
        _savefig("sex_distribution.png")

    # age by class
    plt.figure()
    ages_num = pd.to_numeric(one["age"], errors="coerce")
    data = [ages_num[one["y"] == c].dropna().to_numpy() for c in SUPERCLASSES]
    plt.boxplot(data, labels=SUPERCLASSES, showfliers=False)
    plt.title("Age by diagnostic superclass")
    plt.ylabel("Age (years)")
    _savefig("age_by_class_box.png")

    # records per patient
    rpp = one.groupby("patient_id").size()
    plt.figure()
    rpp.plot(kind="hist", bins=30)
    plt.title("Records per patient")
    plt.xlabel("# records")
    _savefig("records_per_patient.png")

    # missingness for key columns
    cols = [c for c in ["age", "sex", "filename_lr", "filename_hr"] if c in one.columns]
    miss = one[cols].isna().mean().rename("missing_frac").to_frame()
    miss.to_csv(EDA_OUT / "missingness.csv", index=True)

    # ---------- feature correlation heatmap ----------
    # build/load features (engineered preferred)
    feat = _build_or_load_features()
    if not feat.empty:
        X = (feat.drop(columns=["label"], errors="ignore")
                .select_dtypes(include=[np.number])
                .replace([np.inf, -np.inf], np.nan))
        # keep columns that have at least some non-NaN values
        X = X.loc[:, X.notna().any(axis=0)].copy()
        if X.shape[1] >= 2:
            # choose top-N by variance to keep heatmap readable
            var = X.var().sort_values(ascending=False)
            topN = min(30, len(var))
            cols = list(var.index[:topN])
            C = X[cols].corr(method="pearson").to_numpy()

            plt.figure(figsize=(max(8.0, 0.4 * topN), max(6.0, 0.4 * topN)))
            im = plt.imshow(C, vmin=-1, vmax=1, aspect="auto")
            plt.title(f"Feature Correlation (top {topN} by variance)")
            plt.xticks(range(topN), cols, rotation=65, ha="right")
            plt.yticks(range(topN), cols)
            plt.colorbar(im, label="Pearson r")
            _savefig("feature_corr_topN.png")
        else:
            print("[EDA] not enough numeric features for correlation heatmap.")

    # ---------- ANOVA feature ranking ----------
    run_anova_selectkbest_to_eda(top_k=25)

    print("EDA saved to:", EDA_OUT.resolve())

if __name__ == "__main__":
    main()