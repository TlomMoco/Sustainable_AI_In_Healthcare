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

"""
eda.py
PTB-XL exploratory pipeline (minimal-table + engineered features).

What it does
------------
1) Loads PTB-XL CSVs using paths in src.config
2) Builds a leakage-safe *minimal* table with record paths + labels (5-class)
3) Delegates feature engineering to src.data_preprocessing.build_features_table
4) Prints a couple of quick summaries

You can extend the bottom section with your richer plots if needed.
"""

from __future__ import annotations
import ast
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------ Project paths & config ------------------
# Ensure we can import src.*
THIS_DIR = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

from src.config import (
    PTBXL_CSV,
    SCP_CSV,
    DATA_ROOT,
    SAMPLE_RATE,
    RESULTS_DIR,
    SUPERCLASSES,
)

# choose records dir by configured SAMPLE_RATE
RECORDS_DIR = DATA_ROOT / ("records100" if int(SAMPLE_RATE) == 100 else "records500")

# ------------------ Utils ------------------
def _wfdb_ok() -> bool:
    try:
        import wfdb  # noqa: F401
        return True
    except Exception:
        return False


def _drop_dat_suffix(path_str: str) -> str:
    """WFDB expects a basenames without extension (no .dat)."""
    return path_str[:-4] if path_str.endswith(".dat") else path_str


# ------------------ Load CSVs ------------------
def load_raw_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    db = pd.read_csv(PTBXL_CSV)
    scp = pd.read_csv(SCP_CSV, encoding="utf-8-sig")
    scp.columns = [str(c).strip() for c in scp.columns]
    return db, scp


# ------------------ Map scp_codes -> 5-class labels ------------------
def make_label_mapper(db: pd.DataFrame, scp_raw: pd.DataFrame):
    """Return function scp_dict -> class_label using 'diagnostic' rows."""
    # allow for different 'code' column names in scp_statements
    obj_cols = [c for c in scp_raw.columns if scp_raw[c].dtype == "object"]
    # Pick the object column with most overlap against keys seen in db['scp_codes']
    def _extract_keys(series, n=2000):
        keys = set()
        for s in series.head(min(n, len(series))):
            try:
                keys.update(map(str, ast.literal_eval(s).keys()))
            except Exception:
                pass
        return keys

    known = _extract_keys(db["scp_codes"])
    scores = {c: len(set(scp_raw[c].astype(str)).intersection(known)) for c in obj_cols} or {obj_cols[0]: 1}
    code_col = max(scores, key=scores.get)
    scp = scp_raw.copy()
    scp["code"] = scp[code_col].astype(str)
    scp = scp.set_index("code", drop=False)

    for req in ["diagnostic", "diagnostic_class", "diagnostic_subclass"]:
        if req not in scp.columns:
            raise ValueError(f"scp_statements.csv missing column: {req}")

    diag_col = scp["diagnostic"].astype(str).str.strip().str.lower()
    is_diag = pd.to_numeric(diag_col, errors="coerce").fillna(0).gt(0) | diag_col.isin(
        {"1", "1.0", "true", "t", "yes", "y", "on"}
    )
    diag_only = scp.loc[is_diag, ["diagnostic_class", "diagnostic_subclass"]].copy()
    valid_codes = set(diag_only.index)
    label_field = "diagnostic_class"  # 5-class

    def to_diag_label(scp_codes_dict: Dict[str, float], min_conf: float = 0.0) -> Optional[str]:
        try:
            items = [(str(k), float(v)) for k, v in scp_codes_dict.items()]
        except Exception:
            return None
        items = [(k, v) for k, v in items if k in valid_codes and v >= min_conf]
        if not items:
            return None
        agg: Dict[str, float] = {}
        for k, w in items:
            lab = str(diag_only.loc[k, label_field])
            agg[lab] = agg.get(lab, 0.0) + w
        return max(agg.items(), key=lambda kv: kv[1])[0]

    return to_diag_label


# ------------------ Build minimal table ------------------
META_KEEP = ["ecg_id", "patient_id", "age", "sex", "device", "recording_date", "scp_codes", "strat_fold"]

def build_minimal_table(db: pd.DataFrame, to_label, min_conf: float = 0.0) -> pd.DataFrame:
    rows: list[dict] = []
    # choose column based on SAMPLE_RATE
    file_col = "filename_lr" if int(SAMPLE_RATE) == 100 else "filename_hr"

    for _, r in db.iterrows():
        try:
            scp_codes = ast.literal_eval(r.get("scp_codes", "{}"))
        except Exception:
            continue
        y = to_label(scp_codes, min_conf=min_conf)
        if y is None:
            continue

        # metadata
        row: Dict[str, Any] = {}
        for k in META_KEEP:
            v = r.get(k, np.nan)
            if k == "recording_date":
                try:
                    row[k] = pd.to_datetime(v).date().isoformat()
                except Exception:
                    row[k] = np.nan
            else:
                row[k] = v

        # record path (strip .dat suffix)
        rec_rel = str(r[file_col])
        full = _drop_dat_suffix(str((DATA_ROOT / rec_rel).as_posix()))
        row["record_path"] = full
        row["label"] = str(y)
        rows.append(row)

    if not rows:
        raise RuntimeError("No rows produced — check file paths/mappings.")
    df = pd.DataFrame(rows)

    # Optional: keep only classes in SUPERCLASSES order if provided
    if SUPERCLASSES:
        df = df[df["label"].isin(SUPERCLASSES)].copy()

    # Use ecg_id as index if available
    if "ecg_id" in df.columns:
        df = df.set_index(df["ecg_id"].astype(int)).drop(columns=["ecg_id"])
    return df


# ------------------ Feature Engineering (delegated) ------------------
from src.data_preprocessing import build_features_table  # single source of truth


# ------------------ Main ------------------
def main() -> None:
    print("[EDA] config:")
    print(f"  PTBXL_CSV  = {PTBXL_CSV}")
    print(f"  SCP_CSV    = {SCP_CSV}")
    print(f"  DATA_ROOT  = {DATA_ROOT}")
    print(f"  RECORDS    = {RECORDS_DIR}")
    print(f"  SAMPLE_RATE= {SAMPLE_RATE} Hz")
    print(f"  RESULTS    = {RESULTS_DIR}")

    if not PTBXL_CSV.exists() or not SCP_CSV.exists():
        raise FileNotFoundError("PTB-XL CSVs not found. Please check paths in src/config.py.")

    if not RECORDS_DIR.exists():
        raise FileNotFoundError(f"Records dir not found: {RECORDS_DIR}")

    if not _wfdb_ok():
        raise RuntimeError("WFDB not installed. Please `pip install wfdb`.")

    db, scp = load_raw_tables()
    print(f"[EDA] Loaded db: {db.shape} | scp: {scp.shape}")

    to_label = make_label_mapper(db, scp)
    minimal = build_minimal_table(db, to_label, min_conf=0.0)
    print(f"[EDA] Minimal table for deep/feats: {minimal.shape}")
    print(minimal[["record_path", "label"]].head(3))

    # Build engineered features (saved to RESULTS_DIR/basic_signal_features.csv)
    feature_df = build_features_table(
        minimal_df=minimal,
        save_csv=True,
        csv_name="basic_signal_features.csv",
    )

    # Quick summaries (extend with richer EDA if you want)
    print("\n[EDA] Class counts:")
    print(feature_df["label"].value_counts().sort_index())

    if {"age", "sex"}.issubset(minimal.columns):
        # normalize 'sex' to strings
        def _sex_norm(s: pd.Series) -> pd.Series:
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

        minimal = minimal.copy()
        minimal["sex_norm"] = _sex_norm(minimal["sex"])
        print("\n[EDA] Sex distribution:")
        print(minimal["sex_norm"].value_counts())

        # Age stats (if numeric)
        a = pd.to_numeric(minimal.get("age", np.nan), errors="coerce")
        if a.notna().any():
            print("\n[EDA] Age summary:")
            print(a.describe())

    print("\n[EDA] Done. Engineered features saved under results/ by default.")


if __name__ == "__main__":
    main()
