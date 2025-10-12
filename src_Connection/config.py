"""
config.py — Global Configuration
--------------------------------

Central configuration file for the Sustainable AI in Healthcare (DSP5100) project.

Defines:
  • Paths to dataset and results directories
  • Training and federated learning parameters
  • Model architecture selection
  • Normalization and data-split settings

This file acts as the single source of truth for all modules.
"""

from pathlib import Path


# -------------------------------------------------------------------------
# Project Paths
# -------------------------------------------------------------------------
# Root directory of the project
PROJ_ROOT = Path(__file__).resolve().parent.parent

# Dataset: PTB-XL v1.0.3 (update path if moved)
DATA_DIR = PROJ_ROOT / "dataset" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

# Results directory (automatically created if missing)
RESULTS_DIR = PROJ_ROOT / "results"

# CSV metadata files
PTBXL_CSV = DATA_DIR / "ptbxl_database.csv"
SCP_CSV = DATA_DIR / "scp_statements.csv"

# Directory containing waveform records (e.g., records100/, records500/)
DATA_ROOT = DATA_DIR


# -------------------------------------------------------------------------
# Data and Split Settings
# -------------------------------------------------------------------------
SEED = 42                   # Global random seed for reproducibility
SAMPLE_RATE = 100           # Hz, choose 100 for low-res or 500 for high-res
N_CLASSES = 5               # NORM, MI, STTC, HYP, CD
SUPERCLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]

# Patient-level splits (fractions must sum to 1.0)
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


# -------------------------------------------------------------------------
# Federated Learning (FL) Settings
# -------------------------------------------------------------------------
CLIENTS = 4                 # Number of participating clients
EPOCHS_LOCAL = 2            # Local epochs per client per round
BATCH_SIZE = 64
LR = 1e-3                   # Learning rate for Adam optimizer
FEDPROX_MU = 0.01           # FedProx proximal term (0 to disable)
ROUNDS = 15                 # Total federated training rounds

# Freezing configuration (sustainability-driven compute reduction)
FREEZE_THRESHOLD = 600      # Clients with fewer samples freeze early layers
FREEZE_CFG = {
    "patience": 2,          # Rounds without improvement before unfreezing
    "min_delta": 1e-3,      # Minimum improvement to reset patience
    "unfreeze_after": 10,   # Base schedule factor for gradual unfreezing
    "freeze_mode": "gated", # "gated" or "static"
}
# For logging (results/[name].csv) frozen layers vs not (True/False) manually
FREEZE_ENABLED = False       # toggle this for frozen/non-frozen run
EXPERIMENT = {
    "freeze_enabled": FREEZE_ENABLED,
    "run_name": "frozen_run" if FREEZE_ENABLED else "non_frozen_run",
}

# -------------------------------------------------------------------------
# Normalization Parameters
# -------------------------------------------------------------------------
NORM = {
    "enabled": True,        # toggle normalization on/off
    "mode": "zscore",       # Currently only 'zscore' is implemented
    "eps": 1e-6,            # Numerical stability
}


# -------------------------------------------------------------------------
# Model Selection & Tuning / CV
# -------------------------------------------------------------------------
MODEL = {
    "type": "cnn",              # "cnn" or "lstm"
    "lstm_hidden": 128,
    "lstm_layers": 1,
    "bidirectional": True,
}

TUNING = {
    "enabled": False,                   # run CV now?
    "reuse_cached_if_exists": False,
    "use_cached_best": False,            # if enabled=False, load best params from disk if available
    "log_phase": True,                  # log tuning phase in results?
    "log_mode": "same",                 # "same" or "separate" CSV for tuning vs non-tuning
    "phase_labels": {
        "enabled":  "post_cv",          # tuning enabled
        "disabled": "no_cv",            # tuning disabled
        "cached":   "cached_cv"         # tuning disabled but cached params used
    },
}

GRIDSEARCH = {
    "cv": 5,
    "grid": [
        {"lr":1e-4,"batch":32,"epochs":1,"fedprox":0.0},
        {"lr":5e-4,"batch":64,"epochs":2,"fedprox":0.0},
        {"lr":1e-3,"batch":64,"epochs":2,"fedprox":0.001},
        {"lr": 1e-3, "batch": 128, "epochs": 4, "fedprox": 0.0},
        {"lr": 2e-3, "batch": 128, "epochs": 4, "fedprox": 0.001},
    ],
}











# ---------
# ## 0) Config — toggles, label mode, training/FL knobs (from notebook)
# ---------
# NOTE:
# - We do NOT redefine globals above (no shadowing of SEED, PTBXL_CSV, BATCH_SIZE, etc.).
# - Everything here is wrapped in NOTEBOOK so notebooks/scripts can import safely:
#       from src_Connection.config import NOTEBOOK
# - Paths reuse the top-level PTBXL_CSV/SCP_CSV and are cast to str when needed.

NOTEBOOK = {
    # Paths to PTB-XL metadata (reuse top-level paths; cast to str for pandas)
    "PTBXL_CSV": str(PTBXL_CSV),
    "SCP_CSV":   str(SCP_CSV),

    # Label/targets
    "LABEL_MODE": "5class",          # or "10class"
    "SCP_MIN_CONF": 0.0,
    "BINARY_TASK": False,            # keep OFF for 5-class
    "BINARY_SCHEME": "NORM_vs_ALL",
    "CONFUSION_COLLAPSE_TO_3": False,

    # Data / runtime
    "RECORD_FILE_COL": "filename_lr",  # or "filename_hr"
    "FAST_RUN": False,
    "MAX_RECORDS": None,
    "SEED": SEED,                      # reuse global SEED
    "USE_FEATURE_CACHE": False,
    "DEEP_CACHE_TO_DISK": True,
    "SAVE_MIN_TABLE": False,
    "SAVE_ARTIFACTS": False,
    "ART_DIR": "test/artifacts",
    "EDA_SKIP_HEAVY": True,
    "LOW_RAM": False,
    "TORCH_AMP": True,
    "DEEP_FAST_METADATA_ONLY": True,
    "DEEP_MAX_TRAIN_FRAC": 0.80,

    # Sequence & training (notebook-only; does NOT affect script BATCH_SIZE/EPOCHS)
    "SEQ_LEN": 600,
    "DOWNSAMPLE_FACTOR": 2,
    "BATCH_SIZE": 24,
    "EPOCHS": 12,
    "EARLY_STOP_PATIENCE": 0,          # disabled
    "EARLY_STOP_MONITOR": "acc",
    "GRAD_CLIP_NORM": 1.0,
    "BASE_LR": 3e-4,
    "RECURRENT_LR": 2e-4,

    # Run which deep models
    "RUN_TORCH_CNN":  True,
    "RUN_TORCH_RNN":  True,
    "RUN_TORCH_LSTM": True,
    "RUN_TORCH_ANN":  True,
    "DEEP_HYBRID": False,

    # CV
    "RUN_KFOLD_ALL": True,
    "KFOLDS": 5,
    "CV_EPOCHS": 5,

    # Features
    "RUN_FEATURES_BUILD": True,
    "SAVE_FEATURES_CSV": True,
    "FEATURES_CSV_NAME": "basic_signal_features.csv",
    "FEATURES_USE_FOR_BASELINE": True,

    # Federated Learning (Flower) — reuse top-level FL knobs to avoid divergence
    "FL_BASE_MODEL": "ANN",                 # ANN/CNN/RNN/LSTM
    "FL_N_CLIENTS": CLIENTS,
    "FL_SAMPLE_FRAC": 0.75,
    "FL_LOCAL_EPOCHS": 1,
    "FL_ROUNDS": 10,
    "FL_PARTITION": "by_patient",           # by_patient | iid | dirichlet
    "FL_DIRICHLET_ALPHA": 0.3,
    "FL_MIN_SAMPLES_PER_CLIENT": 25,
    "FL_BALANCE_BY_SIZE": True,
    "FL_SERVER_ADDRESS": "127.0.0.1:8080",
}

# Derived (keep)
if NOTEBOOK["FAST_RUN"] and (NOTEBOOK["MAX_RECORDS"] is None):
    NOTEBOOK["MAX_RECORDS"] = 200
if NOTEBOOK["FAST_RUN"]:
    NOTEBOOK["EPOCHS"] = min(NOTEBOOK["EPOCHS"], 4)

# Ensure results directories exist (safe no-op if already present)
try:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "viz").mkdir(parents=True, exist_ok=True)
except Exception:
    pass
