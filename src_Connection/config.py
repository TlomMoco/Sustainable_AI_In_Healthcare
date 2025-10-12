"""
config.py — Global Configuration
--------------------------------

Central configuration file for the Sustainable AI in Healthcare (DSP5100) project.
"""

from pathlib import Path

# -------------------------------------------------------------------------
# Project Paths
# -------------------------------------------------------------------------
PROJ_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = PROJ_ROOT / "dataset" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
RESULTS_DIR = PROJ_ROOT / "results"

# CSV metadata files
PTBXL_CSV = DATA_DIR / "ptbxl_database.csv"
SCP_CSV   = DATA_DIR / "scp_statements.csv"

# Directory containing waveform records (e.g., records100/, records500/)
DATA_ROOT = DATA_DIR

# -------------------------------------------------------------------------
# Data and Split Settings
# -------------------------------------------------------------------------
SEED         = 42
SAMPLE_RATE  = 100                  # 100 (low-res) or 500 (high-res)
N_CLASSES    = 5
SUPERCLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

# For convenience: pick filename column & records dir from SAMPLE_RATE
RECORD_FILE_COL = "filename_lr" if SAMPLE_RATE == 100 else "filename_hr"
RECORDS_DIR     = DATA_ROOT / ("records100" if SAMPLE_RATE == 100 else "records500")

# -------------------------------------------------------------------------
# Federated Learning (FL) Settings
# -------------------------------------------------------------------------
CLIENTS       = 4
EPOCHS_LOCAL  = 2
BATCH_SIZE    = 64
LR            = 1e-3
FEDPROX_MU    = 0.01
ROUNDS        = 15

# Freezing configuration (sustainability-driven compute reduction)
FREEZE_THRESHOLD = 600
FREEZE_CFG = {
    "patience": 2,
    "min_delta": 1e-3,
    "unfreeze_after": 10,
    "freeze_mode": "gated",  # "gated" or "static"
}

FREEZE_ENABLED = False
EXPERIMENT = {
    "freeze_enabled": FREEZE_ENABLED,
    "run_name": "frozen_run" if FREEZE_ENABLED else "non_frozen_run",
}

# Server networking
# - Server binds to FL_SERVER_BIND
# - Clients connect to FL_SERVER_ADDRESS
FL_SERVER_BIND    = "0.0.0.0:8080"   # used by Server.py
FL_SERVER_ADDRESS = "127.0.0.1:8080" # used by Client.py (local dev)

# -------------------------------------------------------------------------
# Normalization Parameters
# -------------------------------------------------------------------------
NORM = {"enabled": True, "mode": "zscore", "eps": 1e-6}

# -------------------------------------------------------------------------
# Model Selection & Tuning / CV
# -------------------------------------------------------------------------
MODEL = {
    "type": "cnn",   # "cnn" | "lstm"
    "lstm_hidden": 128,
    "lstm_layers": 1,
    "bidirectional": True,
}

TUNING = {
    "enabled": False,
    "reuse_cached_if_exists": False,
    "use_cached_best": False,
    "log_phase": True,
    "log_mode": "same",  # "same" or "separate"
    "phase_labels": {"enabled": "post_cv", "disabled": "no_cv", "cached": "cached_cv"},
}

GRIDSEARCH = {
    "cv": 5,
    "grid": [
        {"lr": 1e-4, "batch": 32,  "epochs": 1, "fedprox": 0.0},
        {"lr": 5e-4, "batch": 64,  "epochs": 2, "fedprox": 0.0},
        {"lr": 1e-3, "batch": 64,  "epochs": 2, "fedprox": 0.001},
        {"lr": 1e-3, "batch": 128, "epochs": 4, "fedprox": 0.0},
        {"lr": 2e-3, "batch": 128, "epochs": 4, "fedprox": 0.001},
    ],
}

# -------------------------------------------------------------------------
# NOTEBOOK (kept for script/notebook parity and back-compat)
# -------------------------------------------------------------------------
NOTEBOOK = {
    "PTBXL_CSV": str(PTBXL_CSV),
    "SCP_CSV":   str(SCP_CSV),

    "LABEL_MODE": "5class",
    "SCP_MIN_CONF": 0.0,
    "BINARY_TASK": False,
    "BINARY_SCHEME": "NORM_vs_ALL",
    "CONFUSION_COLLAPSE_TO_3": False,

    "RECORD_FILE_COL": RECORD_FILE_COL,
    "FAST_RUN": False,
    "MAX_RECORDS": None,
    "SEED": SEED,
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

    "SEQ_LEN": 600,
    "DOWNSAMPLE_FACTOR": 2,
    "BATCH_SIZE": 24,
    "EPOCHS": 12,
    "EARLY_STOP_PATIENCE": 0,
    "EARLY_STOP_MONITOR": "acc",
    "GRAD_CLIP_NORM": 1.0,
    "BASE_LR": 3e-4,
    "RECURRENT_LR": 2e-4,

    "RUN_TORCH_CNN":  True,
    "RUN_TORCH_RNN":  True,
    "RUN_TORCH_LSTM": True,
    "RUN_TORCH_ANN":  True,
    "DEEP_HYBRID": False,

    "RUN_KFOLD_ALL": True,
    "KFOLDS": 5,
    "CV_EPOCHS": 5,

    "RUN_FEATURES_BUILD": True,
    "SAVE_FEATURES_CSV": True,
    "FEATURES_CSV_NAME": "basic_signal_features.csv",
    "FEATURES_USE_FOR_BASELINE": True,

    "FL_BASE_MODEL": "ANN",
    "FL_N_CLIENTS": CLIENTS,
    "FL_SAMPLE_FRAC": 0.75,
    "FL_LOCAL_EPOCHS": 1,
    "FL_ROUNDS": 10,
    "FL_PARTITION": "by_patient",   # by_patient | iid | dirichlet
    "FL_DIRICHLET_ALPHA": 0.3,
    "FL_MIN_SAMPLES_PER_CLIENT": 25,
    "FL_BALANCE_BY_SIZE": True,
    "FL_SERVER_ADDRESS": FL_SERVER_ADDRESS,
}

# Derived convenience
if NOTEBOOK["FAST_RUN"] and NOTEBOOK["MAX_RECORDS"] is None:
    NOTEBOOK["MAX_RECORDS"] = 200
if NOTEBOOK["FAST_RUN"]:
    NOTEBOOK["EPOCHS"] = min(NOTEBOOK["EPOCHS"], 4)

# Ensure results directories
try:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "viz").mkdir(parents=True, exist_ok=True)
except Exception:
    pass

# -------------------------------------------------------------------------
# Backwards-compat exports (keep old code working)
# -------------------------------------------------------------------------
FAST_RUN                = NOTEBOOK["FAST_RUN"]
MAX_RECORDS             = NOTEBOOK["MAX_RECORDS"]
USE_FEATURE_CACHE       = NOTEBOOK["USE_FEATURE_CACHE"]
DEEP_CACHE_TO_DISK      = NOTEBOOK["DEEP_CACHE_TO_DISK"]
SAVE_MIN_TABLE          = NOTEBOOK["SAVE_MIN_TABLE"]
SAVE_ARTIFACTS          = NOTEBOOK["SAVE_ARTIFACTS"]
ART_DIR                 = NOTEBOOK["ART_DIR"]
EDA_SKIP_HEAVY          = NOTEBOOK["EDA_SKIP_HEAVY"]
LOW_RAM                 = NOTEBOOK["LOW_RAM"]
TORCH_AMP               = NOTEBOOK["TORCH_AMP"]
DEEP_FAST_METADATA_ONLY = NOTEBOOK["DEEP_FAST_METADATA_ONLY"]
DEEP_MAX_TRAIN_FRAC     = NOTEBOOK["DEEP_MAX_TRAIN_FRAC"]

SEQ_LEN            = NOTEBOOK["SEQ_LEN"]
DOWNSAMPLE_FACTOR  = NOTEBOOK["DOWNSAMPLE_FACTOR"]
EPOCHS             = NOTEBOOK["EPOCHS"]
EARLY_STOP_PATIENCE= NOTEBOOK["EARLY_STOP_PATIENCE"]
EARLY_STOP_MONITOR = NOTEBOOK["EARLY_STOP_MONITOR"]
GRAD_CLIP_NORM     = NOTEBOOK["GRAD_CLIP_NORM"]
BASE_LR            = NOTEBOOK["BASE_LR"]
RECURRENT_LR       = NOTEBOOK["RECURRENT_LR"]

RUN_TORCH_CNN   = NOTEBOOK["RUN_TORCH_CNN"]
RUN_TORCH_RNN   = NOTEBOOK["RUN_TORCH_RNN"]
RUN_TORCH_LSTM  = NOTEBOOK["RUN_TORCH_LSTM"]
RUN_TORCH_ANN   = NOTEBOOK["RUN_TORCH_ANN"]
DEEP_HYBRID     = NOTEBOOK["DEEP_HYBRID"]

RUN_KFOLD_ALL = NOTEBOOK["RUN_KFOLD_ALL"]
KFOLDS       = NOTEBOOK["KFOLDS"]
CV_EPOCHS    = NOTEBOOK["CV_EPOCHS"]

RUN_FEATURES_BUILD        = NOTEBOOK["RUN_FEATURES_BUILD"]
SAVE_FEATURES_CSV         = NOTEBOOK["SAVE_FEATURES_CSV"]
FEATURES_CSV_NAME         = NOTEBOOK["FEATURES_CSV_NAME"]
FEATURES_USE_FOR_BASELINE = NOTEBOOK["FEATURES_USE_FOR_BASELINE"]

FL_BASE_MODEL             = NOTEBOOK["FL_BASE_MODEL"]
FL_N_CLIENTS              = NOTEBOOK["FL_N_CLIENTS"]
FL_SAMPLE_FRAC            = NOTEBOOK["FL_SAMPLE_FRAC"]
FL_LOCAL_EPOCHS           = NOTEBOOK["FL_LOCAL_EPOCHS"]
FL_ROUNDS                 = NOTEBOOK["FL_ROUNDS"]
FL_PARTITION              = NOTEBOOK["FL_PARTITION"]
FL_DIRICHLET_ALPHA        = NOTEBOOK["FL_DIRICHLET_ALPHA"]
FL_MIN_SAMPLES_PER_CLIENT = NOTEBOOK["FL_MIN_SAMPLES_PER_CLIENT"]
FL_BALANCE_BY_SIZE        = NOTEBOOK["FL_BALANCE_BY_SIZE"]
FL_SERVER_ADDRESS         = NOTEBOOK["FL_SERVER_ADDRESS"]
