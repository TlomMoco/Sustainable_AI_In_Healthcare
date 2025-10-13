"""
config.py — Global Configuration
--------------------------------
Central configuration file for the Sustainable AI in Healthcare (DSP5100) project.

What this module controls
-------------------------
• Paths to datasets, artifacts, and results directories
• Data split strategy and PTB-XL sampling details
• Federated learning knobs (clients, local epochs, FedProx μ, rounds, freezing)
• Normalization policy
• Model selection (cnn/lstm) and per-architecture hints
• Optional client-side CV/tuning and grid-search space
• Notebook/script parity via NOTEBOOK dict + backward-compat exports

Where it’s used (connections)
-----------------------------
• src_Connection.data_loader
    - Uses DATA_DIR, PTBXL_CSV, SCP_CSV, RECORD_FILE_COL, RECORDS_DIR, SAMPLE_RATE
    - Uses SPLITS / SEED for deterministic patient-safe splits
• src_Connection.data_preprocessing
    - Consumes NOTEBOOK[*] for deep pipeline toggles (e.g., BATCH_SIZE, EPOCHS)
• src_Connection.models
    - Reads MODEL (type/lstm params), N_CLASSES
• src_Connection.Centralized
    - Pulls NOTEBOOK toggles (EPOCHS, BATCH_SIZE, RUN_TORCH_*), RESULTS_DIR
• src_Connection.Client / Server (Flower)
    - Uses FL_* (ROUNDS, CLIENTS, addresses), LR/BATCH_SIZE/EPOCHS_LOCAL,
      FREEZE_* config, TUNING/GRIDSEARCH, EXPERIMENT.run_name
• src_Connection.results_visualization
    - Uses RESULTS_DIR for plots/exports

Notes
-----
• Changing SAMPLE_RATE toggles which PTB-XL record directory/column are read.
• FREEZE_* is a sustainability lever: reduces compute by freezing feature layers.
• The NOTEBOOK dict mirrors many top-level knobs to keep parity with notebooks.
"""

from pathlib import Path

# -------------------------------------------------------------------------
# Project Paths
# -------------------------------------------------------------------------
# Base project root and canonical dataset/results locations.
PROJ_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR   = PROJ_ROOT / "dataset" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
RESULTS_DIR = PROJ_ROOT / "results"

# CSV metadata files (consumed by data_loader.load_metadata)
PTBXL_CSV = DATA_DIR / "ptbxl_database.csv"
SCP_CSV   = DATA_DIR / "scp_statements.csv"

# Directory containing waveform records (e.g., records100/, records500/)
# For PTB-XL, the record directories live directly under DATA_DIR.
DATA_ROOT = DATA_DIR

# -------------------------------------------------------------------------
# Data and Split Settings
# -------------------------------------------------------------------------
SEED         = 42                      # Global seed used across modules for reproducibility
SAMPLE_RATE  = 100                     # 100 (low-res) or 500 (high-res) PTB-XL sampling
N_CLASSES    = 5                       # Number of diagnostic superclasses used
SUPERCLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]  # Order matters across client/server/metrics

# Global (patient-safe) split ratios used by data_loader.stratified_patient_split_3way
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Convenience: column name and record dir depend on SAMPLE_RATE
#  - filename_lr: links to records100/ (100 Hz)
#  - filename_hr: links to records500/ (500 Hz)
RECORD_FILE_COL = "filename_lr" if SAMPLE_RATE == 100 else "filename_hr"
RECORDS_DIR     = DATA_ROOT / ("records100" if SAMPLE_RATE == 100 else "records500")

# -------------------------------------------------------------------------
# Federated Learning (FL) Settings
# -------------------------------------------------------------------------
CLIENTS       = 4       # Number of federated clients expected
EPOCHS_LOCAL  = 2       # Local epochs per client per round
BATCH_SIZE    = 64      # Default local batch size (client; deep pipeline may override)
LR            = 1e-3    # Default client learning rate
FEDPROX_MU    = 0.01    # μ for FedProx proximal term; 0.0 disables FedProx
ROUNDS        = 15      # Number of FL rounds

# Freezing configuration (sustainability-driven compute reduction)
#  - FREEZE_THRESHOLD: small clients (< threshold) start more frozen (feature extractor)
#  - FREEZE_CFG: reactive adjustments based on validation dynamics (see Client._apply_freeze_policy)
FREEZE_THRESHOLD = 600
FREEZE_CFG = {
    "patience": 2,          # steps without improvement before adjusting
    "min_delta": 1e-3,      # significant improvement threshold
    "unfreeze_after": 10,   # rounds over which to gradually unfreeze for small clients
    "freeze_mode": "gated", # "gated" (reactive) or "static" (future option)
}

# EXPERIMENT toggles naming and enabling of freezing behaviors
FREEZE_ENABLED = False
EXPERIMENT = {
    "freeze_enabled": FREEZE_ENABLED,
    "run_name": "frozen_run" if FREEZE_ENABLED else "non_frozen_run",
}

# Server networking
# - Server binds to FL_SERVER_BIND (used by Server.py)
# - Clients connect to FL_SERVER_ADDRESS (used by Client.py)
# For distributed runs, set FL_SERVER_ADDRESS to "<host>:<port>" reachable by clients.
FL_SERVER_BIND    = "0.0.0.0:8080"   # used by Server.py
FL_SERVER_ADDRESS = "127.0.0.1:8080" # used by Client.py (local dev)

# -------------------------------------------------------------------------
# Normalization Parameters
# -------------------------------------------------------------------------
# NORM["enabled"] gates per-lead z-score normalization (data_loader + client tensorization).
# eps prevents division-by-zero for near-constant leads.
NORM = {"enabled": True, "mode": "zscore", "eps": 1e-6}

# -------------------------------------------------------------------------
# Model Selection & Tuning / CV
# -------------------------------------------------------------------------
# MODEL controls the architecture factory in src_Connection.models.
MODEL = {
    "type": "cnn",   # "cnn" | "lstm" (extendable by your models factory)
    "lstm_hidden": 128,
    "lstm_layers": 1,
    "bidirectional": True,
}

# Client-side tuning controls (src_Connection.tuning.run_client_cv)
# - If enabled, each client may run a small CV to pick lr/batch/epochs/fedprox
TUNING = {
    "enabled": False,
    "reuse_cached_if_exists": False,  # Reuse tuned params if tuning artifacts exist
    "use_cached_best": False,         # Use cached best even if tuning disabled
    "log_phase": True,                # Adds phase label in CSV metrics
    "log_mode": "same",               # "same" -> single file, "separate" -> per-phase files
    "phase_labels": {"enabled": "post_cv", "disabled": "no_cv", "cached": "cached_cv"},
}

# Grid search space for client CV; small and fast by design
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
# Many Centralized/EDA flows read from NOTEBOOK to keep notebooks/scripts consistent.
NOTEBOOK = {
    # Files/paths (stringified for easy JSON/export in notebooks)
    "PTBXL_CSV": str(PTBXL_CSV),
    "SCP_CSV":   str(SCP_CSV),

    # Labeling/eval visualization knobs
    "LABEL_MODE": "5class",
    "SCP_MIN_CONF": 0.0,
    "BINARY_TASK": False,
    "BINARY_SCHEME": "NORM_vs_ALL",
    "CONFUSION_COLLAPSE_TO_3": False,

    # Dataset column / runtime toggles
    "RECORD_FILE_COL": RECORD_FILE_COL,
    "FAST_RUN": False,             # If True, caps records/epochs for quick iterations
    "MAX_RECORDS": None,           # Auto-capped below when FAST_RUN=True
    "SEED": SEED,
    "USE_FEATURE_CACHE": False,
    "DEEP_CACHE_TO_DISK": True,
    "SAVE_MIN_TABLE": False,
    "SAVE_ARTIFACTS": False,
    "ART_DIR": "test/artifacts",
    "EDA_SKIP_HEAVY": True,
    "LOW_RAM": False,
    "TORCH_AMP": True,             # Mixed precision enable
    "DEEP_FAST_METADATA_ONLY": True,
    "DEEP_MAX_TRAIN_FRAC": 0.80,   # Optional subsampling on deep training

    # Deep pipeline runtime knobs
    "SEQ_LEN": 600,
    "DOWNSAMPLE_FACTOR": 2,
    "BATCH_SIZE": 24,              # Deep pipeline default (distinct from client BATCH_SIZE)
    "EPOCHS": 12,
    "EARLY_STOP_PATIENCE": 0,      # 0 disables early stopping in centralized runs
    "EARLY_STOP_MONITOR": "acc",
    "GRAD_CLIP_NORM": 1.0,
    "BASE_LR": 3e-4,
    "RECURRENT_LR": 2e-4,

    # Which torch models to run in Centralized.py
    "RUN_TORCH_CNN":  True,
    "RUN_TORCH_RNN":  True,
    "RUN_TORCH_LSTM": True,
    "RUN_TORCH_ANN":  True,
    "DEEP_HYBRID": False,

    # KFold CV (centralized)
    "RUN_KFOLD_ALL": True,
    "KFOLDS": 5,
    "CV_EPOCHS": 5,

    # Feature engineering exports
    "RUN_FEATURES_BUILD": True,
    "SAVE_FEATURES_CSV": True,
    "FEATURES_CSV_NAME": "basic_signal_features.csv",
    "FEATURES_USE_FOR_BASELINE": True,

    # Mirror top-level FL knobs for notebook UIs and scripts
    "FL_BASE_MODEL": "ANN",
    "FL_N_CLIENTS": CLIENTS,
    "FL_SAMPLE_FRAC": 0.75,
    "FL_LOCAL_EPOCHS": EPOCHS_LOCAL,
    "FL_ROUNDS": ROUNDS,
    "FL_PARTITION": "by_patient",   # by_patient | iid | dirichlet
    "FL_DIRICHLET_ALPHA": 0.3,
    "FL_MIN_SAMPLES_PER_CLIENT": 25,
    "FL_BALANCE_BY_SIZE": True,
    "FL_SERVER_ADDRESS": FL_SERVER_ADDRESS,
}

# Derived convenience for FAST_RUN (keeps quick iterations genuinely quick)
if NOTEBOOK["FAST_RUN"] and NOTEBOOK["MAX_RECORDS"] is None:
    NOTEBOOK["MAX_RECORDS"] = 200
if NOTEBOOK["FAST_RUN"]:
    NOTEBOOK["EPOCHS"] = min(NOTEBOOK["EPOCHS"], 4)

# Ensure results directories exist early (plots/exports won’t fail on mkdir)
try:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "viz").mkdir(parents=True, exist_ok=True)
except Exception:
    # Non-fatal if environment is read-only; callers can mkdir later.
    pass

# -------------------------------------------------------------------------
# Backwards-compat exports (keep old code working)
# -------------------------------------------------------------------------
# These mirror NOTEBOOK keys at module level so legacy code can do:
#   from src_Connection.config import EPOCHS, BATCH_SIZE, ...
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