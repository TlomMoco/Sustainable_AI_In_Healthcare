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

# Subdirectory for hyperparameter tuning results
TUNING_DIR = RESULTS_DIR / "tuning"


# If True, whenever we build a feature table, also cache it to CSV for reuse
SAVE_FEATURES_CSV = True
FEATURES_CSV_NAME = "features.csv"            # filename under ART_DIR


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
FREEZE_ENABLED = True       # toggle this for frozen/non-frozen run

# -------------------------------------------------------------------------
# Normalization Parameters
# -------------------------------------------------------------------------
NORM = {
    "enabled": True,        # toggle normalization on/off
    "mode": "zscore",       # Currently only 'zscore' is implemented
    "eps": 1e-6,            # Numerical stability
}

# -------------------------------------------------------------------------
# ANOVA / lead-selection knobs
# -------------------------------------------------------------------------
ANOVA_FSCORE_THRESHOLD: float = 300.0
ANOVA_FALLBACK_LEADS: int = 6

# -------------------------------------------------------------------------
# Model Selection & Tuning / CV
# -------------------------------------------------------------------------
MODEL = {
    "type": "cnn_lstm",              # "cnn", "lstm", or "cnn_lstm"
    "lstm_hidden": 128,
    "lstm_layers": 1,
    "bidirectional": True,
    # "use_attention": False,  # optional toggle for later
}

EXPERIMENT = {
    "freeze_enabled": FREEZE_ENABLED,
    "run_name": f"{MODEL['type']}_{'frozen_run' if FREEZE_ENABLED else 'non_frozen_run'}",
}

# Class weighting and smoothing to help the minority class (HYP)
CLASS_WEIGHTS = {
    "enabled": True,                  # turn on weighted CE
    "boost": {"HYP": 2.0},            # optional extra emphasis on HYP
    "label_smoothing": 0.05           # mild smoothing for stability
}

TUNING = {
    "enabled": True,                   # run CV now?
    "use_cached_best": False,           # if enabled=True, use cached best params if available
    "log_phase": True,                 # log tuning phase in results?
    "log_mode": "same",                # "same" or "separate" CSV for tuning vs non-tuning
    "phase_labels": {
        "enabled":  "tuned",           # tuning enabled
        "cached":   "tuned",           # tuning disabled but cached params used
        "disabled": "default"          # tuning disabled
    },
}
GRIDSEARCH = {
    "cv": 5,
    "grid": [
        {"lr": 1e-3, "batch": 64,  "epochs": 2, "fedprox": 0.001},
        {"lr": 5e-4, "batch": 64,  "epochs": 2, "fedprox": 0.0},
        {"lr": 1e-3, "batch": 128, "epochs": 2, "fedprox": 0.001},

        # focused candidates (safe with FedProx)
        {"lr": 1e-3, "batch": 64,  "epochs": 3, "fedprox": 0.01},
        {"lr": 5e-4, "batch": 128, "epochs": 3, "fedprox": 0.01},
    ],
}

def tuning_paths(run_name: str, cid: int, freeze_enabled: bool):
    """
    Model-aware tuning cache paths.

    Produces (under results/tuning/<model_type>/):
      client{cid}_best.json
      client{cid}_cv.csv

    The signature remains the same to avoid touching call sites.
    """
    # Use a per-model subdirectory to keep caches separate when switching models
    model_type = str(MODEL.get("type", "unknown")).lower()
    base = (TUNING_DIR / model_type)
    base.mkdir(parents=True, exist_ok=True)

    best_json = base / f"client{cid}_best.json"
    cv_csv = base / f"client{cid}_cv.csv"
    return best_json, cv_csv