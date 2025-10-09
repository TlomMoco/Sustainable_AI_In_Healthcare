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
# 1. Project Paths
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
# 2. Data and Split Settings
# -------------------------------------------------------------------------
SEED = 42               # Global random seed for reproducibility
SAMPLE_RATE = 100       # Hz, choose 100 for low-res or 500 for high-res
N_CLASSES = 5           # NORM, MI, STTC, HYP, CD
SUPERCLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]


# Patient-level splits (fractions must sum to 1.0)
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}


# -------------------------------------------------------------------------
# 3. Federated Learning (FL) Settings
# -------------------------------------------------------------------------
CLIENTS = 4             # Number of participating clients
EPOCHS_LOCAL = 2        # Local epochs per client per round
BATCH_SIZE = 64
LR = 1e-3               # Learning rate for Adam optimizer
FEDPROX_MU = 0.01       # FedProx proximal term (0 to disable)

# Freezing configuration (sustainability-driven compute reduction)
FREEZE_THRESHOLD = 100  # Clients with fewer samples freeze early layers
FREEZE_CFG = {
    "patience": 2,          # Rounds without improvement before unfreezing
    "min_delta": 1e-3,      # Minimum improvement to reset patience
    "unfreeze_after": 10,   # Base schedule factor for gradual unfreezing
    "freeze_mode": "gated", # "gated" or "static"
}


# -------------------------------------------------------------------------
# 4. Normalization Parameters
# -------------------------------------------------------------------------
NORM = {
    "enabled": True,
    "mode": "zscore",       # Currently only 'zscore' is implemented
    "eps": 1e-6,            # Numerical stability
}


# -------------------------------------------------------------------------
# 5. Model Selection
# -------------------------------------------------------------------------
MODEL = {
    "type": "cnn",          # "cnn" or "lstm"
    "lstm_hidden": 128,     # Hidden units for LSTM
    "lstm_layers": 1,       # Number of LSTM layers
    "bidirectional": True,  # Bidirectional LSTM
}