from pathlib import Path


# --- paths ---
# project root: ./Sustainable_AI_Healthcare
PROJ_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR = PROJ_ROOT / "dataset" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
RESULTS_DIR = PROJ_ROOT / "results"
PTBXL_CSV = DATA_DIR / "ptbxl_database.csv"
SCP_CSV = DATA_DIR / "scp_statements.csv"
DATA_ROOT = DATA_DIR        # where WFDB 'records100/records500' live (in your screenshot: directly under dataset/)


# --- training / FL settings ---
# Data / Training
SEED = 42
SAMPLE_RATE = 100           # start with low‑res (100 Hz)
N_CLASSES = 5               # NORM, MI, STTC, HYP, CD
SUPERCLASSES = ["CD", "HYP", "MI", "NORM", "STTC"]


# Federated
CLIENTS = 4
BATCH_SIZE = 64
EPOCHS_LOCAL = 2
LR = 1e-3
FEDPROX_MU = 0.01 # proximal term
FREEZE_THRESHOLD = 600 # small clients freeze backbone
FREEZE_CFG = {
    "patience": 2,          # rounds without improvement before unfreezing
    "min_delta": 1e-3,      # min improvement to reset patience
    "unfreeze_after": 10,   # base schedule scaling factor
    "freeze_mode": "gated", # or "static"
}


# --- Splits ---
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}  # patient-level splits


# --- Normalization ---
NORM = {
    "enabled": True,
    "mode": "zscore",      # currently only 'zscore' implemented
    "eps": 1e-6,
}


# --- Model selection ---
MODEL = {
    "type": "cnn",         # "cnn" or "lstm"
    "lstm_hidden": 128,
    "lstm_layers": 1,
    "bidirectional": True,
}

