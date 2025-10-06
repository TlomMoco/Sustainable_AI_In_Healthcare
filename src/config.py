from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent[1]
DATA_DIR = PROJ_ROOT / "dataset"
RESULTS_DIR = PROJ_ROOT / "results"

PTBXL_CSV = DATA_DIR / "ptbxl_database.csv"
SCP_CSV = DATA_DIR / "scp_statements.csv"

# Root for WFDB records (filename_lr/hr resolves)
DATA_ROOT = DATA_DIR

# Data / Training
SEED = 42
SAMPLE_RATE = 100 # start with low‑res (100 Hz)
N_CLASSES = 5 # NORM, MI, STTC, HYP, CD


# Federated
CLIENTS = 4
BATCH_SIZE = 64
EPOCHS_LOCAL = 2
LR = 1e-3
FREEZE_THRESHOLD = 600 # small clients freeze backbone
FEDPROX_MU = 0.01 # proximal term