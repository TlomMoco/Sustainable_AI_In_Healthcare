from pathlib import Path


# --- paths ---
# project root: ./Sustainable_AI_Healthcare
PROJ_ROOT  = Path(__file__).resolve().parent.parent

# Folders
DATA_DIR = PROJ_ROOT / "dataset" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
RESULTS_DIR = PROJ_ROOT / "results"

# Files
PTBXL_CSV = DATA_DIR / "ptbxl_database.csv"
SCP_CSV = DATA_DIR / "scp_statements.csv"

# where WFDB 'records100/records500' live (in your screenshot: directly under dataset/)
DATA_ROOT = DATA_DIR


# --- training / FL settings ---
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

"""
if __name__ == "__main__":
    print("PROJ_ROOT:", PROJ_ROOT)
    print("PTBXL_CSV exists:", PTBXL_CSV.exists())
    print("SCP_CSV exists:", SCP_CSV.exists())
    print("records100 exists:", (DATA_ROOT / "records100").exists())
"""