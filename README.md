PTB-XL Federated Learning — Simple Run Instructions
===================================================

1) Environment
--------------
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

2) Data placement (PTB-XL v1.0.3)
---------------------------------
Place files like this (example):
<project>/dataset/ptbxl/
  ptbxl_database.csv
  scp_statements.csv
  records100/
  records500/            # optional

3) Configure (src/config.py)
----------------------------
- Set DATA_DIR to your dataset folder path, e.g.:
  DATA_DIR = "<project>/dataset/ptbxl"
  - Keep defaults unless you know you need changes.
    Common edits:
      SAMPLE_RATE = 100
      CLIENTS = 4
      ROUNDS = 15
      EPOCHS_LOCAL = 2
      BATCH_SIZE = 64
      LR = 1e-3
      FREEZE_ENABLED = False
      SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
      MODEL["type"] = "cnn"
      TUNING["enabled"] = False 
      ANOVA_FSCORE_THRESHOLD: float = 300.0
      ANOVA_FALLBACK_LEADS: int = 6

4) Run order (from project root)
--------------------------------
# (A) Optional: EDA & basic features
python -m src.eda

# (B) Centralized baselines
python -m src.Centralized

# (C) Federated training (start server, then clients)
python -m src.Server
# In separate terminals (0..3 for 4 clients)
python -m src.Client --cid 0
python -m src.Client --cid 1
python -m src.Client --cid 2
python -m src.Client --cid 3

# (D) Optional: Grid search (5-fold CV if enabled in config)
python -m src.tuning

# (E) Plots & comparison (e.g., frozen vs non-frozen runs)
python -m src.results_visualization

5) Outputs (results/)
---------------------
<run_name>.csv                # global + per-client round metrics
<run_name>_perclass.csv       # per-class accuracy across rounds
<run_name>_cm.csv             # confusion matrix (long)
viz/                          # PNG plots (accuracy, per-class, confusion)

6) Quick fixes
--------------
- "Data not found": Check DATA_DIR and that ptbxl_database.csv, scp_statements.csv, and records100/ exist.
- "Clients can't connect": Start Server first; leave defaults for local run (127.0.0.1:8080).
- "GPU not used": python -c "import torch; print(torch.cuda.is_available())"