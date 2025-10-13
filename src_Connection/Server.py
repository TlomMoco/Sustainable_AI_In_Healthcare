# src_Connection/Server.py
"""
Server.py — Federated Learning Orchestrator
-------------------------------------------

Coordinates the federated training process across multiple PTB-XL clients.

Launch order (multi-terminal):
    $ python -m src_Connection.Server         # Terminal 1 (server)
    $ python -m src_Connection.Client --cid 0 # Terminal 2
    $ python -m src_Connection.Client --cid 1 # Terminal 3
    $ python -m src_Connection.Client --cid 2 # ...
    $ python -m src_Connection.Client --cid 3

Where this module connects
--------------------------
• Starts a Flower (flwr) server that coordinates clients launched by
  `src_Connection.Client` (PTBClient).
• Aggregates metrics emitted from each client’s `evaluate()` and `fit()`.
• Writes round-level GLOBAL metrics, per-class accuracy, and confusion matrices
  to CSVs under `config.RESULTS_DIR` for downstream visualization:
  - `src_Connection.results_visualization` reads these CSVs to make plots/ANOVA.
• Reads FL hyperparameters and experiment names from `src_Connection.config`.

Outputs (CSV)
-------------
• <run_name>.csv                — round-level GLOBAL accuracy/loss (and client logs from clients)
• <run_name>_perclass.csv       — per-class accuracy derived from CM each round
• <run_name>_cm.csv             — confusion matrices in long format (true,pred,count)
• results/viz/*.png             — (optional) confusion heatmaps if enabled
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import csv
import numpy as np

# Headless-safe plotting (only used if you enable heatmaps)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import flwr as fl
except ImportError as e:
    raise ImportError(
        "Flower is not installed. On zsh, quote the spec to avoid globbing:\n"
        "  pip install 'flwr==1.*'"
    ) from e

# --- Project configuration (server knobs, output dirs, label space) ---------
from src_Connection import (
    CLIENTS,              # total number of clients expected to connect
    FREEZE_CFG,           # passed to clients to coordinate unfreeze schedule
    ROUNDS,               # total FL rounds
    RESULTS_DIR,          # where CSVs and viz are stored
    EXPERIMENT,           # run_name to namespace outputs
    N_CLASSES,            # for confusion matrix sizing
    TUNING,               # logging-phase controls (pre/post CV)
    SUPERCLASSES,         # label order for per-class accuracy header
    FL_SERVER_BIND,       # <-- server bind address (e.g., "0.0.0.0:8080")
)
from src_Connection import ensure_dir


# -------------------------------------------------------------------------
# Helper methods for logging
# -------------------------------------------------------------------------
def _append_global_row(server_round: int, acc: float, loss: float) -> None:
    """Append a GLOBAL row with round-level metrics to the results CSV.

    Called from:
      • LoggingFedAvg.aggregate_evaluate(...)

    Writes to:
      • RESULTS_DIR / f"{EXPERIMENT['run_name']}.csv"
        or phase-suffixed file if TUNING['log_mode'] == 'separate'.

    Columns:
      client_id=GLOBAL, round, accuracy, loss, (phase if combined log mode)
    """
    ensure_dir(RESULTS_DIR)
    exp = EXPERIMENT["run_name"]

    if TUNING.get("log_phase"):
        phase = (TUNING["phase_labels"]["enabled"] if TUNING.get("enabled") else
                 TUNING["phase_labels"]["disabled"])
    else:
        phase = ""

    if TUNING.get("log_phase") and TUNING.get("log_mode") == "separate":
        path = RESULTS_DIR / f"{exp}_{phase or 'no_cv'}.csv"
        header = ["client_id","round","accuracy","loss",
                  "frozen_layers","is_frozen","wall_time_sec","trainable_params"]
        row = ["GLOBAL", server_round, f"{acc:.4f}", f"{loss:.4f}", "", "", "", ""]
    else:
        path = RESULTS_DIR / f"{exp}.csv"
        header = ["client_id","round","accuracy","loss",
                  "frozen_layers","is_frozen","wall_time_sec","trainable_params","phase"]
        row = ["GLOBAL", server_round, f"{acc:.4f}", f"{loss:.4f}", "", "", "", "", phase]

    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)


def _append_perclass_row(server_round: int, cm: np.ndarray) -> None:
    """Append per-class accuracy (derived from confusion matrix) for this round.

    Inputs
    ------
    server_round : int
        Current FL round (1-based).
    cm : np.ndarray [N_CLASSES, N_CLASSES]
        Aggregated confusion counts across clients for this round.

    Output CSV
    ----------
    RESULTS_DIR / f"{EXPERIMENT['run_name']}_perclass.csv"
      Columns: ["round"] + ["acc_<class>" for class in SUPERCLASSES]
    """
    ensure_dir(RESULTS_DIR)
    path = RESULTS_DIR / f"{EXPERIMENT['run_name']}_perclass.csv"
    header = ["round"] + [f"acc_{c}" for c in SUPERCLASSES]
    write_header = not path.exists()
    support = cm.sum(axis=1)
    accs = [(cm[i, i] / support[i]) if support[i] > 0 else 0.0 for i in range(len(SUPERCLASSES))]
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([server_round] + [f"{a:.4f}" for a in accs])


def _append_confusion_rows(server_round: int, cm: np.ndarray) -> None:
    """Append confusion matrix in long format: (round, true, pred, count).

    Output CSV
    ----------
    RESULTS_DIR / f"{EXPERIMENT['run_name']}_cm.csv"
      Columns: round, true, pred, count

    Used by:
      • results_visualization.py → confusion heatmaps & last-round CM plots.
    """
    ensure_dir(RESULTS_DIR)
    path = RESULTS_DIR / f"{EXPERIMENT['run_name']}_cm.csv"
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["round", "true", "pred", "count"])
        n = cm.shape[0]
        for i in range(n):
            for j in range(n):
                w.writerow([server_round, i, j, int(cm[i, j])])


def _save_confusion_heatmap(cm: np.ndarray, server_round: int) -> None:
    """(Optional) Save a row-normalized confusion matrix heatmap for this round.

    Note: Disabled by default for speed; uncomment call in LoggingFedAvg to enable.
    Writes PNG to RESULTS_DIR / "viz".
    """
    viz = RESULTS_DIR / "viz"
    ensure_dir(viz)
    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = cm / cm.sum(axis=1, keepdims=True)
        cmn = np.nan_to_num(cmn)
    plt.figure(figsize=(6, 5))
    plt.imshow(cmn, aspect="auto", vmin=0, vmax=1)
    plt.title(f"Confusion Matrix — {EXPERIMENT['run_name']} (round {server_round})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar(label="Row-normalized")
    plt.tight_layout()
    plt.savefig(viz / f"confusion_{EXPERIMENT['run_name']}_r{server_round:02d}.png", dpi=150)
    plt.close()


# -------------------------------------------------------------------------
# Metric aggregation (weighted average over client examples)
# -------------------------------------------------------------------------
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Compute weighted average of client metrics (e.g., accuracy).

    Flower passes per-client evaluation metrics as (num_examples, metrics_dict).

    Parameters
    ----------
    metrics : list of (int, dict)
        For each client: (number of examples used in evaluation, {"accuracy": ...})

    Returns
    -------
    dict
        {"accuracy": weighted_mean} where weights are num_examples.

    Connected to
    ------------
    • LoggingFedAvg(strategy).aggregate_evaluate() via
      'evaluate_metrics_aggregation_fn=weighted_average'.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0}
    avg_accuracy = sum(num_examples * m.get("accuracy", 0.0)
                       for num_examples, m in metrics) / total_examples
    return {"accuracy": float(avg_accuracy)}


# -------------------------------------------------------------------------
# Custom FedAvg strategy with logging hooks
# -------------------------------------------------------------------------
class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that also logs GLOBAL metrics and confusion matrices per round.

    Hook points
    -----------
    • aggregate_evaluate:
        - Calls parent to compute aggregated loss/metrics.
        - Logs GLOBAL (server-side) accuracy/loss to CSV.
        - Aggregates per-class confusion from client metrics and logs both:
            * per-class accuracy row
            * long-format CM

    Inputs from clients
    -------------------
    • Client.evaluate(...) in src_Connection.Client emits:
        - metrics["accuracy"] as float
        - metrics["cm_i_j"] for each (i, j) cell of confusion matrix
    """

    def aggregate_evaluate(self, server_round, results, failures):
        # Let the base FedAvg compute aggregated (loss, metrics)
        agg = super().aggregate_evaluate(server_round, results, failures)
        if agg is not None:
            loss, metrics = agg
            acc = float(metrics.get("accuracy", 0.0))
            _append_global_row(server_round, acc, loss)

        # Aggregate confusion matrix contributed by each client (sum counts)
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
        for _, evalres in results:
            m = evalres.metrics or {}
            for i in range(N_CLASSES):
                for j in range(N_CLASSES):
                    key = f"cm_{i}_{j}"
                    v = m.get(key, None)
                    if v is not None:
                        cm[i, j] += float(v)

        _append_perclass_row(server_round, cm)
        _append_confusion_rows(server_round, cm)
        # If you want per-round images, uncomment below (slower due to PNG I/O):
        # _save_confusion_heatmap(cm, server_round)
        return agg


# -------------------------------------------------------------------------
# FL server launcher (multi-terminal)
# -------------------------------------------------------------------------
def start_server():
    """Start the Flower server and define training strategy for multi-terminal runs.

    What it configures
    ------------------
    • on_fit_config_fn: a function that sends dynamic config to clients each round.
      - We pass 'round' and 'unfreeze_after' (from FREEZE_CFG) to let clients
        coordinate dynamic freezing/unfreezing policies (see Client._apply_freeze_policy).
    • LoggingFedAvg strategy:
      - Client thresholds: min_fit_clients == min_available_clients == CLIENTS
      - Aggregators:
          evaluate_metrics_aggregation_fn = weighted_average
          fit_metrics_aggregation_fn      = lambda mets: {}  (silence unused warning)
    • Server bind address/rounds:
      - server_address = config.FL_SERVER_BIND (e.g., "0.0.0.0:8080")
      - num_rounds     = config.ROUNDS

    Connected files
    ---------------
    • src_Connection.Client (PTBClient):
        - Receives on_fit_config_fn dict each round.
        - Performs local fit/eval and returns metrics.
    • src_Connection.Experiments:
        - The `federated` subcommand can spawn this server and multiple clients.
    """

    def on_fit_config_fn(rnd: int) -> Dict[str, int]:
        # Sent to clients each round; clients can use 'round' and schedule logic
        return {"round": rnd, "unfreeze_after": int(FREEZE_CFG["unfreeze_after"])}

    strategy = LoggingFedAvg(
        min_fit_clients=int(CLIENTS),        # require all clients for fit
        min_available_clients=int(CLIENTS),  # block until all clients are connected
        evaluate_metrics_aggregation_fn=weighted_average,  # global accuracy = weighted mean
        fit_metrics_aggregation_fn=lambda mets: {},        # ignore fit metrics aggregation
        on_fit_config_fn=on_fit_config_fn,                 # per-round config to clients
    )

    addr = str(FL_SERVER_BIND)  # e.g., "0.0.0.0:8080" from config.py
    print(f"[Server] Starting Flower @ {addr}  |  rounds={int(ROUNDS)}  |  clients={int(CLIENTS)}")
    fl.server.start_server(
        server_address=addr,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=int(ROUNDS)),
    )


# -------------------------------------------------------------------------
# Entry point (multi-terminal)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    start_server()