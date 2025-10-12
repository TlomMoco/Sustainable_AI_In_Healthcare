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

from src_Connection.config import (
    CLIENTS,
    FREEZE_CFG,
    ROUNDS,
    RESULTS_DIR,
    EXPERIMENT,
    N_CLASSES,
    TUNING,
    SUPERCLASSES,
    FL_SERVER_ADDRESS,  # use address from config
)


# -------------------------------------------------------------------------
# Helper methods for logging
# -------------------------------------------------------------------------
def _append_global_row(server_round: int, acc: float, loss: float) -> None:
    """Append a GLOBAL row with round-level metrics to the results CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
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
    """Append per-class accuracy (derived from confusion matrix) for this round."""
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
    """Append confusion matrix rows in long format (true, pred, count)."""
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
    """(Optional) Save a row-normalized confusion matrix heatmap for this round."""
    viz = RESULTS_DIR / "viz"
    viz.mkdir(parents=True, exist_ok=True)
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
    """
    Compute weighted average of client metrics.
    Args:
        metrics: List of (num_examples, {"accuracy": value, ...}) pairs.
    Returns:
        Dict[str, float]: Weighted average metrics (currently accuracy only).
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
    """FedAvg that also logs GLOBAL metrics and confusion matrices per round."""

    def aggregate_evaluate(self, server_round, results, failures):
        agg = super().aggregate_evaluate(server_round, results, failures)
        if agg is not None:
            loss, metrics = agg
            acc = float(metrics.get("accuracy", 0.0))
            _append_global_row(server_round, acc, loss)

        # Aggregate confusion matrix contributed by each client
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
        # If you want per-round images, uncomment below (slower):
        # _save_confusion_heatmap(cm, server_round)
        return agg


# -------------------------------------------------------------------------
# FL server launcher (multi-terminal)
# -------------------------------------------------------------------------
def start_server():
    """Start the Flower server and define training strategy for multi-terminal runs."""

    def on_fit_config_fn(rnd: int) -> Dict[str, int]:
        # Sent to clients each round; clients can use 'round' and schedule logic
        return {"round": rnd, "unfreeze_after": int(FREEZE_CFG["unfreeze_after"])}

    strategy = LoggingFedAvg(
        min_fit_clients=int(CLIENTS),
        min_available_clients=int(CLIENTS),
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=lambda mets: {},  # silence fit-metrics warning
        on_fit_config_fn=on_fit_config_fn,
    )

    addr = str(FL_SERVER_ADDRESS)  # e.g., "127.0.0.1:8080" from config.py
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