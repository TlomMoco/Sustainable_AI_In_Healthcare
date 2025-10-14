"""
Server.py — Federated Learning Orchestrator
-------------------------------------------

Coordinates the federated training process across multiple PTB-XL clients.

Responsibilities:
  • Initialize the Flower (flwr) federated averaging server
  • Configure each training round (number, unfreezing schedule, etc.)
  • Aggregate metrics across clients (weighted by number of examples)

Launch:
    $ python -m src.Server

Then start each client in separate terminals:
    $ python -m src.Client --cid 0
    $ python -m src.Client --cid 1
    $ python -m src.Client --cid 2
    $ python -m src.Client --cid 3
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import matplotlib.pyplot as plt  # kept for parity with your environment

from src.config import (
    CLIENTS, FREEZE_CFG, ROUNDS, RESULTS_DIR, EXPERIMENT,
    N_CLASSES, TUNING, SUPERCLASSES
)
from src.utils import append_csv_locked

# -------------------------------------------------------------------------
# Global results logging
# -------------------------------------------------------------------------
GLOBAL_ROW = {
    "client_id": "GLOBAL",   # matches client csv schema but marks global row
    "frozen_layers": "",     # not meaningful at global scope
    "is_frozen": "",
    "wall_time_sec": "",
    "trainable_params": "",
}


# -------------------------------------------------------------------------
# FL server launcher
# -------------------------------------------------------------------------
def start_server():
    """Start the Flower server and define training strategy (wait for all clients)."""

    def on_fit_config_fn(rnd: int) -> Dict[str, int]:
        """Provide configuration to clients for each round."""
        return {"round": rnd, "unfreeze_after": FREEZE_CFG["unfreeze_after"]}

    # --- Federated averaging strategy (WAIT FOR ALL CLIENTS) -------------
    # fraction_* = 1.0 => try to use all available clients
    # min_available_clients = CLIENTS => do not start until all are connected
    # min_fit/val clients = CLIENTS => every round needs all clients
    # accept_failures = False => if a client drops, the round fails instead of silently proceeding
    strategy = LoggingFedAvg(
        fraction_fit=1.0,
        min_fit_clients=CLIENTS,
        fraction_evaluate=1.0,
        min_evaluate_clients=CLIENTS,
        min_available_clients=CLIENTS,
        accept_failures=False,

        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=lambda mets: {},  # silence warning
        on_fit_config_fn=on_fit_config_fn,
    )

    # --- Launch the server -----------------------------------------------
    # Note: We do not set a round_timeout so the server patiently waits for all clients.
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
    )


# -------------------------------------------------------------------------
# Helper methods for logging
# -------------------------------------------------------------------------

def _append_global_row(server_round: int, acc: float, loss: float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exp = EXPERIMENT["run_name"]

    # Compute phase consistent with clients
    if TUNING.get("log_phase"):
        if TUNING.get("enabled"):
            phase = TUNING["phase_labels"]["enabled"]    # "post_cv"
        elif TUNING.get("use_cached_best"):
            phase = TUNING["phase_labels"]["cached"]     # "cached_cv"
        else:
            phase = TUNING["phase_labels"]["disabled"]   # "no_cv"
    else:
        phase = ""

    separate = (TUNING.get("log_phase") and TUNING.get("log_mode") == "separate")
    if separate:
        path = RESULTS_DIR / f"{exp}_{phase or 'no_cv'}.csv"
        fieldnames = ["client_id","round","accuracy","loss",
                      "frozen_layers","is_frozen","wall_time_sec","trainable_params"]
        row = {"client_id":"GLOBAL","round":server_round,"accuracy":f"{acc:.4f}","loss":f"{loss:.4f}",
               "frozen_layers":"","is_frozen":"","wall_time_sec":"","trainable_params":""}
    else:
        path = RESULTS_DIR / f"{exp}.csv"
        fieldnames = ["client_id","round","accuracy","loss",
                      "frozen_layers","is_frozen","wall_time_sec","trainable_params","phase"]
        row = {"client_id":"GLOBAL","round":server_round,"accuracy":f"{acc:.4f}","loss":f"{loss:.4f}",
               "frozen_layers":"","is_frozen":"","wall_time_sec":"","trainable_params":"","phase":phase}

    append_csv_locked(path, row, fieldnames)


def _append_perclass_row(server_round: int, cm: np.ndarray) -> None:
    path = RESULTS_DIR / f"{EXPERIMENT['run_name']}_perclass.csv"
    fieldnames = ["round"] + [f"acc_{c}" for c in SUPERCLASSES]
    support = cm.sum(axis=1)
    accs = [(cm[i, i] / support[i]) if support[i] > 0 else 0.0 for i in range(len(SUPERCLASSES))]
    row = {"round": server_round, **{f"acc_{c}": f"{a:.4f}" for c, a in zip(SUPERCLASSES, accs)}}
    append_csv_locked(path, row, fieldnames)


def _append_confusion_rows(server_round: int, cm: np.ndarray) -> None:
    path = RESULTS_DIR / f"{EXPERIMENT['run_name']}_cm.csv"
    fieldnames = ["round", "true", "pred", "count"]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            row = {"round": server_round, "true": i, "pred": j, "count": int(cm[i, j])}
            append_csv_locked(path, row, fieldnames)


# -------------------------------------------------------------------------
# Custom FedAvg strategy with logging
# -------------------------------------------------------------------------
class LoggingFedAvg(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, server_round, results, failures):
        agg = super().aggregate_evaluate(server_round, results, failures)
        if agg is not None:
            loss, metrics = agg
            acc = float(metrics.get("accuracy", 0.0))
            _append_global_row(server_round, acc, loss)

        # Aggregate confusion matrix across clients
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.float64)
        for _, evalres in results:
            m = evalres.metrics or {}
            for i in range(N_CLASSES):
                for j in range(N_CLASSES):
                    key = f"cm_{i}_{j}"
                    if key in m:
                        cm[i, j] += float(m[key])

        _append_perclass_row(server_round, cm)
        _append_confusion_rows(server_round, cm)
        return agg


# -------------------------------------------------------------------------
# Metric aggregation
# -------------------------------------------------------------------------
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Compute weighted average of client metrics.

    Args:
        metrics: List of (num_examples, {"accuracy": value}) pairs from clients.

    Returns:
        Dict[str, float]: Weighted average accuracy.
    """
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {"accuracy": 0.0}

    avg_accuracy = sum(num_examples * m.get("accuracy", 0.0)
                       for num_examples, m in metrics) / total_examples
    return {"accuracy": avg_accuracy}


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    start_server()