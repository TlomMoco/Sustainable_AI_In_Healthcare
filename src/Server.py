"""
Server.py — Federated Learning Orchestrator
-------------------------------------------

Coordinates the federated training process across multiple PTB-XL clients.

Launch:
    $ python -m src.Server
Then start each client in separate terminals:
    $ python -m src.Client --cid 0
    ... --cid 1
    ... --cid 2
    ... --cid 3
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np

from src.config import (
    CLIENTS, FREEZE_CFG, ROUNDS, RESULTS_DIR, EXPERIMENT,
    N_CLASSES, TUNING, SUPERCLASSES
)
from src.utils import append_csv_locked


# -------------------------------------------------------------------------
# Phase helper (consistent with client logging)
# -------------------------------------------------------------------------
def _current_phase() -> str:
    """Return 'tuned', 'default', or '' based on config, mirroring client logging."""
    if TUNING.get("log_phase"):
        if TUNING.get("enabled"):
            return TUNING["phase_labels"]["enabled"]   # "tuned"
        elif TUNING.get("use_cached_best"):
            return TUNING["phase_labels"]["cached"]    # "tuned" alias
        else:
            return TUNING["phase_labels"]["disabled"]  # "default"
    return ""


# -------------------------------------------------------------------------
# FL server launcher
# -------------------------------------------------------------------------
def start_server():
    """Start the Flower server and define training strategy (wait for all clients)."""

    def on_fit_config_fn(rnd: int) -> Dict[str, int]:
        """Provide configuration to clients for each round."""
        return {"round": rnd, "unfreeze_after": FREEZE_CFG["unfreeze_after"]}

    # Use all clients every round; fail if any are missing
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

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
    )


# -------------------------------------------------------------------------
# Global results logging
# -------------------------------------------------------------------------
def _append_global_row(server_round: int, acc: float, loss: float) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    exp = EXPERIMENT["run_name"]
    phase = _current_phase()

    # If separate mode: split files by phase in the filename.
    # If same mode: write a single CSV and include 'phase' column per row.
    separate = (TUNING.get("log_phase") and TUNING.get("log_mode") == "separate")
    if separate:
        path = RESULTS_DIR / f"{exp}_{phase or 'no_cv'}.csv"
        fieldnames = ["client_id","round","accuracy","loss",
                      "frozen_layers","is_frozen","wall_time_sec","trainable_params"]
        row = {
            "client_id":"GLOBAL","round":server_round,"accuracy":f"{acc:.4f}","loss":f"{loss:.4f}",
            "frozen_layers":"","is_frozen":"","wall_time_sec":"","trainable_params":""
        }
    else:
        path = RESULTS_DIR / f"{exp}.csv"
        fieldnames = ["client_id","round","accuracy","loss",
                      "frozen_layers","is_frozen","wall_time_sec","trainable_params","phase"]
        row = {
            "client_id":"GLOBAL","round":server_round,"accuracy":f"{acc:.4f}","loss":f"{loss:.4f}",
            "frozen_layers":"","is_frozen":"","wall_time_sec":"","trainable_params":"","phase":phase
        }

    append_csv_locked(path, row, fieldnames)


def _append_perclass_row(server_round: int, cm: np.ndarray) -> None:
    """Write per-class accuracies; separate files by phase to avoid mixing."""
    exp = EXPERIMENT["run_name"]
    phase = _current_phase()
    path = RESULTS_DIR / f"{exp}_perclass_{phase or 'no_cv'}.csv"

    fieldnames = ["round"] + [f"acc_{c}" for c in SUPERCLASSES]
    support = cm.sum(axis=1)
    accs = [(cm[i, i] / support[i]) if support[i] > 0 else 0.0 for i in range(len(SUPERCLASSES))]
    row = {"round": server_round, **{f"acc_{c}": f"{a:.4f}" for c, a in zip(SUPERCLASSES, accs)}}
    append_csv_locked(path, row, fieldnames)


def _append_confusion_rows(server_round: int, cm: np.ndarray) -> None:
    """Write confusion matrix rows; separate files by phase to avoid mixing."""
    exp = EXPERIMENT["run_name"]
    phase = _current_phase()
    path = RESULTS_DIR / f"{exp}_cm_{phase or 'no_cv'}.csv"

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

        # Aggregate confusion matrix across clients (sum)
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