"""
Server.py — Federated Learning Orchestrator
-------------------------------------------

Coordinates the federated training process across multiple PTB-XL clients.

Responsibilities:
  • Initialize the Flower (flwr) federated averaging server
  • Configure each training round (number, unfreezing schedule, etc.)
  • Aggregate metrics across clients (weighted by number of examples)

This script should be launched first in a terminal:
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
from src.config import CLIENTS, FREEZE_CFG


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
# FL server launcher
# -------------------------------------------------------------------------
def start_server():
    """Start the Flower server and define training strategy."""

    def on_fit_config_fn(rnd: int) -> Dict[str, int]:
        """Provide configuration to clients for each round."""
        return {"round": rnd, "unfreeze_after": FREEZE_CFG["unfreeze_after"]}

    # Federated averaging strategy
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=CLIENTS,                    # number of clients per round
        min_available_clients=CLIENTS,              # required clients online
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=lambda mets: {}, # no-op, silences warning
        on_fit_config_fn=on_fit_config_fn,          # send round config
    )

    # Launch the server (default 20 rounds)
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=20),
    )


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    start_server()