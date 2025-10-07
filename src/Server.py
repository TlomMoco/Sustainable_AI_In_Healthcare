from __future__ import annotations
import flwr as fl
from typing import Dict, List, Tuple

# aggregate client metrics (weighted by number of examples)
def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    # metrics is a list of (num_examples, {"accuracy": value}) tuples
    total_examples = sum(num_examples for num_examples, _ in metrics)
    avg_accuracy = (
        sum(num_examples * m["accuracy"] for num_examples, m in metrics) / total_examples
    )
    return {"accuracy": avg_accuracy}


def start_server():
    def on_fit_config_fn(rnd: int):
        return {"round": rnd, "unfreeze_after": 3}  # unfreeze after 3 rounds

    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=on_fit_config_fn,
    )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=6),
    )


if __name__ == "__main__":
    start_server()