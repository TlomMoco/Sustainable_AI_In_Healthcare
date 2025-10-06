from __future__ import annotations
import flwr as fl



def start_server():
    strategy = fl.server.strategy.FedAvg(min_fit_clients=2, min_available_clients=2)
    fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy, config=fl.server.ServerConfig(num_rounds=3))


if __name__ == "__main__":
    start_server()