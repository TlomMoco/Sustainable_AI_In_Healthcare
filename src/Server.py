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

import csv
import flwr as fl
import numpy as np
import matplotlib.pyplot as plt
from src.config import CLIENTS, FREEZE_CFG, ROUNDS, RESULTS_DIR, EXPERIMENT, N_CLASSES, TUNING, SUPERCLASSES

# -------------------------------------------------------------------------
# Global results logging
# -------------------------------------------------------------------------
GLOBAL_ROW = {
    "client_id": "GLOBAL",          # matches client csv schema but marks global row
    "frozen_layers": "",             # not meaningful at global scope
    "is_frozen": "",
    "wall_time_sec": "",
    "trainable_params": "",
}


# -------------------------------------------------------------------------
# FL server launcher
# -------------------------------------------------------------------------
def start_server():
    """Start the Flower server and define training strategy."""

    def on_fit_config_fn(rnd: int) -> Dict[str, int]:
        """Provide configuration to clients for each round."""
        return {"round": rnd, "unfreeze_after": FREEZE_CFG["unfreeze_after"]}

    # Federated averaging strategy
    strategy = LoggingFedAvg(
        min_fit_clients=CLIENTS,                    # number of clients per round
        min_available_clients=CLIENTS,              # required clients online
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=lambda mets: {}, # no-op, silences warning
        on_fit_config_fn=on_fit_config_fn,          # send round config
    )

    # Launch the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=ROUNDS),
    )


# -------------------------------------------------------------------------
# Helper methods for logging
# -------------------------------------------------------------------------

# --- Append a row to the global results CSV ------------------------------
def _append_global_row(server_round: int, acc: float, loss: float) -> None:
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

# --- Append per-class accuracy and confusion matrix -----------------------
def _append_perclass_row(server_round: int, cm: np.ndarray) -> None:
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

# --- Append confusion matrix in long format ------------------------------
def _append_confusion_rows(server_round: int, cm: np.ndarray) -> None:
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

# --- Save confusion matrix heatmap --------------------------------------
def _save_confusion_heatmap(cm: np.ndarray, server_round: int) -> None:
    viz = RESULTS_DIR / "viz"
    viz.mkdir(parents=True, exist_ok=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        cmn = cm / cm.sum(axis=1, keepdims=True)
        cmn = np.nan_to_num(cmn)
    plt.figure(figsize=(6, 5))
    plt.imshow(cmn, aspect="auto")
    plt.title(f"Confusion Matrix — {EXPERIMENT['run_name']} (round {server_round})")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar(label="Row-normalized")
    plt.tight_layout()
    plt.savefig(viz / f"confusion_{EXPERIMENT['run_name']}_r{server_round:02d}.png", dpi=150)
    plt.close()


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








# ---------
# ## 19) Federated Learning — Flower Server & Orchestration (from notebook; adapted)
# ---------
from __future__ import annotations
import numpy as np, pandas as pd, torch
try:
    import flwr as fl
except ImportError:
    raise ImportError("Please install Flower: pip install flwr==1.*")

from . import config as CFG
from .utils import set_seed, pick_device, log
from .data_loader import make_feature_table, train_test_split
from .data_preprocessing import make_label_encoder, ECGDataset
from .Client import make_client
from .models import create_model
from .results_visualization import TorchAdapter, evaluate_models

def _partition_clients(features_df, n_clients: int, mode: str = "by_patient", alpha: float = 0.3, min_per=25):
    """Returns list of index arrays per client (train subset only)."""
    rng = np.random.RandomState(CFG.SEED)
    train_mask, _ = train_test_split(features_df)
    idx_all = features_df.index[train_mask]

    y_train = features_df.loc[idx_all, "label"].astype(str)
    paths_train = features_df.loc[idx_all, "record_path"].astype(str)

    # simple patient grouping
    if mode == "by_patient" and "patient_id" in features_df.columns:
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, pid in zip(idx_all, features_df.loc[idx_all, "patient_id"]):
            groups[int(pid)].append(int(idx))
        buckets = [[] for _ in range(n_clients)]
        sizes = [0]*n_clients
        for _, gidx in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            j = int(np.argmin(sizes)); buckets[j].extend(gidx); sizes[j] += len(gidx)
    elif mode == "iid":
        shuf = idx_all.to_numpy().copy(); rng.shuffle(shuf)
        buckets = [list(x) for x in np.array_split(shuf, n_clients)]
    else:
        # Dirichlet over labels
        buckets = [[] for _ in range(n_clients)]
        for lab in sorted(y_train.unique()):
            idx_lab = np.array(y_train[y_train == lab].index, dtype=int)
            rng.shuffle(idx_lab)
            props = rng.dirichlet([alpha]*n_clients)
            counts = (props * len(idx_lab)).astype(int)
            while counts.sum() < len(idx_lab): counts[rng.randint(0, n_clients)] += 1
            pos = 0
            for k in range(n_clients):
                take = counts[k]
                if take > 0:
                    buckets[k].extend(idx_lab[pos:pos+take].tolist()); pos += take
    # enforce min per client
    flat = [i for lst in buckets for i in lst]
    need = [(k, min_per - len(lst)) for k, lst in enumerate(buckets) if len(lst) < min_per and min_per>0]
    if need:
        pool = [i for i in flat]
        buckets = [[] for _ in range(n_clients)]
        rng.shuffle(pool)
        for k, req in need:
            take = min(req, len(pool))
            buckets[k].extend(pool[:take]); pool = pool[take:]
        for j, x in enumerate(pool):
            buckets[j % n_clients].append(x)
    return buckets

def start_server():
    # Flower FedAvg server
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=CFG.FL_SAMPLE_FRAC,
        min_fit_clients=CFG.FL_N_CLIENTS,
        min_available_clients=CFG.FL_N_CLIENTS,
        on_fit_config_fn=lambda r: {"local_epochs": CFG.FL_LOCAL_EPOCHS},
        on_evaluate_config_fn=lambda r: {},
    )
    fl.server.start_server(server_address=CFG.FL_SERVER_ADDRESS, strategy=strategy, config=fl.server.ServerConfig(num_rounds=CFG.FL_ROUNDS))

def start_simulation():
    # Lightweight simulation using Flower's VirtualClientEngine
    feature_df, features_df = make_feature_table(save_csv=False)
    train_mask, test_mask = train_test_split(features_df)
    le = make_label_encoder(features_df.loc[train_mask, "label"], features_df.loc[test_mask, "label"])
    n_classes = len(le.classes_)
    device = pick_device()

    buckets = _partition_clients(features_df, CFG.FL_N_CLIENTS, mode=CFG.FL_PARTITION, alpha=CFG.FL_DIRICHLET_ALPHA, min_per=CFG.FL_MIN_SAMPLES_PER_CLIENT)
    clients = []
    for k, idxs in enumerate(buckets):
        idxs = np.array(idxs, dtype=int)
        paths = features_df.loc[idxs, "record_path"].astype(str).values
        labs  = le.transform(features_df.loc[idxs, "label"].astype(str).values)
        # simple 90/10 split per client
        cut = max(1, int(0.9 * len(idxs)))
        train_paths, val_paths = paths[:cut], paths[cut:]
        y_tr, y_va = labs[:cut], labs[cut:]
        client = make_client(CFG.FL_BASE_MODEL, n_classes, train_paths, y_tr, val_paths, y_va, device=device)
        clients.append(client)

    # Init global weights from fresh model
    global_model = create_model(CFG.FL_BASE_MODEL, n_classes, binary=(n_classes==2)).to(device)
    init_params = [p.detach().cpu().numpy() for _, p in global_model.state_dict().items()]

    def client_fn(cid: str):
        # map id to prebuilt client
        return clients[int(cid)]

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=CFG.FL_SAMPLE_FRAC,
        min_fit_clients=len(clients),
        min_available_clients=len(clients),
        on_fit_config_fn=lambda r: {"local_epochs": CFG.FL_LOCAL_EPOCHS},
        initial_parameters=fl.common.ndarrays_to_parameters(init_params),
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=CFG.FL_ROUNDS),
        strategy=strategy,
    )

    # Evaluate global model after sim by aggregating latest client params (use best client params heuristic)
    # (Optional: load the last returned weights into 'global_model')

    test_paths = features_df.loc[test_mask, "record_path"].astype(str).values
    y_test     = features_df.loc[test_mask, "label"].astype(str).values
    from .results_visualization import TorchAdapter, evaluate_models
    adapters = {"FL FedAvg Model": TorchAdapter(global_model, le, (n_classes==2), batch=CFG.BATCH_SIZE, device=device)}
    print(evaluate_models(adapters, test_paths, y_test, le, (n_classes==2), device))

if __name__ == "__main__":
    # You can run either a real server or the local simulation:
    # start_server()
    start_simulation()
