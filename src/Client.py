"""
Client.py — PTB-XL Federated Learning Client
--------------------------------------------

Implements a single federated learning (FL) client for ECG classification
under the Sustainable AI in Healthcare project (DSP5100).

Each client:
  • Loads its local patient subset from PTB-XL (unevenly partitioned)
  • Builds a CNN or LSTM model depending on config.py
  • Trains locally using FedAvg (with optional FedProx term)
  • Applies dynamic layer freezing to save compute
  • Logs per-round metrics (accuracy, loss, time, #trainable params)

This setup simulates data heterogeneity and sustainability-oriented
training efficiency.

"""

from __future__ import annotations
from collections import OrderedDict
from typing import List, Tuple
from dataclasses import dataclass

import argparse
import csv
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl

from src.config import (
    LR, BATCH_SIZE, EPOCHS_LOCAL, FREEZE_THRESHOLD, FEDPROX_MU,
    SEED, RESULTS_DIR, FREEZE_CFG, SPLITS, NORM, MODEL
)
from src.data_loader import (
    load_metadata, map_superclasses, filter_single_label,
    stratified_patient_split_3way, load_waveform,
    compute_perlead_norm_stats, normalize_signal, stratified_patient_split
)
from src.models import create_model
from src.utils import set_seed



# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def make_tensor_dataset(df, mu=None, sigma=None, eps=1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a dataframe of PTB-XL rows into (X, y) tensors."""
    X, y = [], []
    classes = ["NORM", "MI", "STTC", "HYP", "CD"]
    for _, row in df.iterrows():
        sig = load_waveform(row)  # (12, T)
        if mu is not None and sigma is not None and NORM["enabled"]:
            sig = normalize_signal(sig, mu, sigma, eps=NORM.get("eps", 1e-6))
        X.append(sig)
        y.append(classes.index(row["y"]))
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)
    return X, y


def get_loaders(df, batch_size=BATCH_SIZE, mu=None, sigma=None):
    """Return DataLoader for given dataframe."""
    X, y = make_tensor_dataset(df, mu=mu, sigma=sigma)
    ds = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


# -------------------------------------------------------------------------
# Freeze state dataclass
# -------------------------------------------------------------------------
@dataclass
class FreezeState:
    """Tracks per-client improvement-gated freezing dynamics."""
    best_loss: float = float("inf")
    no_improve: int = 0
    patience: int = FREEZE_CFG["patience"]
    min_delta: float = FREEZE_CFG["min_delta"]
    last_frozen: int = 0


# -------------------------------------------------------------------------
# Federated client implementation
# -------------------------------------------------------------------------
class PTBClient(fl.client.NumPyClient):
    """Federated learning client for PTB-XL."""

    def __init__(self, cid: int):
        # --- Basic setup --------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cid = cid
        set_seed(SEED + cid)
        ptb = load_metadata()
        df = filter_single_label(map_superclasses(ptb))

        # --- Global 70/15/15 split (patient-wise) -------------------------
        train_global, val_global, test_global = stratified_patient_split_3way(
            df, splits=(SPLITS["train"], SPLITS["val"], SPLITS["test"]), seed=SEED
        )

        # --- Partition clients unevenly ----------------------------------
        patients = train_global.patient_id.unique()
        np.random.seed(SEED)
        np.random.shuffle(patients)
        ratios = np.array([0.5, 0.33, 0.15, 0.02])
        ratios /= ratios.sum()
        sizes = [int(r * len(patients)) for r in ratios]
        clients, start = [], 0
        for s in sizes:
            pids = patients[start:start + s]
            clients.append(train_global[train_global.patient_id.isin(pids)])
            start += s

        self.train_df = clients[int(cid)]

        # --- Local validation set ----------------------------------------
        val_candidates = val_global[val_global.patient_id.isin(self.train_df.patient_id.unique())]
        if len(val_candidates) >= 1:
            self.val_df = val_candidates
        else:
            local_train, local_val = stratified_patient_split(self.train_df, test_size=0.15, seed=SEED)
            self.train_df, self.val_df = local_train, local_val

        # --- Global test set ---------------------------------------------
        self.test_df = test_global

        # --- Normalization statistics ------------------------------------
        self.mu, self.sigma = compute_perlead_norm_stats(self.train_df)

        # --- Model setup --------------------------------------------------
        if MODEL["type"] == "cnn":
            self.model = create_model(5, model_type="cnn").to(self.device)
        else:
            self.model = create_model(
                5, model_type="lstm",
                hidden=MODEL.get("lstm_hidden", 128),
                layers=MODEL.get("lstm_layers", 1),
                bidir=MODEL.get("bidirectional", True),
                use_stem=False,
            ).to(self.device)
        self.ce = nn.CrossEntropyLoss()

        # Optional initial freezing for small clients
        if len(self.train_df) < FREEZE_THRESHOLD:
            for p in self.model.features.parameters():
                p.requires_grad = False

        # Initialize improvement-tracking state
        self.state = FreezeState()

        print(f"[Client {self.cid}] initialized with {len(self.train_df)} records.")

    # ---------------------------------------------------------------------
    # Flower interface methods
    # ---------------------------------------------------------------------
    def get_parameters(self, config):
        """Return model weights as a list of numpy arrays."""
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Load global weights into local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    # ---------------------------------------------------------------------
    # Freezing utilities
    # ---------------------------------------------------------------------
    def _base_freeze_target(self, n_train, total_layers, round_num):
        """Baseline freezing rule: fewer samples → more frozen layers."""
        if n_train >= 2 * FREEZE_THRESHOLD:
            return 0
        if n_train < FREEZE_THRESHOLD:
            return int(total_layers * max(0.0, 1 - round_num / FREEZE_CFG["unfreeze_after"]))
        return total_layers // 2

    def _apply_freeze_policy(self, round_num, n_train, total_layers):
        """Combine base freeze with improvement-gated adjustment."""
        base = self._base_freeze_target(n_train, total_layers, round_num)
        gated = max(0, base - self.state.no_improve)
        n_to_freeze = min(gated, total_layers - 1)
        for i, layer in enumerate(self.model.features):
            for p in layer.parameters():
                p.requires_grad = (i >= n_to_freeze)
        return n_to_freeze

    # ---------------------------------------------------------------------
    # Local training
    # ---------------------------------------------------------------------
    def fit(self, parameters, config):
        """Perform one federated training round."""
        self.set_parameters(parameters)
        round_num = config.get("round", 1)

        total_layers = len(list(self.model.features))
        n_train = len(self.train_df)
        frozen_layers = self._apply_freeze_policy(round_num, n_train, total_layers)

        print(f"[Client {self.cid}] Round {round_num}: {n_train} samples, frozen_layers={frozen_layers}/{total_layers}")

        # --- Local training loop ----------------------------------------
        t0 = time.perf_counter()
        train_loader = get_loaders(self.train_df, mu=self.mu, sigma=self.sigma)
        opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                         lr=LR, weight_decay=1e-4)

        # FedProx global weights snapshot
        global_params = [p.detach().clone() for p in self.model.parameters()]

        self.model.train()
        for _ in range(EPOCHS_LOCAL):
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = self.ce(logits, yb)
                if FEDPROX_MU > 0:
                    prox = sum(torch.sum((w - w0) ** 2)
                               for w, w0 in zip(self.model.parameters(), global_params))
                    loss += (FEDPROX_MU / 2.0) * prox
                loss.backward()
                opt.step()

        wall_s = time.perf_counter() - t0
        loss_local, acc_local = self.evaluate_local()

        # --- Update freeze-gating state ---------------------------------
        if self.state.best_loss - loss_local > self.state.min_delta:
            self.state.best_loss = loss_local
            self.state.no_improve = 0
        else:
            self.state.no_improve += 1

        # --- Log and print ----------------------------------------------
        self._log_local_metrics_csv(round_num, acc_local, loss_local,
                                    frozen_layers, total_layers, wall_s)
        print(f"[Client {self.cid}] R{round_num}: acc={acc_local:.4f} "
              f"loss={loss_local:.4f} time={wall_s:.1f}s "
              f"freeze={frozen_layers}/{total_layers} "
              f"(no_improve={self.state.no_improve})")

        return self.get_parameters({}), len(self.train_df), {"local_time": wall_s}

    # ---------------------------------------------------------------------
    # Evaluation helpers
    # ---------------------------------------------------------------------
    def evaluate_local(self) -> Tuple[float, float]:
        """Evaluate model on local validation set."""
        loader = get_loaders(self.val_df, batch_size=128, mu=self.mu, sigma=self.sigma)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss_sum += float(self.ce(logits, yb).item()) * len(yb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum())
                total += len(yb)
                mean_loss = loss_sum / total
                acc = correct / total
        return loss_sum / total, correct / total

    def evaluate(self, parameters, config):
        """Evaluate on local test data (used by Flower server)."""
        self.set_parameters(parameters)
        loader = get_loaders(self.test_df, batch_size=128, mu=self.mu, sigma=self.sigma)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = self.ce(logits, yb)
                loss_sum += float(loss.item()) * len(yb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(len(yb))
        return float(loss_sum / total), total, {"accuracy": correct / total}

    # ---------------------------------------------------------------------
    # Logging utilities
    # ---------------------------------------------------------------------
    def _log_local_metrics_csv(self, round_num: int, acc: float, loss: float,
                               frozen_layers: int, total_layers: int, wall_s: float):
        """Append per-round metrics to CSV."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = "frozen_client_metrics.csv" if frozen_layers > 0 else "not_frozen_client_metrics.csv"
        path = RESULTS_DIR / filename
        write_header = not path.exists()
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["client_id", "round", "local_accuracy", "local_loss",
                            "frozen_layers", "total_layers", "wall_time_sec",
                            "trainable_params"])
            w.writerow([self.cid, round_num, f"{acc:.4f}", f"{loss:.4f}",
                        frozen_layers, total_layers, f"{wall_s:.2f}", trainable_params])


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID (0..3)")
    args = parser.parse_args()
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=PTBClient(args.cid))


if __name__ == "__main__":
    main()