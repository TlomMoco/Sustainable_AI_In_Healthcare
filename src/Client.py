from __future__ import annotations
from collections import OrderedDict
from typing import List, Tuple

import argparse
import csv, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from dataclasses import dataclass

from config import (
    LR, BATCH_SIZE, EPOCHS_LOCAL, FREEZE_THRESHOLD, FEDPROX_MU, SEED, RESULTS_DIR, FREEZE_CFG,
    SPLITS, NORM, MODEL
)
from data_loader import (
    load_metadata, map_superclasses, filter_single_label,
    stratified_patient_split_3way, load_waveform,
    compute_perlead_norm_stats, normalize_signal, stratified_patient_split
)
from models import create_model
from utils import set_seed


def make_tensor_dataset(df, mu=None, sigma=None, eps=1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Materialize a (TensorDataset) from a dataframe of PTB-XL rows.
    Applies per-lead z-score with (mu,sigma) if provided.
    """
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
    X, y = make_tensor_dataset(df, mu=mu, sigma=sigma)
    ds = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


@dataclass
class FreezeState:
    """Tracks per-client improvement-gated freezing dynamics."""
    best_loss: float = float("inf")
    no_improve: int = 0
    patience: int = FREEZE_CFG["patience"]
    min_delta: float = FREEZE_CFG["min_delta"]
    last_frozen: int = 0

class PTBClient(fl.client.NumPyClient):
    def __init__(self, cid: int):
        ptb = load_metadata()
        df = filter_single_label(map_superclasses(ptb))

        # Global 70/15/15 by patient
        train_global, val_global, test_global = stratified_patient_split_3way(
            df, splits=(SPLITS["train"], SPLITS["val"], SPLITS["test"]), seed=SEED
        )

        # Partition clients from *global train* patients (uneven)
        patients = train_global.patient_id.unique()
        np.random.seed(SEED);
        np.random.shuffle(patients)
        ratios = np.array([0.5, 0.33, 0.15, 0.02]);
        ratios = ratios / ratios.sum()
        sizes = [int(r * len(patients)) for r in ratios]
        clients = [];
        start = 0
        for s in sizes:
            pids = patients[start:start + s]
            clients.append(train_global[train_global.patient_id.isin(pids)])
            start += s

        self.train_df = clients[int(cid)]
        # Use a *local* validation set (drawn from global val but restricted to this client's patients if any)
        # If the client's patients don't intersect global val (likely), just take a 15% patient-level slice from its train.
        val_candidates = val_global[val_global.patient_id.isin(self.train_df.patient_id.unique())]
        if len(val_candidates) >= 1:
            self.val_df = val_candidates
        else:
            # fallback: split local train by patient for validation
            local_train, local_val = stratified_patient_split(self.train_df, test_size=0.15, seed=SEED)
            self.train_df, self.val_df = local_train, local_val

        # Keep test as global test (same for all clients) but *sample* for speed if needed
        self.test_df = test_global  # or test_global.sample(frac=0.25, random_state=SEED)

        # --- normalization stats from *local train only* ---
        self.mu, self.sigma = compute_perlead_norm_stats(self.train_df)

        # --- Model setup from config ---
        if MODEL["type"] == "cnn":
            self.model = create_model(5, model_type="cnn")
        else:
            self.model = create_model(
                5, model_type="lstm",
                hidden=MODEL.get("lstm_hidden", 128),
                layers=MODEL.get("lstm_layers", 1),
                bidir=MODEL.get("bidirectional", True),
                use_stem=False,  # you can flip this True to add a light conv stem
            )
        self.ce = nn.CrossEntropyLoss()

        # optional initial freeze for very small clients
        if len(self.train_df) < FREEZE_THRESHOLD:
            for p in self.model.features.parameters():
                p.requires_grad = False



    def get_parameters(self, config):
        """Return the model weights as a list of CPU numpy arrays."""
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]


    def set_parameters(self, parameters: List[np.ndarray]):
        """Load server-provided weights into the local model (strict key match)."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)


    def _base_freeze_target(self, n_train, total_layers, round_num):
        """Decide baseline freezing ratio before gating."""
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


    def fit(self, parameters, config):
        """
        Perform one local training round.

        Args:
            parameters: model weights from the server.
            config: dict containing the current round number and FL options.

        Steps:
            1. Load server weights.
            2. Apply freezing policy based on dataset size and improvement.
            3. Train locally (EPOCHS_LOCAL).
            4. Evaluate locally and update FreezeState.
            5. Log results to CSV.
        """

        self.set_parameters(parameters)
        round_num = config.get("round", 1)

        total_layers = len(list(self.model.features))
        n_train = len(self.train_df)
        frozen_layers = self._apply_freeze_policy(round_num, n_train, total_layers)

        t0 = time.perf_counter()
        train_loader = get_loaders(self.train_df, mu=self.mu, sigma=self.sigma)
        opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                         lr=LR, weight_decay=1e-4)  # small wd for stability

        # Save current (global) params for FedProx
        global_params = [p.detach().clone() for p in self.model.parameters()]

        self.model.train()
        for _ in range(EPOCHS_LOCAL):
            for xb, yb in train_loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = self.ce(logits, yb)
                if FEDPROX_MU > 0:
                    prox = sum(torch.sum((w - w0) ** 2) for w, w0 in zip(self.model.parameters(), global_params))
                    loss = loss + (FEDPROX_MU / 2.0) * prox
                loss.backward()
                opt.step()

        wall_s = time.perf_counter() - t0
        loss_local, acc_local = self.evaluate_local()

        # --- update gating state for next round ---
        if self.state.best_loss - loss_local > self.state.min_delta:
            self.state.best_loss = loss_local
            self.state.no_improve = 0
        else:
            self.state.no_improve += 1

        # --- log metrics to correct file ---
        self._log_local_metrics_csv(round_num, acc_local, loss_local,
                                    frozen_layers, total_layers, wall_s)

        print(f"[Client {self.cid}] R{round_num}: acc={acc_local:.4f} "
              f"loss={loss_local:.4f} time={wall_s:.1f}s "
              f"freeze={frozen_layers}/{total_layers} (no_improve={self.state.no_improve})")

        return self.get_parameters({}), len(self.train_df), {"local_time": wall_s}


    def evaluate_local(self) -> Tuple[float, float]:
        """
        Evaluate on the held-out local subset.
        Returns (mean_loss, accuracy).
        """
        loader = get_loaders(self.val_df, batch_size=128, mu=self.mu, sigma=self.sigma)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in loader:
                logits = self.model(xb)
                loss_sum += float(self.ce(logits, yb))
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum())
                total += len(yb)
        mean_loss = loss_sum / total
        acc = correct / total
        return mean_loss, acc


    def _log_local_metrics_csv(self, round_num: int, acc: float, loss: float, frozen_layers: int, total_layers: int, wall_s: float):
        """ Append local metrics to a CSV file for analysis. """

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        filename = "frozen_client_metrics.csv" if frozen_layers > 0 else "not_frozen_client_metrics.csv"
        path = RESULTS_DIR / filename
        header = ["client_id", "round", "local_accuracy", "local_loss", "frozen_layers", "total_layers",
                  "wall_time_sec", "trainable_params"]
        write_header = not path.exists()
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow([self.cid, round_num, f"{acc:.4f}", f"{loss:.4f}", frozen_layers, total_layers, f"{wall_s:.2f}",
                        trainable_params])


    def evaluate(self, parameters, config):
        """ Evaluate the model on the local test dataset."""

        self.set_parameters(parameters)

        loader = get_loaders(self.test_df, batch_size=128, mu=self.mu, sigma=self.sigma)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in loader:
                logits = self.model(xb)
                loss = self.ce(logits, yb)
                loss_sum += float(loss.item()) * len(yb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(len(yb))
        return float(loss_sum / total), total, {"accuracy": correct / total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID (0..3)")
    args = parser.parse_args()
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=PTBClient(args.cid))


if __name__ == "__main__":
    main()