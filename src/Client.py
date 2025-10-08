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
from pathlib import Path

from .config import LR, BATCH_SIZE, EPOCHS_LOCAL, FREEZE_THRESHOLD, FEDPROX_MU, SEED, RESULTS_DIR
from .data_loader import load_metadata, map_superclasses, filter_single_label, stratified_patient_split, load_waveform
from .models import create_cnn_model
from .utils import set_seed



def make_tensor_dataset(df) -> Tuple[torch.Tensor, torch.Tensor]:
    X, y = [], []
    classes = ["NORM", "MI", "STTC", "HYP", "CD"]
    for _, row in df.iterrows():
        sig = load_waveform(row) # (12,T)
        X.append(sig)
        y.append(classes.index(row["y"]))
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)
    return X, y


def get_loaders(df, batch_size=BATCH_SIZE):
    X, y = make_tensor_dataset(df)
    ds = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)



class PTBClient(fl.client.NumPyClient):
    def __init__(self, cid: int):
        self.cid = cid

        set_seed(SEED + cid)
        ptb = load_metadata()
        df = filter_single_label(map_superclasses(ptb))
        train_df, test_df = stratified_patient_split(df, test_size=0.2, seed=SEED)

        # --- Split training data by patient (unevenly) ---
        patients = train_df.patient_id.unique()
        np.random.seed(SEED)
        np.random.shuffle(patients)

        # Define uneven ratios (sum = 1.0)
        ratios = [0.5, 0.33, 0.15, 0.02]
        ratios = np.array(ratios) / np.sum(ratios)
        sizes = [int(r * len(patients)) for r in ratios]

        clients = []
        start = 0
        for s in sizes:
            pids = patients[start:start + s]
            client_df = train_df[train_df.patient_id.isin(pids)]
            clients.append(client_df)
            start += s

        # assign partition for this client
        self.train_df = clients[int(cid)]
        self.test_df = test_df.sample(frac=0.25, random_state=SEED)

        # --- Log client data size and freezing status ---
        n_samples = len(self.train_df)
        print(f"[Client {self.cid}] has {n_samples} records from "
              f"{self.train_df.patient_id.nunique()} unique patients.")
        if n_samples < FREEZE_THRESHOLD:
            print(f"[Client {self.cid}] BELOW freeze threshold ({FREEZE_THRESHOLD}) → features frozen.")
        else:
            print(f"[Client {self.cid}] ABOVE freeze threshold ({FREEZE_THRESHOLD}) → features trainable.")

        """
        with open("results/client_data_log.txt", "a") as f:
            status = "frozen" if n_samples < FREEZE_THRESHOLD else "trainable"
            f.write(f"Client {self.cid}: {n_samples} records, "
                    f"{self.train_df.patient_id.nunique()} patients ({status})\n")"""

        # --- Model setup ---
        self.model = create_cnn_model(5)
        self.ce = nn.CrossEntropyLoss()

        # dynamic freezing for small clients
        if len(self.train_df) < FREEZE_THRESHOLD:
            for p in self.model.features.parameters():
                p.requires_grad = False


    # ---- Flower API ----
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        round_num = config.get("round", 1)
        unfreeze_after = config.get("unfreeze_after", 8)

        # ===== Graduated + dynamic freezing (works even with 3 or 5 conv blocks) =====
        total_layers = len(list(self.model.features))
        n_train = len(self.train_df)
        frozen_layers = 0

        if n_train < FREEZE_THRESHOLD:
            freeze_ratio = max(0.0, 1.0 - round_num / max(1, unfreeze_after))
            n_to_freeze = int(total_layers * freeze_ratio)
            frozen_layers = n_to_freeze
            for i, layer in enumerate(self.model.features):
                for p in layer.parameters():
                    p.requires_grad = (i >= n_to_freeze)
        elif n_train < 2 * FREEZE_THRESHOLD:
            n_to_freeze = total_layers // 2
            frozen_layers = n_to_freeze
            for i, layer in enumerate(self.model.features):
                for p in layer.parameters():
                    p.requires_grad = (i >= n_to_freeze)
        else:
            for p in self.model.features.parameters():
                p.requires_grad = True

        print(f"[Client {self.cid}] Round {round_num}: {n_train} samples, frozen_layers={frozen_layers}/{total_layers}")

        # ===== Timing start =====
        t0 = time.perf_counter()

        train_loader = get_loaders(self.train_df)
        opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=LR)

        # Save global params for FedProx
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

        # ===== Timing end + local eval =====
        wall_s = time.perf_counter() - t0
        loss_local, acc_local = self.evaluate_local()
        self._log_local_metrics_csv(round_num, acc_local, loss_local, frozen_layers, total_layers, wall_s)
        print(f"[Client {self.cid}] Round {round_num}: local acc={acc_local:.4f}, loss={loss_local:.4f}, time={wall_s:.1f}s")

        return self.get_parameters({}), len(self.train_df), {"local_time": wall_s}


    def evaluate_local(self) -> Tuple[float, float]:
        """Evaluate the model on the local validation subset."""
        loader = get_loaders(self.test_df, batch_size=128)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in loader:
                logits = self.model(xb)
                loss_sum += float(self.ce(logits, yb).item()) * len(yb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(len(yb))
        acc = correct / total if total > 0 else 0.0
        loss_mean = loss_sum / total if total > 0 else 0.0
        return loss_mean, acc

    def _log_local_metrics_csv(self, round_num: int, acc: float, loss: float, frozen_layers: int, total_layers: int,
                               wall_s: float):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "not_frozen_client_metrics.csv"
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
        self.set_parameters(parameters)

        loader = get_loaders(self.test_df, batch_size=128)
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