from __future__ import annotations
import argparse
from collections import OrderedDict
from typing import List, Tuple


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl


from .config import LR, BATCH_SIZE, EPOCHS_LOCAL, FREEZE_THRESHOLD, FEDPROX_MU, SEED
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
        set_seed(SEED + cid)
        ptb = load_metadata()
        df = filter_single_label(map_superclasses(ptb))
        train_df, test_df = stratified_patient_split(df, test_size=0.2, seed=SEED)
        # simple partition by cid
        parts = np.array_split(train_df.sample(frac=1.0, random_state=SEED), 4)
        self.train_df = parts[cid]
        self.test_df = test_df.sample(frac=0.25, random_state=SEED) # small eval subset

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
        train_loader = get_loaders(self.train_df)
        opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=LR)


        # Save global params for FedProx proximal term
        global_params = [p.detach().clone() for p in self.model.parameters()]


        self.model.train()
        for _ in range(EPOCHS_LOCAL):
            for xb, yb in train_loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = self.ce(logits, yb)
                # FedProx term
                if FEDPROX_MU > 0:
                    prox = 0.0
                    for w, w0 in zip(self.model.parameters(), global_params):
                        prox = prox + torch.sum((w - w0) ** 2)
                        loss = loss + (FEDPROX_MU / 2.0) * prox
                    loss.backward()
                    opt.step()
            return self.get_parameters({}), len(self.train_df), {}

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
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=PTBClient(args.cid))


if __name__ == "__main__":
    main()