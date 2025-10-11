"""
Client.py — PTB-XL Federated Learning Client
--------------------------------------------

Implements a single federated learning (FL) client for ECG classification
under the Sustainable AI in Healthcare project (DSP5100).

Each client:
  • Loads a patient-specific PTB-XL subset (uneven split)
  • Builds a CNN or LSTM model depending on config.py
  • Trains locally using FedAvg (with optional FedProx term)
  • Applies dynamic & reactive layer freezing
  • Logs sustainability metrics per round
  • Communicates with the Flower server (FedAvg)
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
    SEED, RESULTS_DIR, FREEZE_CFG, SPLITS, NORM, MODEL, EXPERIMENT,
    SAMPLE_RATE, N_CLASSES, TUNING, GRIDSEARCH
)
from src.data_loader import (
    load_metadata, map_superclasses, filter_single_label,
    stratified_patient_split_3way, load_waveform,
    compute_perlead_norm_stats, normalize_signal
)
from src.tuning import run_client_cv
from src.models import create_model
from src.utils import set_seed, ensure_dir


# -------------------------------------------------------------------------
# Freeze state tracker
# -------------------------------------------------------------------------
@dataclass
class FreezeState:
    best_loss: float = float("inf")
    best_acc: float = 0.0
    no_improve: int = 0
    patience: int = FREEZE_CFG["patience"]
    min_delta: float = FREEZE_CFG["min_delta"]
    last_frozen: int = 0


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def make_tensor_dataset(df, mu=None, sigma=None):
    """Convert a PTB-XL dataframe to tensors (X, y)."""
    X, y = [], []
    classes = ["NORM", "MI", "STTC", "HYP", "CD"]
    for _, row in df.iterrows():
        sig = load_waveform(row)
        if mu is not None and sigma is not None and NORM["enabled"]:
            sig = normalize_signal(sig, mu, sigma, eps=NORM["eps"])
        X.append(sig)
        y.append(classes.index(row["y"]))
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)
    return X, y


def get_loaders(df, batch_size=BATCH_SIZE, mu=None, sigma=None):
    """Return a DataLoader for the given DataFrame."""
    X, y = make_tensor_dataset(df, mu, sigma)
    ds = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


# -------------------------------------------------------------------------
# Federated Client
# -------------------------------------------------------------------------
class PTBClient(fl.client.NumPyClient):
    """Federated learning client with reactive layer freezing."""

    def __init__(self, cid: int):
        # --- Setup -------------------------------------------------------
        self.cid = cid
        set_seed(SEED + cid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Load and prepare data --------------------------------------
        ptb = load_metadata()
        df = filter_single_label(map_superclasses(ptb))

        # Explicit split tuple to ensure consistent order
        train_g, val_g, test_g = stratified_patient_split_3way(
            df,
            splits=(SPLITS["train"], SPLITS["val"], SPLITS["test"]),
            seed=SEED,
        )

        # Uneven client splits (by patient) ------------------------------
        patients = train_g.patient_id.unique()
        np.random.seed(SEED)
        np.random.shuffle(patients)

        ratios = np.array([0.5, 0.33, 0.15, 0.02])
        ratios /= ratios.sum()
        sizes = (ratios * len(patients)).astype(int)
        sizes[-1] = len(patients) - sizes[:-1].sum()  # assign remainder to last client

        clients, start = [], 0
        for s in sizes:
            pids = patients[start:start + s]
            client_df = train_g[train_g.patient_id.isin(pids)]
            clients.append(client_df)
            start += s

        assert 0 <= cid < len(clients), f"cid {cid} out of range (have {len(clients)} clients)"
        self.train_df = clients[cid]
        self.val_df, self.test_df = val_g, test_g

        # --- Visibility: true training volume ----------------------------
        n_records = len(self.train_df)
        samples_per_record = 10 * SAMPLE_RATE * 12  # 10s * Hz * 12 leads
        total_samples = n_records * samples_per_record
        hours = (n_records * 10.0) / 3600.0  # 10 seconds per record -> hours
        print(
            f"[Client {self.cid}] train records={n_records:,} "
            f"≈ {hours:.1f} h ECG | samples={total_samples:,} @ {SAMPLE_RATE} Hz"
        )

        # --- Per-lead normalization (local stats) ------------------------
        self.mu, self.sigma = compute_perlead_norm_stats(self.train_df)

        # --- Model setup -------------------------------------------------
        self.model = create_model(
            n_classes=N_CLASSES,
            model_type=MODEL["type"],
            hidden=MODEL.get("lstm_hidden", 128),
            layers=MODEL.get("lstm_layers", 1),
            bidir=MODEL.get("bidirectional", True),
        ).to(self.device)

        self.ce = nn.CrossEntropyLoss()
        self.state = FreezeState()

        # --- Per Client hyperparams (for tuning experiments) -----------
        self.lr = LR
        self.batch_size = BATCH_SIZE
        self.local_epochs = EPOCHS_LOCAL
        self.fedprox_mu = FEDPROX_MU  # proximal term (not normalization μ)

        # ----------------------------------------------------------------
        # Hyperparameter tuning via CV (pre-flight) + cached reuse
        # ----------------------------------------------------------------
        import json

        self.cv_done = False
        self.hp_source = "default"  # "default" | "cached" | "tuned"

        outdir = RESULTS_DIR / "tuning" / EXPERIMENT["run_name"]
        best_path = outdir / f"client{self.cid}_best.json"

        def _apply_hparams(hp: dict, src: str):
            self.lr = float(hp["lr"])
            self.batch_size = int(hp["batch"])
            self.local_epochs = int(hp["epochs"])
            self.fedprox_mu = float(hp.get("fedprox", 0.0))
            self.hp_source = src

        if TUNING.get("enabled", False):
            outdir.mkdir(parents=True, exist_ok=True)

            # Reuse cached best if allowed and present
            if TUNING.get("reuse_cached_if_exists", True) and best_path.exists():
                with open(best_path, "r", encoding="utf-8") as f:
                    best = (json.load(f) or {}).get("params", {})
                if best:
                    _apply_hparams(best, "cached")
                    print(f"[Client {self.cid}] Using cached tuned HPs → "
                          f"lr={self.lr}, batch={self.batch_size}, epochs={self.local_epochs}, μ={self.fedprox_mu}")
            else:
                # Run CV now
                print(f"[Client {self.cid}] Starting pre-flight CV: grid={len(GRIDSEARCH.get('grid', []))}, "
                      f"k={GRIDSEARCH.get('cv', 5)}")
                res = run_client_cv(
                    self.train_df,
                    k=GRIDSEARCH.get("cv"),
                    grid=GRIDSEARCH.get("grid"),
                    device=self.device,
                    save_csv=outdir / f"client{self.cid}_cv.csv",
                    save_json=best_path,
                )
                best = (res or {}).get("best") or {}
                if best:
                    _apply_hparams(best, "tuned")
                    print(f"[Client {self.cid}] CV best={res.get('best_mean_acc', 0):.4f} → "
                          f"lr={self.lr}, batch={self.batch_size}, epochs={self.local_epochs}, μ={self.fedprox_mu}")
                else:
                    print(f"[Client {self.cid}] CV skipped/insufficient groups; using defaults.")

            print(f"[Client {self.cid}] Pre-flight CV finished.")
            self.cv_done = True

        # If CV is OFF but we want to reuse cached best from a previous run
        elif TUNING.get("use_cached_best", True) and best_path.exists():
            with open(best_path, "r", encoding="utf-8") as f:
                best = (json.load(f) or {}).get("params", {})
            if best:
                _apply_hparams(best, "cached")
                print(f"[Client {self.cid}] Reusing cached tuned HPs (CV off) → "
                      f"lr={self.lr}, batch={self.batch_size}, epochs={self.local_epochs}, μ={self.fedprox_mu}")


        # Static/head-only start for very small clients (only when freezing is enabled)
        if EXPERIMENT["freeze_enabled"] and len(self.train_df) < FREEZE_THRESHOLD:
            print(f"[Client {self.cid}] Head-only training to start (small client).")
            # 1) freeze backbone (features)
            for p in self.model.features.parameters():
                p.requires_grad = False
            # 2) within the head, train ONLY the last Linear (keep other head layers frozen)
            for j, m in enumerate(self.model.head):
                req_grad = (j == len(self.model.head) - 1)  # last module is Linear(96->N_CLASSES) in your CNN
                for p in getattr(m, "parameters", lambda: [])():
                    p.requires_grad = req_grad

        print(f"[Client {self.cid}] Initialized model: {MODEL['type']} on {self.device}")


    # ---------------------------------------------------------------------
    # Federated API
    # ---------------------------------------------------------------------
    # Flower calls this with a config kwarg — accept it even if unused
    def get_parameters(self, config: dict | None = None):
        return [v.detach().cpu().numpy() for _, v in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in params_dict}), strict=True)


    # ---------------------------------------------------------------------
    # Freezing logic (hybrid: size + reactive)
    # ---------------------------------------------------------------------
    def _apply_freeze_policy(self, round_num: int, loss_local: float, acc_local: float):
        """Adjust layer freezing based on client size and performance."""

        # Global toggle — disable all freezing if requested
        if not EXPERIMENT["freeze_enabled"]:
            for p in self.model.parameters():
                p.requires_grad = True
            if hasattr(self, "state"):
                self.state.last_frozen = 0
            return {"policy": "disabled", "frozen_layers": 0}

        total = len(list(self.model.features))
        n_train = len(self.train_df)

        # Base: size-aware target (static baseline)
        if n_train >= 2 * FREEZE_THRESHOLD:
            target = 0
        elif n_train < FREEZE_THRESHOLD:
            target = int(total * max(0.0, 1 - round_num / FREEZE_CFG["unfreeze_after"]))
        else:
            target = total // 4  # gentle mid-size default

        # Reactive adjustment based on performance trends
        improved = (self.state.best_loss - loss_local) > self.state.min_delta or (acc_local > self.state.best_acc)
        degraded = (loss_local - self.state.best_loss) > self.state.min_delta or (
                acc_local < self.state.best_acc - 1e-3
        )

        if degraded:
            self.state.no_improve += 1
        elif improved:
            self.state.no_improve = max(0, self.state.no_improve - 1)

        # Every `patience` steps without improvement: adjust one layer
        adjust = (self.state.no_improve // self.state.patience)
        if degraded:
            target = min(total - 1, target + adjust)  # freeze more
        elif improved:
            target = max(0, target - adjust)  # unfreeze

        if len(self.train_df) < FREEZE_THRESHOLD and improved and self.state.no_improve == 0:
            # allow the Linear(160->96) to train too
            for p in self.model.head[1].parameters():  # head[1] is Linear(160->96)
                p.requires_grad = True

        # Apply the actual freezing pattern
        for i, layer in enumerate(self.model.features):
            for p in layer.parameters():
                p.requires_grad = (i >= target)

        # Update internal state
        self.state.last_frozen = target
        self.state.best_loss = min(self.state.best_loss, loss_local)
        self.state.best_acc = max(self.state.best_acc, acc_local)

        # Return summary
        return {
            "policy": "dynamic",
            "frozen_layers": target,
            "improved": improved,
            "degraded": degraded,
            "no_improve": self.state.no_improve,
        }


    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        round_num = config.get("round", 1)

        # Pre-evaluate to determine freeze adjustment
        loss_before, acc_before = self._evaluate_dataset(self.val_df)
        self._apply_freeze_policy(round_num, loss_before, acc_before)

        # Local training with optional FedProx term
        t0 = time.perf_counter()
        loader = get_loaders(self.train_df, batch_size=self.batch_size, mu=self.mu, sigma=self.sigma)
        opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                         lr=self.lr, weight_decay=1e-4)
        global_params = [p.detach().clone() for p in self.model.parameters()]  # FedProx snapshot

        self.model.train()
        for _ in range(self.local_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                logits = self.model(xb)
                loss = self.ce(logits, yb)
                if self.fedprox_mu > 0:
                    prox = sum(torch.sum((w - w0) ** 2) for w, w0 in zip(self.model.parameters(), global_params))
                    loss += (self.fedprox_mu / 2.0) * prox
                loss.backward()
                opt.step()

        wall_s = time.perf_counter() - t0
        loss_after, acc_after = self._evaluate_dataset(self.val_df)

        # Update state for next round
        if loss_after < self.state.best_loss:
            self.state.best_loss = loss_after
        if acc_after > self.state.best_acc:
            self.state.best_acc = acc_after

        self._log_metrics(round_num, acc_after, loss_after, wall_s)
        print(f"[Client {self.cid}] R{round_num}: acc={acc_after:.4f}, loss={loss_after:.4f}, "
              f"time={wall_s:.1f}s, frozen_layers={self.state.last_frozen}")

        return self.get_parameters({}), len(self.train_df), {"accuracy": acc_after}


    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------
    # -- helper to evaluate on a given dataset ----------------------------
    def _evaluate_dataset(self, df) -> Tuple[float, float]:
        if len(df) == 0:
            return 0.0, 0.0
        loader = get_loaders(df, 128, mu=self.mu, sigma=self.sigma)
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
        return loss_sum / total, correct / total

    # -- main evaluation method (called by server) ----------------------
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Compute metrics + confusion counts on TEST split
        loader = get_loaders(self.test_df, 128, mu=self.mu, sigma=self.sigma)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0

        # Confusion matrix: rows=true, cols=pred
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)

        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss_sum += float(self.ce(logits, yb).item()) * len(yb)
                pred = logits.argmax(dim=1)

                correct += int((pred == yb).sum())
                total += len(yb)

                # update confusion counts
                for t, p in zip(yb.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy()):
                    cm[int(t), int(p)] += 1

        loss = (loss_sum / total) if total > 0 else 0.0
        acc = (correct / total) if total > 0 else 0.0

        # Flower expects flat dict[str,float]
        metrics = {"accuracy": acc}
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                metrics[f"cm_{i}_{j}"] = float(cm[i, j])

        return loss, len(self.test_df), metrics


    # ---------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------
    def _log_metrics(self, round_num: int, acc: float, loss: float, wall_s: float):
        ensure_dir(RESULTS_DIR)
        exp = EXPERIMENT["run_name"]

        # Determine phase label (simple: tuned vs no_cv)
        if TUNING.get("log_phase"):
            phase = (TUNING["phase_labels"]["enabled"] if TUNING.get("enabled") else
                     TUNING["phase_labels"]["disabled"])
        else:
            phase = ""

        # Choose file path depending on log mode
        if TUNING.get("log_phase") and TUNING.get("log_mode") == "separate":
            path = RESULTS_DIR / f"{exp}_{phase or 'no_cv'}.csv"
            header = ["client_id", "round", "accuracy", "loss",
                      "frozen_layers", "is_frozen", "wall_time_sec", "trainable_params"]
            write_phase = False
        else:
            path = RESULTS_DIR / f"{exp}.csv"
            header = ["client_id", "round", "accuracy", "loss",
                      "frozen_layers", "is_frozen", "wall_time_sec", "trainable_params", "phase"]
            write_phase = True

        write_header = not path.exists()
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        is_frozen = self.state.last_frozen > 0

        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            row = [self.cid, round_num, f"{acc:.4f}", f"{loss:.4f}",
                   self.state.last_frozen, int(is_frozen), f"{wall_s:.2f}", trainable_params]
            if write_phase:
                row.append(phase)
            w.writerow(row)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True)
    args = parser.parse_args()
    client = PTBClient(args.cid)
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client(),
    )


if __name__ == "__main__":
    main()