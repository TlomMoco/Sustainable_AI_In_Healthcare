"""Client.py — PTB-XL Federated Learning Client
--------------------------------------------

Implements a single federated learning (FL) client for ECG classification
under the Sustainable AI in Healthcare project (DSP5100).

Each client:
  • Loads a patient-specific PTB-XL subset (uneven split)
  • Builds a CNN/RNN/LSTM/ANN model depending on config.py
  • Trains locally using FedAvg (with optional FedProx term)
  • Applies dynamic & reactive layer freezing
  • Logs sustainability metrics per round
  • Communicates with the Flower server (FedAvg)
  • Logs per-round metrics (accuracy, loss, time, #trainable params)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    import flwr as fl
except ImportError as e:
    raise ImportError("Please install Flower: pip install flwr==1.*") from e

# --- Project imports (absolute) ---
from src.config import (
    LR, BATCH_SIZE, EPOCHS_LOCAL, FREEZE_THRESHOLD, FEDPROX_MU,
    SEED, RESULTS_DIR, FREEZE_CFG, SPLITS, NORM, MODEL, EXPERIMENT,
    SAMPLE_RATE, N_CLASSES, TUNING, GRIDSEARCH, CLIENTS, FL_SERVER_ADDRESS,
)
from src.data_loader import (
    load_metadata, map_superclasses, filter_single_label,
    stratified_patient_split_3way, load_waveform,
    compute_perlead_norm_stats, normalize_signal,
)
from src.models import create_model
from src.utils import set_seed, ensure_dir


# -------------------------------------------------------------------------
# Small helpers for dataset -> tensors
# -------------------------------------------------------------------------
CLASSES = ["NORM", "MI", "STTC", "HYP", "CD"]

def _make_tensor_dataset(df, mu=None, sigma=None):
    X, y = [], []
    for _, row in df.iterrows():
        sig = load_waveform(row)
        if mu is not None and sigma is not None and NORM["enabled"]:
            sig = normalize_signal(sig, mu, sigma, eps=NORM["eps"])
        X.append(sig)
        y.append(CLASSES.index(row["y"]))
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)
    return X, y

def _loader_for(df, batch_size=BATCH_SIZE, mu=None, sigma=None, shuffle=True):
    X, y = _make_tensor_dataset(df, mu, sigma)
    ds = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(ds, batch_size=int(batch_size), shuffle=bool(shuffle))


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
# Federated Client
# -------------------------------------------------------------------------
class PTBClient(fl.client.NumPyClient):
    """Federated learning client with reactive layer freezing."""

    def __init__(self, cid: int):
        # --- Setup -------------------------------------------------------
        self.cid = int(cid)
        set_seed(SEED + self.cid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Load + map labels, single label per row --------------------
        ptb = load_metadata()
        df = filter_single_label(map_superclasses(ptb))

        # --- 3-way patient-safe split (global) --------------------------
        train_g, val_g, test_g = stratified_patient_split_3way(
            df,
            splits=(SPLITS["train"], SPLITS["val"], SPLITS["test"]),
            seed=SEED,
        )

        # --- Uneven per-client allocation by patient --------------------
        # Generalized for arbitrary CLIENTS:
        patients = train_g.patient_id.unique()
        rng = np.random.RandomState(SEED)
        rng.shuffle(patients)

        if CLIENTS <= 1:
            buckets = [patients]
        else:
            # Make uneven buckets by drawing proportions from a Dirichlet, then slicing patients.
            props = rng.dirichlet(alpha=[0.8] + [0.4]*(CLIENTS-1))
            counts = np.floor(props * len(patients)).astype(int)
            # fix rounding to match total
            while counts.sum() < len(patients):
                counts[rng.randint(0, CLIENTS)] += 1
            splits = np.cumsum(counts)
            buckets = np.split(patients, splits[:-1])

        assert 0 <= self.cid < len(buckets), f"cid={self.cid} out of range (0..{len(buckets)-1})"
        pids = set(map(int, buckets[self.cid]))
        self.train_df = train_g[train_g.patient_id.isin(pids)].copy()
        self.val_df   = val_g.copy()
        self.test_df  = test_g.copy()

        # --- Visibility: rough volume ----------------------------------
        n_records = len(self.train_df)
        samples_per_record = 10 * SAMPLE_RATE * 12  # 10s * Hz * 12 leads
        total_samples = int(n_records * samples_per_record)
        hours = (n_records * 10.0) / 3600.0
        print(
            f"[Client {self.cid}] train records={n_records:,} "
            f"≈ {hours:.1f} h ECG | samples={total_samples:,} @ {SAMPLE_RATE} Hz"
        )

        # --- Local normalization stats ----------------------------------
        self.mu, self.sigma = compute_perlead_norm_stats(self.train_df)  # shape (12,), (12,)

        # --- Model -------------------------------------------------------
        self.model = create_model(
            n_classes=N_CLASSES,
            model_type=MODEL["type"],
            hidden=MODEL.get("lstm_hidden", 128),
            layers=MODEL.get("lstm_layers", 1),
            bidir=MODEL.get("bidirectional", True),
        ).to(self.device)

        self.ce = nn.CrossEntropyLoss()
        self.state = FreezeState()

        # --- Hyperparams (subject to tuning) ----------------------------
        self.lr = float(LR)
        self.batch_size = int(BATCH_SIZE)
        self.local_epochs = int(EPOCHS_LOCAL)
        self.fedprox_mu = float(FEDPROX_MU)

        # --- Optional client-side CV / cached reuse ---------------------
        self.hp_source = "default"  # "default" | "cached" | "tuned"
        outdir = RESULTS_DIR / "tuning" / EXPERIMENT["run_name"]
        best_path = outdir / f"client{self.cid}_best.json"

        def _apply(hp: dict, source: str):
            self.lr = float(hp.get("lr", self.lr))
            self.batch_size = int(hp.get("batch", self.batch_size))
            self.local_epochs = int(hp.get("epochs", self.local_epochs))
            self.fedprox_mu = float(hp.get("fedprox", self.fedprox_mu))
            self.hp_source = source

        if TUNING.get("enabled", False):
            from src.tuning import run_client_cv
            outdir.mkdir(parents=True, exist_ok=True)

            if TUNING.get("reuse_cached_if_exists", True) and best_path.exists():
                try:
                    best = (json.loads(best_path.read_text()) or {}).get("params", {})
                    if best:
                        _apply(best, "cached")
                        print(f"[Client {self.cid}] Using cached tuned HPs → "
                              f"lr={self.lr}, batch={self.batch_size}, epochs={self.local_epochs}, μ={self.fedprox_mu}")
                except Exception:
                    pass

            if self.hp_source == "default":  # still not set from cache
                print(f"[Client {self.cid}] Starting CV: grid={len(GRIDSEARCH.get('grid', []))}, "
                      f"k={int(GRIDSEARCH.get('cv', 5))}")
                res = run_client_cv(
                    self.train_df,
                    k=int(GRIDSEARCH.get("cv", 5)),
                    grid=list(GRIDSEARCH.get("grid", [])),
                    device=self.device,
                    save_csv=outdir / f"client{self.cid}_cv.csv",
                    save_json=best_path,
                )
                best = (res or {}).get("best") or {}
                if best:
                    _apply(best, "tuned")
                    print(f"[Client {self.cid}] CV best={res.get('best_mean_acc', 0):.4f} → "
                          f"lr={self.lr}, batch={self.batch_size}, epochs={self.local_epochs}, μ={self.fedprox_mu}")
                else:
                    print(f"[Client {self.cid}] CV skipped/insufficient groups; using defaults.")

        elif TUNING.get("use_cached_best", False) and best_path.exists():
            try:
                best = (json.loads(best_path.read_text()) or {}).get("params", {})
                if best:
                    _apply(best, "cached")
                    print(f"[Client {self.cid}] Reusing cached tuned HPs (CV off) → "
                          f"lr={self.lr}, batch={self.batch_size}, epochs={self.local_epochs}, μ={self.fedprox_mu}")
            except Exception:
                pass

        # --- Optional head-only warm start for tiny clients -------------
        if EXPERIMENT.get("freeze_enabled", False) and len(self.train_df) < int(FREEZE_THRESHOLD):
            print(f"[Client {self.cid}] Head-only warm start (small client).")
            # Try freezing features/head if present
            if hasattr(self.model, "features"):
                for p in self.model.features.parameters():
                    p.requires_grad = False
            if hasattr(self.model, "head"):
                # enable only the last linear in head
                for j, module in enumerate(self.model.head):
                    req = (j == len(self.model.head) - 1)
                    for p in getattr(module, "parameters", lambda: [])():
                        p.requires_grad = req

        print(f"[Client {self.cid}] Model: {MODEL['type']} on {self.device} | HP src: {self.hp_source}")

    # ---------------------------------------------------------------------
    # Flower API
    # ---------------------------------------------------------------------
    def get_parameters(self, config: Optional[dict] = None):
        return [p.detach().cpu().numpy() for _, p in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        state = self.model.state_dict()
        for k, v in zip(state.keys(), parameters):
            state[k] = torch.tensor(v)
        self.model.load_state_dict(state, strict=True)

    # ---------------------------------------------------------------------
    # Freezing policy (hybrid: client size + reactive)
    # ---------------------------------------------------------------------
    def _apply_freeze_policy(self, round_num: int, loss_local: float, acc_local: float):
        if not EXPERIMENT.get("freeze_enabled", False):
            for p in self.model.parameters():
                p.requires_grad = True
            self.state.last_frozen = 0
            return {"policy": "disabled", "frozen_layers": 0}

        # If model has no .features, bail out gracefully
        if not hasattr(self.model, "features"):
            return {"policy": "unavailable", "frozen_layers": 0}

        total_layers = len(list(self.model.features))
        n_train = len(self.train_df)

        # Base target based on size
        if n_train >= 2 * int(FREEZE_THRESHOLD):
            target = 0
        elif n_train < int(FREEZE_THRESHOLD):
            target = int(total_layers * max(0.0, 1 - round_num / float(FREEZE_CFG["unfreeze_after"])))
        else:
            target = max(0, total_layers // 4)

        # Reactive part
        improved = ((self.state.best_loss - loss_local) > self.state.min_delta) or (acc_local > self.state.best_acc)
        degraded = ((loss_local - self.state.best_loss) > self.state.min_delta) or \
                   (acc_local < self.state.best_acc - 1e-3)

        if degraded:
            self.state.no_improve += 1
        elif improved:
            self.state.no_improve = max(0, self.state.no_improve - 1)

        adjust = (self.state.no_improve // self.state.patience)
        if degraded:
            target = min(total_layers - 1, target + adjust)
        elif improved:
            target = max(0, target - adjust)

        # Apply pattern: freeze first `target` feature layers
        for i, layer in enumerate(self.model.features):
            for p in layer.parameters():
                p.requires_grad = (i >= target)

        self.state.last_frozen = target
        self.state.best_loss = min(self.state.best_loss, loss_local)
        self.state.best_acc = max(self.state.best_acc, acc_local)

        return {
            "policy": "dynamic",
            "frozen_layers": target,
            "improved": improved,
            "degraded": degraded,
            "no_improve": self.state.no_improve,
        }

    # ---------------------------------------------------------------------
    # Training / Evaluation on local data
    # ---------------------------------------------------------------------
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        round_num = int(config.get("round", 1))

        # Pre-eval (val) to update freezing plan
        loss_before, acc_before = self._eval_on(self.val_df)
        self._apply_freeze_policy(round_num, loss_before, acc_before)

        # Train
        t0 = time.perf_counter()
        loader = _loader_for(self.train_df, batch_size=self.batch_size, mu=self.mu, sigma=self.sigma, shuffle=True)
        opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                         lr=self.lr, weight_decay=1e-4)

        # FedProx snapshot
        global_params = [p.detach().clone() for p in self.model.parameters()] if self.fedprox_mu > 0 else None

        self.model.train()
        for _ in range(self.local_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = self.ce(logits, yb)
                if self.fedprox_mu > 0 and global_params is not None:
                    prox = sum(torch.sum((w - w0) ** 2) for w, w0 in zip(self.model.parameters(), global_params))
                    loss = loss + (self.fedprox_mu / 2.0) * prox
                loss.backward()
                opt.step()

        wall_s = time.perf_counter() - t0
        loss_after, acc_after = self._eval_on(self.val_df)

        # Track bests
        self.state.best_loss = min(self.state.best_loss, loss_after)
        self.state.best_acc = max(self.state.best_acc, acc_after)

        self._log_metrics(round_num, acc_after, loss_after, wall_s)
        print(f"[Client {self.cid}] R{round_num}: acc={acc_after:.4f}, loss={loss_after:.4f}, "
              f"time={wall_s:.1f}s, frozen_layers={self.state.last_frozen}")

        return self.get_parameters({}), len(self.train_df), {"accuracy": float(acc_after)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # Evaluate on TEST
        loader = _loader_for(self.test_df, batch_size=128, mu=self.mu, sigma=self.sigma, shuffle=False)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0

        # Confusion counts (rows:true, cols:pred)
        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss_sum += float(self.ce(logits, yb).item()) * len(yb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum())
                total += len(yb)
                for t, p in zip(yb.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy()):
                    cm[int(t), int(p)] += 1

        loss = (loss_sum / total) if total > 0 else 0.0
        acc = (correct / total) if total > 0 else 0.0

        metrics = {"accuracy": float(acc)}
        # Flatten confusion matrix into metrics
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                metrics[f"cm_{i}_{j}"] = float(cm[i, j])

        return float(loss), len(self.test_df), metrics

    # -- helper eval on arbitrary df ------------------------------------
    def _eval_on(self, df) -> Tuple[float, float]:
        if len(df) == 0:
            return 0.0, 0.0
        loader = _loader_for(df, batch_size=128, mu=self.mu, sigma=self.sigma, shuffle=False)
        self.model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss_sum += float(self.ce(logits, yb).item()) * len(yb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum())
                total += len(yb)
        return (loss_sum / total if total else 0.0), (correct / total if total else 0.0)

    # ---------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------
    def _log_metrics(self, round_num: int, acc: float, loss: float, wall_s: float):
        ensure_dir(RESULTS_DIR)
        exp = EXPERIMENT["run_name"]

        if TUNING.get("log_phase", False):
            phase = (TUNING["phase_labels"]["enabled"] if TUNING.get("enabled") else
                     TUNING["phase_labels"]["disabled"])
        else:
            phase = ""

        if TUNING.get("log_phase", False) and TUNING.get("log_mode", "same") == "separate":
            path = RESULTS_DIR / f"{exp}_{phase or 'no_cv'}.csv"
            header = ["client_id", "round", "accuracy", "loss",
                      "frozen_layers", "is_frozen", "wall_time_sec", "trainable_params"]
            include_phase = False
        else:
            path = RESULTS_DIR / f"{exp}.csv"
            header = ["client_id", "round", "accuracy", "loss",
                      "frozen_layers", "is_frozen", "wall_time_sec", "trainable_params", "phase"]
            include_phase = True

        write_header = not path.exists()
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        is_frozen = self.state.last_frozen > 0

        with path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            row = [self.cid, round_num, f"{acc:.4f}", f"{loss:.4f}",
                   self.state.last_frozen, int(is_frozen), f"{wall_s:.2f}", trainable_params]
            if include_phase:
                row.append(phase)
            w.writerow(row)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID (0..CLIENTS-1)")
    args = parser.parse_args()

    client = PTBClient(args.cid)

    # Flower >=1.0: start_client + NumPyClient instance
    fl.client.start_client(
        server_address=str(FL_SERVER_ADDRESS),
        client=client,  # NumPyClient instance
    )


if __name__ == "__main__":
    main()
