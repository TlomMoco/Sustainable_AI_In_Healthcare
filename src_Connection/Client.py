"""Client.py — PTB-XL Federated Learning Client
--------------------------------------------
Federated client with optional dynamic freezing and FedProx.
Trains CNN/RNN/LSTM/ANN defined in src_Connection.models on local splits.

High-level responsibilities
---------------------------
1) Data: load PTB-XL metadata, map labels to superclasses, enforce single-label rows,
   perform a *global* patient-safe 3-way split, then partition the global train set
   across clients by patient-id (optionally uneven via Dirichlet proportions).
2) Normalization: compute per-lead (12 leads) mean/std from the *local* train split.
3) Model: instantiate the selected architecture via models.create_model(...).
4) Training: local epochs with class-weighted CrossEntropy; optional FedProx regularization.
5) Freezing: reactive layer freezing policy with a head-only warm start for small clients.
6) Flower API: get_parameters / set_parameters / fit / evaluate implement NumPyClient.
7) Logging: append per-round metrics to CSV under RESULTS_DIR.

Key connections (modules this file uses)
----------------------------------------
• src_Connection.config:
    LR, BATCH_SIZE, EPOCHS_LOCAL, FREEZE_THRESHOLD, FEDPROX_MU,
    SEED, RESULTS_DIR, FREEZE_CFG, SPLITS, NORM, MODEL, EXPERIMENT,
    SAMPLE_RATE, N_CLASSES, TUNING, GRIDSEARCH, CLIENTS, FL_SERVER_ADDRESS, SUPERCLASSES
• src_Connection.data_loader:
    load_metadata, map_superclasses, filter_single_label,
    stratified_patient_split_3way, load_waveform,
    compute_perlead_norm_stats, normalize_signal
• src_Connection.models: create_model
• src_Connection.utils: set_seed, ensure_dir, pick_device, sanitize_mps_env, torch_loader_kwargs
• Flower (flwr): client protocol integration (NumPyClient)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
try:
    import flwr as fl
except ImportError as e:
    # Explicit error to guide setup for FL runs.
    raise ImportError("Please install Flower: pip install 'flwr==1.*'") from e

# --- Project imports (absolute) ---
from src_Connection import (
    LR, BATCH_SIZE, EPOCHS_LOCAL, FREEZE_THRESHOLD, FEDPROX_MU,
    SEED, RESULTS_DIR, FREEZE_CFG, SPLITS, NORM, MODEL, EXPERIMENT,
    SAMPLE_RATE, N_CLASSES, TUNING, GRIDSEARCH, CLIENTS, FL_SERVER_ADDRESS,
    SUPERCLASSES,
)
from src_Connection import (
    load_metadata, map_superclasses, filter_single_label,
    stratified_patient_split_3way, load_waveform,
    compute_perlead_norm_stats, normalize_signal,
)
from src_Connection import create_model
from src_Connection import (
    set_seed, ensure_dir, pick_device, sanitize_mps_env, torch_loader_kwargs
)


# -------------------------------------------------------------------------
# Small helpers for dataset -> tensors
# -------------------------------------------------------------------------
CLASSES = list(SUPERCLASSES)  # Keep aligned with config; used for index mapping.

def _make_tensor_dataset(df, mu=None, sigma=None):
    """
    Convert a PTB-XL dataframe (rows -> records) to a TensorDataset.

    Pipeline per row
    ----------------
    1) load_waveform(row): returns np.ndarray of shape (12, T) in lead-major order.
    2) Optional per-lead normalization in (12, T) space using mu/sigma if NORM["enabled"].
    3) Transpose to time-major (T, 12) expected by temporal models (RNN/LSTM etc.).
    4) Map label string to integer index via CLASSES.index(row["y"]).

    Parameters
    ----------
    df : pandas.DataFrame
        Must include label column "y" and metadata needed by load_waveform(row).
    mu, sigma : Optional[np.ndarray]
        Per-lead mean/std (shape (12,)); computed from *local* training split.

    Returns
    -------
    torch.utils.data.TensorDataset
        (X, y) where X is float32 of shape (N, T, 12) and y is long of shape (N,).
    """
    X, y = [], []
    for _, row in df.iterrows():
        sig = load_waveform(row)                # (12, T)
        if mu is not None and sigma is not None and NORM["enabled"]:
            sig = normalize_signal(sig, mu, sigma, eps=NORM["eps"])  # (12, T)
        sig = sig.T.astype(np.float32, copy=False)  # -> (T, 12)
        X.append(sig)
        y.append(CLASSES.index(row["y"]))
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)
    return torch.utils.data.TensorDataset(X, y)

def _loader_for(df, batch_size=BATCH_SIZE, mu=None, sigma=None, shuffle=True, device_type="cpu"):
    """
    Build a DataLoader from a dataframe using _make_tensor_dataset.

    Uses torch_loader_kwargs from utils to keep consistent pin_memory/num_workers
    and set shuffle deterministically when needed.

    Parameters
    ----------
    df : pandas.DataFrame
    batch_size : int
    mu, sigma : Optional[np.ndarray]
        Per-lead mean/std for normalization.
    shuffle : bool
    device_type : {"cpu","cuda","mps"}
        Used by torch_loader_kwargs to pick performant defaults.

    Returns
    -------
    torch.utils.data.DataLoader
    """
    ds = _make_tensor_dataset(df, mu, sigma)
    return torch.utils.data.DataLoader(
        ds, **torch_loader_kwargs(bool(shuffle), int(batch_size), str(device_type))
    )


# -------------------------------------------------------------------------
# Freeze state tracker
# -------------------------------------------------------------------------
@dataclass
class FreezeState:
    """Track client-side freezing dynamics across FL rounds.

    Attributes
    ----------
    best_loss : float
        Best (lowest) unweighted validation loss observed so far.
    best_acc : float
        Best validation accuracy observed so far.
    no_improve : int
        Count of consecutive "no improvement" steps (for patience scheduling).
    patience : int
        Steps to wait before adjusting freeze target (from FREEZE_CFG["patience"]).
    min_delta : float
        Minimum change in loss to qualify as improvement (FREEZE_CFG["min_delta"]).
    last_frozen : int
        Number of frozen layers in the last round (for logging/visibility).
    """
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
    """Federated learning client with reactive layer freezing and FedProx.

    Lifecycle
    ---------
    __init__:
      • Seed & device
      • Load metadata; map labels to SUPERCLASSES; filter to single-label rows
      • Global stratified patient split → train/val/test (same for all clients)
      • Partition global train among clients by patient-id (Dirichlet unevenness)
      • Compute local per-lead normalization stats from *local* train
      • Build model via create_model(); set up class weighting for CE
      • Optional: hyperparameter tuning (CV or reuse cached best)
      • Optional: head-only warm start for small clients

    Flower hooks:
      get_parameters / set_parameters / fit / evaluate

    Notes
    -----
    • FedProx adds μ/2 * Σ‖w - w_global‖² to the training loss if FEDPROX_MU > 0.
    • Freezing acts on model.features if present, keeping heads trainable.
    """

    def __init__(self, cid: int):
        # --- Setup -------------------------------------------------------
        self.cid = int(cid)
        set_seed(SEED + self.cid)
        sanitize_mps_env()
        self.device = pick_device()

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
        # Distribute patient_ids across clients; non-iid via Dirichlet proportions.
        patients = train_g.patient_id.unique()
        rng = np.random.RandomState(SEED)
        rng.shuffle(patients)

        if CLIENTS <= 1:
            buckets = [patients]
        else:
            # Dirichlet proportions for uneven splits -> floor to counts, then fill remainder.
            props = rng.dirichlet(alpha=[0.8] + [0.4]*(CLIENTS-1))
            counts = np.floor(props * len(patients)).astype(int)
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
        # Back-of-envelope estimate of samples processed per record.
        n_records = len(self.train_df)
        samples_per_record = 10 * SAMPLE_RATE * 12  # 10s * Hz * 12 leads
        total_samples = int(n_records * samples_per_record)
        hours = (n_records * 10.0) / 3600.0
        print(
            f"[Client {self.cid}] train records={n_records:,} "
            f"≈ {hours:.1f} h ECG | samples={total_samples:,} @ {SAMPLE_RATE} Hz"
        )

        # --- Local normalization stats ----------------------------------
        # Computed in (12, T) space; tensors are transposed to (T, 12) after normalization.
        self.mu, self.sigma = compute_perlead_norm_stats(self.train_df)  # (12,), (12,)

        # --- Model -------------------------------------------------------
        # Factory builds model consistent with N_CLASSES and MODEL config.
        self.model = create_model(
            n_classes=N_CLASSES,
            model_type=MODEL["type"],
            hidden=MODEL.get("lstm_hidden", 128),
            layers=MODEL.get("lstm_layers", 1),
            bidir=MODEL.get("bidirectional", True),
        ).to(self.device)

        # --- Class-weighted CE for training; plain CE for eval ----------
        # Inverse-frequency weights per class with re-normalization among present classes.
        counts_series = self.train_df["y"].value_counts()
        counts = np.array([int(counts_series.get(cname, 0)) for cname in CLASSES], dtype=np.int64)
        if counts.sum() > 0:
            total = counts.sum()
            w = np.zeros_like(counts, dtype=np.float32)
            for i, c in enumerate(counts):
                if c > 0:
                    w[i] = total / (N_CLASSES * float(c))
                else:
                    w[i] = 0.0
            present = w > 0
            if present.any():
                w[present] *= (present.sum() / float(w[present].sum()))
            else:
                w[:] = 1.0
        else:
            w = np.ones(N_CLASSES, dtype=np.float32)

        class_weights = torch.tensor(w, dtype=torch.float32, device=self.device)
        self.ce_train = nn.CrossEntropyLoss(weight=class_weights)  # used in training
        self.ce_eval  = nn.CrossEntropyLoss()                      # used in val/test
        print(f"[Client {self.cid}] class weights:", {c: float(v) for c, v in zip(CLASSES, w)})

        self.state = FreezeState()

        # --- Hyperparams (subject to tuning) ----------------------------
        self.lr = float(LR)
        self.batch_size = int(BATCH_SIZE)
        self.local_epochs = int(EPOCHS_LOCAL)
        self.fedprox_mu = float(FEDPROX_MU)

        # --- Optional client-side CV / cached reuse ---------------------
        # If TUNING.enabled: run k-fold CV on local train_df or reuse cached best.
        self.hp_source = "default"  # "default" | "cached" | "tuned"
        outdir = RESULTS_DIR / "tuning" / EXPERIMENT["run_name"]
        best_path = outdir / f"client{self.cid}_best.json"

        def _apply(hp: dict, source: str):
            """Apply tuned/cached hyperparameters to the client state."""
            self.lr = float(hp.get("lr", self.lr))
            self.batch_size = int(hp.get("batch", self.batch_size))
            self.local_epochs = int(hp.get("epochs", self.local_epochs))
            self.fedprox_mu = float(hp.get("fedprox", self.fedprox_mu))
            self.hp_source = source

        if TUNING.get("enabled", False):
            from src_Connection import run_client_cv
            outdir.mkdir(parents=True, exist_ok=True)

            # Reuse cached results if allowed and present.
            if TUNING.get("reuse_cached_if_exists", True) and best_path.exists():
                try:
                    best = (json.loads(best_path.read_text()) or {}).get("params", {})
                    if best:
                        _apply(best, "cached")
                        print(f"[Client {self.cid}] Using cached tuned HPs → "
                              f"lr={self.lr}, batch={self.batch_size}, epochs={self.local_epochs}, μ={self.fedprox_mu}")
                except Exception:
                    pass

            # Otherwise, launch CV and persist best.
            if self.hp_source == "default":
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
            # Reuse cached best even when CV is disabled (explicit opt-in).
            try:
                best = (json.loads(best_path.read_text()) or {}).get("params", {})
                if best:
                    _apply(best, "cached")
                    print(f"[Client {self.cid}] Reusing cached tuned HPs (CV off) → "
                          f"lr={self.lr}, batch={self.batch_size}, epochs={self.local_epochs}, μ={self.fedprox_mu}")
            except Exception:
                pass

        # --- Optional head-only warm start for tiny clients -------------
        # If small client (few local records) and freezing enabled, freeze feature extractor
        # and train only the head initially (last module in model.head).
        if EXPERIMENT.get("freeze_enabled", False) and len(self.train_df) < int(FREEZE_THRESHOLD):
            print(f"[Client {self.cid}] Head-only warm start (small client).")
            if hasattr(self.model, "features"):
                for p in self.model.features.parameters():
                    p.requires_grad = False
            if hasattr(self.model, "head"):
                for j, module in enumerate(self.model.head):
                    req = (j == len(self.model.head) - 1)
                    for p in getattr(module, "parameters", lambda: [])():
                        p.requires_grad = req

        print(f"[Client {self.cid}] Model: {MODEL['type']} on {self.device} | HP source: {self.hp_source}")

    # ---------------------------------------------------------------------
    # Flower API
    # ---------------------------------------------------------------------
    def get_parameters(self, config: Optional[dict] = None):
        """Serialize model parameters to a list of NumPy arrays for Flower."""
        return [p.detach().cpu().numpy() for _, p in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        """Load model parameters from a list of NumPy arrays (Flower → PyTorch)."""
        state = self.model.state_dict()
        for k, v in zip(state.keys(), parameters):
            state[k] = torch.tensor(v)
        self.model.load_state_dict(state, strict=True)

    # ---------------------------------------------------------------------
    # Freezing policy (hybrid: client size + reactive)
    # ---------------------------------------------------------------------
    def _apply_freeze_policy(self, round_num: int, loss_local: float, acc_local: float):
        """Determine how many 'features' layers to freeze for this round.

        Policy
        ------
        • Disabled if EXPERIMENT["freeze_enabled"] is False or model lacks .features
        • If client is large (≥ 2 * FREEZE_THRESHOLD): train all layers
        • If small (< FREEZE_THRESHOLD): freeze more early layers, then gradually
          unfreeze over rounds up to FREEZE_CFG["unfreeze_after"]
        • Otherwise: freeze a quarter of the features as a middle ground
        • Reactive adjustment:
            - If degraded (val loss up or acc down), increase frozen count after patience
            - If improved, decrease frozen count after patience

        Updates self.state and sets requires_grad on each layer accordingly.

        Returns
        -------
        dict : diagnostic info for logging.
        """
        if not EXPERIMENT.get("freeze_enabled", False):
            for p in self.model.parameters():
                p.requires_grad = True
            self.state.last_frozen = 0
            return {"policy": "disabled", "frozen_layers": 0}

        if not hasattr(self.model, "features"):
            return {"policy": "unavailable", "frozen_layers": 0}

        total_layers = len(list(self.model.features))
        n_train = len(self.train_df)

        # Base target depends on client scale.
        if n_train >= 2 * int(FREEZE_THRESHOLD):
            target = 0
        elif n_train < int(FREEZE_THRESHOLD):
            target = int(total_layers * max(0.0, 1 - round_num / float(FREEZE_CFG["unfreeze_after"])))
        else:
            target = max(0, total_layers // 4)

        # Reactive adjustments based on validation dynamics.
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

        # Apply requires_grad according to target.
        for i, layer in enumerate(self.model.features):
            for p in layer.parameters():
                p.requires_grad = (i >= target)

        self.state.last_frozen = target
        self.state.best_loss = min(self.state.best_loss, loss_local)
        self.state.best_acc = max(self.state.best_acc, acc_local)

        return {"policy": "dynamic", "frozen_layers": target,
                "improved": improved, "degraded": degraded, "no_improve": self.state.no_improve}

    # ---------------------------------------------------------------------
    # Training / Evaluation on local data
    # ---------------------------------------------------------------------
    def fit(self, parameters, config):
        """Flower hook: perform local training for this round.

        Steps
        -----
        1) Load global parameters from server.
        2) Pre-evaluate on local val to update freezing policy.
        3) Train for `self.local_epochs` with class-weighted CE (+FedProx if enabled).
        4) Post-evaluate on local val; update bests; log metrics.
        5) Return updated parameters and the number of training examples.

        Returns
        -------
        (parameters, num_examples, metrics_dict)
        """
        self.set_parameters(parameters)
        round_num = int(config.get("round", 1))

        # Pre-eval (val) to update freezing plan — use UNWEIGHTED eval loss
        loss_before, acc_before = self._eval_on(self.val_df)
        self._apply_freeze_policy(round_num, loss_before, acc_before)

        # Train
        t0 = time.perf_counter()
        loader = _loader_for(
            self.train_df, batch_size=self.batch_size, mu=self.mu, sigma=self.sigma,
            shuffle=True, device_type=self.device.type
        )
        opt = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                         lr=self.lr, weight_decay=1e-4)

        # FedProx snapshot of global weights (w0) for proximal term.
        global_params = [p.detach().clone() for p in self.model.parameters()] if self.fedprox_mu > 0 else None

        self.model.train()
        for _ in range(self.local_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                # Weighted CE during training.
                loss = self.ce_train(logits, yb)
                if self.fedprox_mu > 0 and global_params is not None:
                    prox = sum(torch.sum((w - w0) ** 2) for w, w0 in zip(self.model.parameters(), global_params))
                    loss = loss + (self.fedprox_mu / 2.0) * prox
                loss.backward()
                opt.step()

        wall_s = time.perf_counter() - t0
        # Post-train validation — UNWEIGHTED loss
        loss_after, acc_after = self._eval_on(self.val_df)

        # Track bests for freezing dynamics
        self.state.best_loss = min(self.state.best_loss, loss_after)
        self.state.best_acc = max(self.state.best_acc, acc_after)

        self._log_metrics(round_num, acc_after, loss_after, wall_s)
        print(f"[Client {self.cid}] R{round_num}: acc={acc_after:.4f}, loss={loss_after:.4f}, "
              f"time={wall_s:.1f}s, frozen_layers={self.state.last_frozen}")

        return self.get_parameters({}), len(self.train_df), {"accuracy": float(acc_after)}

    def evaluate(self, parameters, config):
        """Flower hook: evaluate on the held-out *local test* split.

        Returns
        -------
        (loss, num_examples, metrics_dict)
            metrics_dict includes overall accuracy and confusion matrix entries
            as keys cm_i_j for i,j in 0..N_CLASSES-1.
        """
        self.set_parameters(parameters)
        loader = _loader_for(
            self.test_df, batch_size=128, mu=self.mu, sigma=self.sigma,
            shuffle=False, device_type=self.device.type
        )
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0

        cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                # Unweighted CE for evaluation fairness.
                loss_sum += float(self.ce_eval(logits, yb).item()) * len(yb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum())
                total += len(yb)
                for t, p in zip(yb.view(-1).cpu().numpy(), pred.view(-1).cpu().numpy()):
                    cm[int(t), int(p)] += 1

        loss = (loss_sum / total) if total > 0 else 0.0
        acc = (correct / total) if total > 0 else 0.0

        metrics = {"accuracy": float(acc)}
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                metrics[f"cm_{i}_{j}"] = float(cm[i, j])

        return float(loss), len(self.test_df), metrics

    def _eval_on(self, df) -> Tuple[float, float]:
        """Evaluate on an arbitrary local dataframe using *unweighted* CE.

        Used for:
        • Pre-/post-train validation (val_df)
        • Quick health checks on arbitrary splits

        Returns
        -------
        (loss_mean, accuracy)
        """
        if len(df) == 0:
            return 0.0, 0.0
        loader = _loader_for(
            df, batch_size=128, mu=self.mu, sigma=self.sigma,
            shuffle=False, device_type=self.device.type
        )
        self.model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss_sum += float(self.ce_eval(logits, yb).item()) * len(yb)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum())
                total += len(yb)
        return (loss_sum / total if total else 0.0), (correct / total if total else 0.0)

    # ---------------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------------
    def _log_metrics(self, round_num: int, acc: float, loss: float, wall_s: float):
        """Append per-round client metrics to a CSV under RESULTS_DIR.

        File layout
        -----------
        • When TUNING.log_phase and log_mode == "separate":
            <RESULTS_DIR>/<run_name>_<phase>.csv
        • Otherwise:
            <RESULTS_DIR>/<run_name>.csv (includes 'phase' column)

        Columns
        -------
        client_id, round, accuracy, loss, frozen_layers,
        is_frozen, wall_time_sec, trainable_params, [phase?]
        """
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
    """CLI entrypoint for a single federated client process.

    Usage
    -----
    python Client.py --cid 0

    Notes
    -----
    • The server address comes from FL_SERVER_ADDRESS in config.py.
    • Launch multiple client processes with different --cid values (0..CLIENTS-1).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID (0..CLIENTS-1)")
    args = parser.parse_args()

    client = PTBClient(args.cid)

    fl.client.start_client(
        server_address=str(FL_SERVER_ADDRESS),
        client=client,  # NumPyClient instance
    )


if __name__ == "__main__":
    main()