"""
models.py — ECG Classification Architectures
--------------------------------------------

Defines neural network architectures used for ECG signal classification
within the Sustainable AI in Healthcare project (DSP5100).

Supported model types:
  • CNN  — Compact 1D convolutional network (default)
  • LSTM — Lightweight recurrent model for sequential ECG input
  • RNN  — Simple tanh RNN
  • ANN  — Mean-pooled MLP over leads

Input: either [N, T, C] or [N, C, T] with C=12 (12-lead ECG).
Output: logits sized [N, n_classes] (or [N, 1] when binary=True).

Where this module is used
-------------------------
• src.Centralized
    - Calls create_model(...) for chosen deep models (CNN/RNN/LSTM/ANN).
    - Uses the `features` and `head` attributes for potential freeze policies.
• src.Client (federated)
    - Calls create_model(...) and may freeze layers dynamically via .features/.head.
• src.Experiments
    - Calls create_model(...) inside K-Fold CV loops.
• src.results_visualization
    - Expects forward(...) to return logits; adapters apply Sigmoid/Argmax externally.

Conventions and contracts
-------------------------
• Forward returns raw logits (no Sigmoid/Softmax). Binary tasks use a single logit.
• All models accept either (N,T,C) or (N,C,T). We standardize internally.
• Freezing: CNN exposes nn.Sequential `features` and `head`. LSTM/RNN/ANN expose
  `features` (ModuleList/Identity) and `head` for compatibility with client-side
  freezing (see Client._apply_freeze_policy).
"""

from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn

# Classical ML baseline (kept for optional use)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.multiclass import OneVsRestClassifier


# -------------------------------------------------------------------------
# Classical ML baseline (imputed + scaled + OvR logistic regression)
# -------------------------------------------------------------------------
def create_logistic_baseline(class_weight: str | None = "balanced") -> Pipeline:
    """
    Create a simple classical baseline: Impute → Scale → OvR Logistic Regression.

    Why this exists
    ---------------
    • Useful for quick baselines on engineered feature tables produced by
      data_loader.make_feature_table_legacy or make_feature_table.
    • OvR wrapper avoids sklearn>=1.5 multi_class API quirks.

    Parameters
    ----------
    class_weight : Optional[str]
        'balanced' reweights classes inversely to frequency; None disables weighting.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Steps: SimpleImputer(median) → StandardScaler → OneVsRest(LogisticRegression)

    Connected to
    ------------
    • Not used in deep pipelines directly, but handy for feature-only experiments.
    """
    base_lr = LogisticRegression(
        max_iter=500,
        solver="lbfgs",
        class_weight=class_weight,  # set to None to disable class balancing
    )
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", OneVsRestClassifier(base_lr, n_jobs=-1)),
    ])


# -------------------------------------------------------------------------
# Helpers: input layout handling + generic classification head
# -------------------------------------------------------------------------
def _to_NCT(x: torch.Tensor) -> torch.Tensor:
    """
    Ensure input is [N, C, T] where C=12.

    Accepts
    -------
    • [N, T, C] with C=12 → permutes to [N, C, T]
    • [N, C, T] with C=12 → returned as-is

    Raises
    ------
    ValueError if the channel dimension is not 12.

    Connected to
    ------------
    • Used by all models to standardize internal layout irrespective of the
      dataset return shape (ECGDataset returns (T,C); loader stacks to (N,T,C)).
    """
    if x.dim() != 3:
        raise ValueError(f"Expected a 3D tensor [N,T,C] or [N,C,T], got {tuple(x.shape)}")

    N, A, B = x.shape
    if A == 12:           # [N, C, T]
        return x
    if B == 12:           # [N, T, C]
        return x.permute(0, 2, 1)

    raise ValueError(
        f"Expected 12 ECG leads on channel axis; got shape {tuple(x.shape)}. "
        "If your data is not 12-lead, adjust the models accordingly."
    )


def _time_dim(_: torch.Tensor) -> int:
    """Return the time dimension index once in [N, C, T] (always -1)."""
    return -1


def _cls_head(in_features: int, n_classes: int, binary: bool = False) -> nn.Module:
    """
    Build a final linear classification layer.

    Notes
    -----
    • For binary=True, outputs a single logit [N,1] (use BCEWithLogitsLoss).
    • For multi-class, outputs [N, n_classes] (use CrossEntropyLoss).
    """
    return nn.Linear(in_features, 1 if binary else n_classes)


# -------------------------------------------------------------------------
# CNN architecture
# -------------------------------------------------------------------------
class TinyECGCNN(nn.Module):
    """
    Robust 1D CNN for 12-lead ECG classification.

    IO contract
    -----------
    • Accepts [N,T,C] or [N,C,T]; internally standardized to [N,C,T].
    • Returns logits [N, n_classes] (or [N,1] if binary).

    Freezing
    --------
    • Exposes `features` (nn.Sequential of conv blocks) and `head` (MLP+cls)
      so Client._apply_freeze_policy can freeze lower layers.

    Connected to
    ------------
    • Used by Centralized.run_deep_models and Experiments.run_kfold_all via create_model("cnn", ...).
    """
    def __init__(self, n_classes: int, input_c: int = 12, binary: bool = False):
        super().__init__()
        self.binary = binary

        # Feature extractor: progressively increases channels and downsamples T
        self.features = nn.Sequential(
            nn.Conv1d(input_c, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),   # T → T/4

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),   # T → T/8

            nn.Conv1d(64, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),

            nn.Conv1d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(160),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),  # shape → [N, 160, 1]
        )

        # Classification head: compact MLP + final linear
        self.head = nn.Sequential(
            nn.Flatten(),              # [N, 160]
            nn.Linear(160, 96),
            nn.ReLU(),
            nn.Dropout(0.5),
            _cls_head(96, n_classes, binary),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_NCT(x)                # [N, C, T]
        z = self.features(x)          # [N, 160, 1]
        return self.head(z)           # [N, n_classes] or [N, 1]


# -------------------------------------------------------------------------
# LSTM architecture
# -------------------------------------------------------------------------
class TinyECGLSTM(nn.Module):
    """
    Lightweight LSTM model for sequential ECG processing.

    Design
    ------
    • Optional conv 'stem' to downsample/denoise before the RNN (use_stem=True).
    • batch_first=True LSTM → input [N, T, C'].
    • Temporal mean pooling of LSTM outputs before MLP head.

    Freezing
    --------
    • `features` is a ModuleList([stem_or_identity, rnn]) to allow per-block freezing.
    • `head` is a small MLP + final linear.

    Connected to
    ------------
    • Centralized/Experiments via create_model("lstm", ...).
    • Client freezing heuristics iterate over model.features.
    """
    def __init__(
        self,
        n_classes: int,
        input_c: int = 12,
        hidden: int = 128,
        layers: int = 1,
        bidir: bool = True,
        binary: bool = False,
        use_stem: bool = False,
    ):
        super().__init__()
        self.binary = binary
        self.use_stem = bool(use_stem)

        # Optional convolutional stem: [N,C,T] → [N,32,T/4]
        if self.use_stem:
            stem = nn.Sequential(
                nn.Conv1d(input_c, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            rnn_in = 32
        else:
            stem = nn.Identity()
            rnn_in = input_c

        # LSTM layers (batch_first: [N,T,C'])
        rnn = nn.LSTM(
            input_size=rnn_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidir,
        )
        rnn_out = hidden * (2 if bidir else 1)

        # Expose blocks for freezing
        self.features = nn.ModuleList([stem, rnn])

        # Head: project pooled LSTM output
        self.head = nn.Sequential(
            nn.Linear(rnn_out, 96),
            nn.ReLU(),
            nn.Dropout(0.5),
            _cls_head(96, n_classes, binary),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_NCT(x)           # [N, C, T]
        z = self.features[0](x)  # stem or Identity: [N, C', T]
        z = z.permute(0, 2, 1)   # → [N, T, C']
        z, _ = self.features[1](z)  # LSTM outputs [N,T,H]
        z = z.mean(dim=1)        # temporal mean pooling → [N,H]
        return self.head(z)


# -------------------------------------------------------------------------
# Simple RNN (tanh) architecture
# -------------------------------------------------------------------------
class RNNSimple(nn.Module):
    """
    Two-layer vanilla RNN with dropout in between.

    Pipeline
    --------
    • Standardize to [N,C,T] → permute to [N,T,C].
    • RNN(tanh) → Dropout → RNN(tanh) → last-timestep → head.

    Freezing
    --------
    • `features` = [rnn1, rnn2] to enable partially frozen recurrent stacks.

    Connected to
    ------------
    • Centralized/Experiments via create_model("rnn", ...).
    """
    def __init__(self, n_classes: int, input_c: int = 12, hidden1: int = 128,
                 hidden2: int = 64, binary: bool = False):
        super().__init__()
        self.binary = binary
        self.rnn1 = nn.RNN(input_c, hidden1, nonlinearity="tanh", batch_first=True)
        self.drop = nn.Dropout(0.15)
        self.rnn2 = nn.RNN(hidden1, hidden2, nonlinearity="tanh", batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden2, 96),
            nn.ReLU(),
            _cls_head(96, n_classes, binary),
        )
        self.features = nn.ModuleList([self.rnn1, self.rnn2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_NCT(x)              # [N, C, T]
        x = x.permute(0, 2, 1)      # [N, T, C]
        x, _ = self.rnn1(x)
        x = self.drop(x)
        x, _ = self.rnn2(x)
        x = x[:, -1, :]             # last timestep → [N, hidden2]
        return self.head(x)


# -------------------------------------------------------------------------
# ANN with temporal mean pooling
# -------------------------------------------------------------------------
class ANNPooled(nn.Module):
    """
    Simple MLP over per-lead features obtained via temporal mean pooling.

    Approach
    --------
    • Convert [N,C,T] to [N,C] via mean over time (per-lead summary).
    • MLP( C → 256 → 128 ) → Linear to outputs.

    Freezing
    --------
    • `features` is Identity to maintain a consistent interface
      with other models during freezing policies.

    Connected to
    ------------
    • Centralized/Experiments via create_model("ann", ...).
    """
    def __init__(self, n_classes: int, input_c: int = 12, binary: bool = False):
        super().__init__()
        self.binary = binary
        self.features = nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(input_c, 256), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(256, 128),     nn.ReLU()
        )
        self.head = nn.Sequential(_cls_head(128, n_classes, binary))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_NCT(x)                      # [N, C, T]
        tdim = _time_dim(x)                 # -1
        x = x.mean(dim=tdim)                # [N, C] — average over time
        z = self.mlp(x)                     # [N, 128]
        return self.head(z)                 # [N, n_classes] or [N,1]


# -------------------------------------------------------------------------
# Unified factory (compatible with all current call sites)
# -------------------------------------------------------------------------
def create_model(*args: Any, **kwargs: Any) -> nn.Module:
    """
    Flexible factory to satisfy both usages in the codebase:
      - create_model(model_type, n_classes, binary=..., **extra)
      - create_model(n_classes=<int>, model_type=<str>, **extra)
      - create_model(n_classes=<int>)  # defaults to 'cnn'

    Supported model_type values: 'cnn', 'lstm', 'rnn', 'ann'
    Extra kwargs are forwarded to specific constructors (e.g., hidden/layers/bidir for LSTM).

    Connected to
    ------------
    • Centralized.run_deep_models(...)
    • Client.__init__(...) for federated runs
    • Experiments.run_kfold_all(...)
    • tuning._model_ctor() for CV
    """
    model_type = kwargs.pop("model_type", None)

    # Parse both positional and keyword combos
    if len(args) == 0:
        n_classes = kwargs.pop("n_classes")
    elif len(args) == 1:
        if isinstance(args[0], str):
            model_type = args[0]
            n_classes = kwargs.pop("n_classes")
        else:
            n_classes = int(args[0])
    elif len(args) >= 2:
        if isinstance(args[0], str):
            model_type = args[0]
            n_classes = int(args[1])
        else:
            n_classes = int(args[0])
            model_type = str(args[1])
    else:
        raise ValueError("create_model: could not parse arguments")

    model_type = (model_type or "cnn").lower()
    binary = bool(kwargs.pop("binary", False))
    input_c = int(kwargs.pop("input_c", 12))

    if model_type == "cnn":
        return TinyECGCNN(n_classes, input_c=input_c, binary=binary)
    if model_type == "lstm":
        return TinyECGLSTM(n_classes, input_c=input_c, binary=binary, **kwargs)
    if model_type == "rnn":
        return RNNSimple(n_classes, input_c=input_c, binary=binary, **kwargs)
    if model_type == "ann":
        return ANNPooled(n_classes, input_c=input_c, binary=binary, **kwargs)

    raise ValueError(f"Unknown model_type: {model_type}")


__all__ = [
    "create_model",
    "create_logistic_baseline",
    "TinyECGCNN",
    "TinyECGLSTM",
    "RNNSimple",
    "ANNPooled",
]
