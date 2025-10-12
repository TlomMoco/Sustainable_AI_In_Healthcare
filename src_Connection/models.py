"""
models.py — ECG Classification Architectures
--------------------------------------------

Defines neural network architectures used for ECG signal classification
within the Sustainable AI in Healthcare project (DSP5100).

Supported model types:
  • CNN  — Compact 1D convolutional network (default)
  • LSTM — Lightweight recurrent model for sequential ECG input

Both models are designed for 12-lead ECG signals with input shape (12, T),
and output logits corresponding to the 5 diagnostic superclasses:
["NORM", "MI", "STTC", "HYP", "CD"]
"""

from __future__ import annotations
import torch
import torch.nn as nn

# Classical ML bits
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
    Logistic baseline with:
      • median imputation (handles NaNs)
      • standardization
      • One-vs-Rest wrapper (avoids multi_class deprecation warning in sklearn>=1.5)
      • optional class_weight='balanced' to help class imbalance
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
    Ensure input is [N, C, T]:
      - If x is [N, T, C], permute to [N, C, T]
      - If x is already [N, C, T], return as-is
    """
    if x.dim() != 3:
        raise ValueError(f"Expected a 3D tensor [N,T,C] or [N,C,T], got {tuple(x.shape)}")
    N, A, B = x.shape
    if A == 12:  # [N, C, T]
        return x
    if B == 12:  # [N, T, C]
        return x.permute(0, 2, 1)
    return x  # fallback: assume already [N, C, T]


def _time_dim(x: torch.Tensor) -> int:
    """Return the index of the time dimension (after we may have NCT)."""
    return -1


def _cls_head(in_features: int, n_classes: int, binary: bool = False) -> nn.Module:
    return nn.Linear(in_features, 1 if binary else n_classes)


# -------------------------------------------------------------------------
# CNN architecture
# -------------------------------------------------------------------------
class TinyECGCNN(nn.Module):
    """
    Robust 1D CNN for 12-lead ECG classification.
    Accepts [N,T,C] or [N,C,T]; internally standardized to [N,C,T].
    Exposes `features` and `head` for freezing policies.
    """
    def __init__(self, n_classes: int, input_c: int = 12, binary: bool = False):
        super().__init__()
        self.binary = binary

        self.features = nn.Sequential(
            nn.Conv1d(input_c, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),   # → T/4

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),   # → T/8

            nn.Conv1d(64, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),

            nn.Conv1d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(160),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),  # → [N, 160, 1]
        )

        self.head = nn.Sequential(
            nn.Flatten(),              # → [N, 160]
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
    Accepts [N,T,C] or [N,C,T]; standardized internally.
    Exposes `features` (ModuleList) and `head` for freezing policies.
    """
    def __init__(self, n_classes: int, input_c: int = 12, hidden: int = 128,
                 layers: int = 1, bidir: bool = True, binary: bool = False, use_stem: bool = False):
        super().__init__()
        self.binary = binary
        self.use_stem = use_stem

        if use_stem:
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

        rnn = nn.LSTM(
            input_size=rnn_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidir,
        )
        rnn_out = hidden * (2 if bidir else 1)

        self.features = nn.ModuleList([stem, rnn])
        self.head = nn.Sequential(
            nn.Linear(rnn_out, 96),
            nn.ReLU(),
            nn.Dropout(0.5),
            _cls_head(96, n_classes, binary),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _to_NCT(x)          # [N, C, T]
        z = self.features[0](x) if self.use_stem else x
        if isinstance(self.features[0], nn.Sequential) and self.use_stem:
            z = z.permute(0, 2, 1)  # [N, C’, T] → [N, T, C’]
        else:
            z = z.permute(0, 2, 1)  # [N, C, T]  → [N, T, C]
        z, _ = self.features[1](z)  # LSTM
        z = z.mean(dim=1)           # temporal mean pooling
        return self.head(z)


# -------------------------------------------------------------------------
# Simple RNN (tanh) architecture
# -------------------------------------------------------------------------
class RNNSimple(nn.Module):
    """
    Two-layer vanilla RNN with dropout in between.
    Accepts [N,T,C] or [N,C,T]; standardized internally.
    Exposes `features` and `head`.
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
        x = x[:, -1, :]             # last timestep
        return self.head(x)


# -------------------------------------------------------------------------
# ANN with temporal mean pooling
# -------------------------------------------------------------------------
class ANNPooled(nn.Module):
    """
    Simple MLP over per-lead features obtained via temporal mean pooling.
    Accepts [N,T,C] or [N,C,T]. Exposes `features` (Identity) and `head`.
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
        x = x.mean(dim=tdim)                # [N, C] → average over time
        z = self.mlp(x)                     # [N, 128]
        return self.head(z)                 # [N, n_classes] or [N,1]


# -------------------------------------------------------------------------
# Unified factory (compatible with all current call sites)
# -------------------------------------------------------------------------
def create_model(*args, **kwargs) -> nn.Module:
    """
    Flexible factory to satisfy both usages in your codebase:
      - create_model(model_type, n_classes, binary=..., **extra)
      - create_model(n_classes=<int>, model_type=<str>, **extra)
      - create_model(n_classes=<int>)  # defaults to cnn

    Supported model_type values: 'cnn', 'lstm', 'rnn', 'ann'
    Extra kwargs are passed to the specific constructor (e.g., hidden/layers/bidir for LSTM).
    """
    model_type = kwargs.pop("model_type", None)

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