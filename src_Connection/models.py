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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



# -------------------------------------------------------------------------
# Classical ML baseline (optional reference)
# -------------------------------------------------------------------------
def create_logistic_baseline() -> Pipeline:
    """Return a simple baseline model using logistic regression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=200, multi_class="ovr"))
    ])


# -------------------------------------------------------------------------
# CNN architecture
# -------------------------------------------------------------------------
class TinyECGCNN(nn.Module):
    """
    A small but expressive 1D CNN for 12-lead ECG classification.

    Input:  (B, 12, T)
    Output: (B, n_classes)
    """

    def __init__(self, n_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — early temporal reduction
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),   # → T/4 total reduction

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),   # → T/8

            # Block 3
            nn.Conv1d(64, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),

            # Block 4
            nn.Conv1d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Block 5
            nn.Conv1d(128, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(160),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),  # Global average pooling → (B, 160, 1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),          # → (B, 160)
            nn.Linear(160, 96),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(96, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        z = self.features(x)     # (B, 160, 1)
        return self.head(z)      # (B, n_classes)


# -------------------------------------------------------------------------
# LSTM architecture
# -------------------------------------------------------------------------
class TinyECGLSTM(nn.Module):
    """
    Lightweight LSTM model for sequential ECG processing.

    Input:  (B, 12, T)
    Output: (B, n_classes)

    The model exposes self.features as a ModuleList to allow
    dynamic freezing during federated training.
    """

    def __init__(
        self,
        n_classes: int,
        hidden: int = 128,
        layers: int = 1,
        bidir: bool = True,
        use_stem: bool = False,
    ):
        super().__init__()
        self.use_stem = use_stem

        # Optional convolutional stem for feature extraction
        if use_stem:
            self.stem = nn.Sequential(
                nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            rnn_in = 32
        else:
            self.stem = nn.Identity()
            rnn_in = 12

        # LSTM encoder
        self.rnn = nn.LSTM(
            input_size=rnn_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidir,
        )
        rnn_out = hidden * (2 if bidir else 1)

        # Fully connected head
        self.head = nn.Sequential(
            nn.Linear(rnn_out, 96),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(96, n_classes),
        )

        # Expose stem + RNN as 'features' for freezing policy
        self.features = nn.ModuleList([self.stem, self.rnn])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        z = self.stem(x)          # (B, C, T)
        z = z.permute(0, 2, 1)    # (B, T, C)
        z, _ = self.rnn(z)        # (B, T, H)
        z = z.mean(dim=1)         # Temporal pooling
        return self.head(z)


# -------------------------------------------------------------------------
# Model factory
# -------------------------------------------------------------------------
def create_model(n_classes: int, model_type: str = "cnn", **kwargs) -> nn.Module:
    """
    Factory function for model creation.

    Args:
        n_classes: number of output classes (5 for PTB-XL)
        model_type: 'cnn' or 'lstm'
        kwargs: extra parameters (hidden, layers, bidir, etc.)

    Returns:
        torch.nn.Module: model instance
    """
    if model_type == "cnn":
        return TinyECGCNN(n_classes)
    elif model_type == "lstm":
        return TinyECGLSTM(n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")








# ---------
# ## 12) Models + Heads (from notebook)
# ---------
import torch
from torch import nn
from . import config as CFG

def _head(in_features: int, n_classes: int, binary: bool = False):
    return nn.Linear(in_features, 1 if binary else n_classes)

class TinyECGCNN(nn.Module):
    def __init__(self, n_classes: int, input_c: int = 12, binary: bool = False):
        super().__init__()
        self.binary = binary
        self.avg2 = nn.AvgPool1d(2)
        self.body = nn.Sequential(
            nn.Conv1d(input_c, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),     nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64,128, 3, padding=1),     nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = _head(128, n_classes, binary)
    def forward(self, x):              # x: [N, T, C]
        x = x.permute(0,2,1)           # → [N, C, T]
        x = self.avg2(x)
        x = self.body(x).squeeze(-1)   # [N, 128]
        return self.fc(x)

class TinyECGLSTM(nn.Module):
    def __init__(self, n_classes: int, input_c: int = 12, binary: bool = False):
        super().__init__()
        self.binary = binary
        self.l1 = nn.LSTM(input_c, 128, batch_first=True)
        self.drop = nn.Dropout(0.15)
        self.l2 = nn.LSTM(128, 64, batch_first=True)
        self.fc1 = nn.Linear(64, 96)
        self.fc2 = _head(96, n_classes, binary)
    def forward(self, x):
        x = x[:, ::2, :]
        x, _ = self.l1(x); x = self.drop(x)
        x, _ = self.l2(x)
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class RNNsimple(nn.Module):
    # ---------
    # ## 12) Models — RNN (from notebook)
    # ---------
    def __init__(self, n_classes: int, input_c: int = 12, binary: bool = False):
        super().__init__()
        self.binary = binary
        self.rnn1 = nn.RNN(input_c, 128, batch_first=True)
        self.drop = nn.Dropout(0.15)
        self.rnn2 = nn.RNN(128, 64, batch_first=True)
        self.fc1  = nn.Linear(64, 96)
        self.fc2  = _head(96, n_classes, binary)
    def forward(self, x):
        x = x[:, ::2, :]
        x, _ = self.rnn1(x); x = self.drop(x)
        x, _ = self.rnn2(x)
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class ANNpooled(nn.Module):
    # ---------
    # ## 12) Models — ANN (from notebook)
    # ---------
    def __init__(self, n_classes: int, input_c: int = 12, binary: bool = False):
        super().__init__()
        self.binary = binary
        self.fc = nn.Sequential(
            nn.Linear(input_c, 256), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(256, 128),     nn.ReLU()
        )
        self.out = _head(128, n_classes, binary)
    def forward(self, x):
        x = x.mean(dim=1)              # global mean over time
        x = self.fc(x)
        return self.out(x)

# ---------
# ## 12) Factory (extended)
# ---------
def create_model(model_type: str, n_classes: int, *, input_c: int = 12, binary: bool = False):
    mt = str(model_type).lower()
    if mt == "cnn":  return TinyECGCNN(n_classes, input_c=input_c, binary=binary)
    if mt == "lstm": return TinyECGLSTM(n_classes, input_c=input_c, binary=binary)
    if mt == "rnn":  return RNNsimple(n_classes, input_c=input_c, binary=binary)
    if mt == "ann":  return ANNpooled(n_classes, input_c=input_c, binary=binary)
    raise ValueError(f"Unknown model_type: {model_type}")
