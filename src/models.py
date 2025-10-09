from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


@dataclass
class Shapes:
    n_classes: int


# ===== ML baseline =====
def create_logistic_baseline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=200, multi_class="ovr"))
    ])


# ===== Simple 1D CNN for 12‑lead ECG =====
class TinyECGCNN(nn.Module):
    def __init__(self, n_classes: int):

        super().__init__()
        self.features = nn.Sequential(
            # Block 1 (early reduction)
            nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),                         # ~ T/4 so far

            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),                         # ~ T/8

            # Block 3
            nn.Conv1d(64, 96, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(96), nn.ReLU(),

            # Block 4
            nn.Conv1d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),

            # Block 5
            nn.Conv1d(128, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(160), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),                 # (B, 160, 1)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(160, 96), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(96, n_classes),
        )

    def forward(self, x):
        # x: (B, 12, T)
        z = self.features(x)
        return self.head(z)


# ===== Simple LSTM for 12‑lead ECG =====
class TinyECGLSTM(nn.Module):
    """
    A lightweight LSTM classifier for 12-lead ECG.
    Exposes `self.features` as a ModuleList:
      [optional_conv_stem, lstm_block]  -> so the freeze policy can freeze by block.
    """
    def __init__(self, n_classes: int, hidden: int = 128, layers: int = 1, bidir: bool = True, use_stem: bool = False):
        super().__init__()
        self.use_stem = use_stem
        if use_stem:
            self.stem = nn.Sequential(
                nn.Conv1d(12, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.MaxPool1d(2),
            )
            rnn_in = 32
        else:
            self.stem = nn.Identity()
            rnn_in = 12

        self.rnn = nn.LSTM(
            input_size=rnn_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidir,
        )
        rnn_out = hidden * (2 if bidir else 1)

        self.head = nn.Sequential(
            nn.Linear(rnn_out, 96), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(96, n_classes),
        )

        # Expose blocks for freezing
        self.features = nn.ModuleList([self.stem, self.rnn])

    def forward(self, x):
        # x: (B, 12, T)
        z = self.stem(x)                 # (B, C, T)
        z = z.permute(0, 2, 1)           # (B, T, C)
        z, _ = self.rnn(z)               # (B, T, H)
        z = z.mean(dim=1)                # temporal average pooling
        return self.head(z)


def create_model(n_classes: int, model_type: str = "cnn", **kwargs) -> nn.Module:
    if model_type == "cnn":
        return TinyECGCNN(n_classes)
    elif model_type == "lstm":
        return TinyECGLSTM(n_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

