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

def create_cnn_model(n_classes: int) -> nn.Module:
    return TinyECGCNN(n_classes)

