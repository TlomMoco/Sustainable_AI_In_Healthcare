"""
models.py — ECG Classification Architectures
--------------------------------------------

Defines neural network architectures used for ECG signal classification
within the Sustainable AI in Healthcare project (DSP5100).

Supported model types:
  • CNN  — Compact 1D convolutional network (default)
  • LSTM — Lightweight recurrent model for sequential ECG input

Both models accept a variable number of input leads (n_leads) with input
shape (B, n_leads, T) and output logits for the 5 superclasses.
"""

from __future__ import annotations
import torch
import torch.nn as nn


# -------------------------------------------------------------------------
# CNN architecture
# -------------------------------------------------------------------------
class TinyECGCNN(nn.Module):
    """
    A small but expressive 1D CNN for ECG classification.

    Input:  (B, in_ch, T)
    Output: (B, n_classes)
    """

    def __init__(self, n_classes: int, in_ch: int = 12):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — early temporal reduction
            nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3),
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
        z = self.features(x)     # (B, 160, 1)
        return self.head(z)      # (B, n_classes)


# -------------------------------------------------------------------------
# LSTM architecture
# -------------------------------------------------------------------------
class TinyECGLSTM(nn.Module):
    """
    Lightweight LSTM model for sequential ECG processing.

    Input:  (B, in_ch, T)
    Output: (B, n_classes)

    The model exposes self.features as a ModuleList to allow
    dynamic freezing during federated training.
    """

    def __init__(
        self,
        n_classes: int,
        in_ch: int = 12,
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
                nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
            )
            rnn_in = 32
        else:
            self.stem = nn.Identity()
            rnn_in = in_ch

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
        # x: (B, C, T)
        z = self.stem(x)  # (B, C, T')
        z = z.permute(0, 2, 1)  # (B, T', C)

        # LSTM returns (out, (h, c))
        out, hc = self.rnn(z)
        if isinstance(hc, tuple):
            h, c = hc
        else:
            # safety fallback, but LSTM returns tuple in PyTorch
            h = hc

        # h shape: (num_layers * num_directions, B, hidden)
        if self.rnn.bidirectional:
            # concat last layer's forward/backward states
            h_last = torch.cat([h[-2], h[-1]], dim=1)  # (B, 2*hidden)
        else:
            h_last = h[-1]  # (B, hidden)

        return self.head(h_last)


# -------------------------------------------------------------------------
# CNN-LSTM hybrid
# -------------------------------------------------------------------------
# --- reuse TinyECGLSTM with convolutional stem enabled ---
class CNNLSTM(TinyECGLSTM):
    """
    Thin alias that makes it explicit we are running a CNN+LSTM hybrid.
    Keeps the same features layout [stem, rnn] for freezing.
    """
    def __init__(self, n_classes: int, in_ch: int = 12, **kwargs):
        # Ensure the convolutional stem is active
        kwargs.setdefault("use_stem", True)
        super().__init__(n_classes, in_ch=in_ch, **kwargs)


# -------------------------------------------------------------------------
# GRU architecture (mirrors LSTM; supports optional conv stem)
# -------------------------------------------------------------------------
class TinyECGGRU(nn.Module):
    """
    Lightweight GRU model for sequential ECG processing.

    Input:  (B, in_ch, T)
    Output: (B, n_classes)

    Exposes self.features as [stem (or Identity), rnn] to keep freezing logic
    identical to TinyECGLSTM.
    """
    def __init__(
        self,
        n_classes: int,
        in_ch: int = 12,
        hidden: int = 128,
        layers: int = 1,
        bidir: bool = True,
        use_stem: bool = False,
    ):
        super().__init__()
        self.use_stem = use_stem

        # Optional convolutional stem (same interface as LSTM)
        if self.use_stem:
            self.stem = nn.Sequential(
                nn.Conv1d(in_ch, 32, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(),
            )
            rnn_in = 64
        else:
            self.stem = nn.Identity()
            rnn_in = in_ch

        # GRU encoder
        self.rnn = nn.GRU(
            input_size=rnn_in,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidir,
        )
        rnn_out = hidden * (2 if bidir else 1)

        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(rnn_out, n_classes),
        )

        # Keep the same freezing handle as LSTM
        self.features = nn.ModuleList([self.stem, self.rnn])

    def forward(self, x):
        # x: (B, C, T) -> stem expects (B, C, T)
        z = self.stem(x)                 # (B, C', T)
        z = z.permute(0, 2, 1)           # (B, T, C')
        z, _ = self.rnn(z)               # (B, T, H')
        z = z.mean(dim=1)                # simple temporal pooling
        return self.head(z)              # (B, n_classes)


# ---- TinyECG-MLP (temporal avg pool + MLP) -----------------------------
class TinyECGMLP(nn.Module):
    def __init__(self, n_classes: int, in_ch: int = 12, hidden: int = 128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)  # (B,C,T) -> (B,C,1)
        # Ingen “features” å fryse, men behold grensesnittet:
        self.features = nn.ModuleList([])
        self.head = nn.Sequential(
            nn.Flatten(),              # (B,C)
            nn.Linear(in_ch, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pool(x)     # (B,C,1)
        return self.head(z)  # (B,n_classes)



# -------------------------------------------------------------------------
# Model factory
# -------------------------------------------------------------------------
def create_model(
    n_classes: int,
    model_type: str = "cnn",
    n_leads: int = 12,
    **kwargs
) -> nn.Module:
    """
    Factory function for model creation.

    Args:
        n_classes: number of output classes (5 for PTB-XL)
        model_type: 'cnn' or 'lstm'
        n_leads: number of input channels/leads to use
        kwargs: extra parameters (hidden, layers, bidir, use_stem, etc.)

    Returns:
        torch.nn.Module: model instance
    """
    if model_type == "cnn":
        return TinyECGCNN(n_classes, in_ch=n_leads)
    elif model_type == "lstm":
        return TinyECGLSTM(n_classes, in_ch=n_leads, **kwargs)
    elif model_type in ("cnn_lstm", "lstm_cnn"):
        # Explicit CNN-LSTM model (same freezing behavior as TinyECGLSTM with stem)
        return CNNLSTM(n_classes, in_ch=n_leads, **kwargs)
    elif model_type == "gru":
        return TinyECGGRU(n_classes, in_ch=n_leads, **kwargs)
    elif model_type in ("cnn_gru", "gru_cnn"):
        kwargs.setdefault("use_stem", True)
        return TinyECGGRU(n_classes, in_ch=n_leads, **kwargs)
    elif model_type == "mlp":
        return TinyECGMLP(n_classes, in_ch=n_leads, hidden=kwargs.get("mlp_hidden", 128))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
