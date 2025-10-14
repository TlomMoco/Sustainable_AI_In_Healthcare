# src/data_preprocessing.py
"""
Lightweight dataset / dataloader adapters for PTB-XL deep models.

Exposes:
  • ECGDataset — returns (T, C) float32 tensors (models can permute as needed)
  • make_label_encoder — union-fit across train/test series
  • make_train_val_loaders — stratified split + DataLoaders
  • make_predict_loader — DataLoader for inference-only paths

Where this module plugs in
--------------------------
• src.Centralized
    - Uses ECGDataset to build stable train-eval loaders
    - Uses make_label_encoder and make_train_val_loaders for deep runs
• src.Client (federated)
    - Reimplements its own on-the-fly tensorization, but you can reuse ECGDataset
      for centralized/ablation-style client runs as well
• src.models
    - Assumes inputs shaped (B, T, C) or (B, C, T); models permute as needed
• src.data_loader
    - Provides load_waveform_np used by ECGDataset to materialize (T, C) arrays

Notes
-----
• T is the effective timesteps after downsampling cfg.DOWNSAMPLE_FACTOR.
• C is the number of ECG leads (default 12). We pad/truncate to target_leads.
• CNNs with Conv1d typically expect (B, C, T). RNN/LSTM often expect (B, T, C).
  This dataset returns (T, C); the model should permute inside forward().
"""

from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import SEQ_LEN, DOWNSAMPLE_FACTOR, BATCH_SIZE, SEED
from src.utils import torch_loader_kwargs
from src.data_loader import load_waveform_np


class ECGDataset(Dataset):
    """
    Memory-light dataset that loads WFDB records on demand and returns
    per-sample float32 tensors of shape (T, C).

    Shapes
    ------
    • (T, C): time-major with C leads (default 12)
      - CNN/Conv1d can permute to (C, T) in the model: x = x.permute(0, 2, 1)
      - RNN/LSTM often consume (B, T, C) directly, so no permute needed.

    Inputs
    ------
    paths : Iterable[str]
        WFDB record *base* paths (no extension), as produced by data_loader.make_feature_table.
    y : Optional[Iterable[int]]
        Integer-encoded labels aligned with `paths`. If None, returns only tensors (predict mode).
    seq_len : Optional[int]
        Base sequence length before downsampling. Defaults to config.SEQ_LEN.
    ds_factor : Optional[int]
        Downsample factor; effective T = seq_len // ds_factor. Defaults to config.DOWNSAMPLE_FACTOR.
    target_leads : int
        Ensure tensors have exactly this many leads by pad/truncate (default 12).

    Robustness
    ----------
    • If a record fails to load, returns a zeros tensor with the correct (T, C) shape.
    """

    def __init__(
        self,
        paths: Iterable[str],
        y: Iterable[int] | None = None,
        *,
        seq_len: int | None = None,
        ds_factor: int | None = None,
        target_leads: int = 12,
    ):
        # Store immutable arrays for fast __getitem__
        self.paths = np.asarray(list(paths))
        self.labels = None if y is None else np.asarray(list(y), dtype=np.int64)

        # Derive effective length after downsampling
        factor = int(ds_factor or DOWNSAMPLE_FACTOR)
        base_T = int(seq_len or SEQ_LEN)
        self.T = max(1, base_T // max(1, factor))
        self.factor = factor
        self.target_leads = int(target_leads)

    def __len__(self) -> int:
        """Number of records."""
        return int(self.paths.shape[0])

    def __getitem__(self, i: int):
        """Load i-th record, shape to (T, C), and (optionally) return label.

        Returns
        -------
        (xt, yt) : torch.Tensor, torch.LongTensor
            When labels are provided.
        xt : torch.Tensor
            When labels are None (predict mode).
        """
        path = str(self.paths[i])
        try:
            # Load as time-major float32 of shape (T, C)
            x = load_waveform_np(path, T=self.T, factor=self.factor)  # (T, C) float32
            if x.dtype != np.float32:
                x = x.astype("float32", copy=False)

            # Normalize lead count: pad/truncate to target_leads for model compatibility
            if x.ndim != 2:
                x = np.zeros((self.T, self.target_leads), dtype="float32")
            elif x.shape[1] < self.target_leads:
                pad = np.zeros((x.shape[0], self.target_leads - x.shape[1]), dtype="float32")
                x = np.concatenate([x, pad], axis=1)
            elif x.shape[1] > self.target_leads:
                x = x[:, : self.target_leads]

        except Exception:
            # Be robust to occasional read/parse failures; skip bad rows.
            x = np.zeros((self.T, self.target_leads), dtype="float32")

        xt = torch.from_numpy(x)  # (T, C), float32
        if self.labels is None:
            return xt
        yt = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return xt, yt


def make_label_encoder(y_series_train, y_series_test) -> LabelEncoder:
    """
    Fit a LabelEncoder on the *union* of train/test label strings.

    Why union-fit?
    --------------
    Prevents mismatched encodings when a class appears only in test (rare but possible
    with strict patient-wise splits). Ensures consistent class→index mapping.

    Parameters
    ----------
    y_series_train, y_series_test : array-like of str

    Returns
    -------
    sklearn.preprocessing.LabelEncoder

    Used by
    -------
    • src.Centralized (to encode labels before building datasets)
    """
    union = np.unique(
        np.concatenate(
            [np.asarray(y_series_train, dtype=str), np.asarray(y_series_test, dtype=str)]
        )
    )
    return LabelEncoder().fit(union)


def make_train_val_loaders(
    train_paths: np.ndarray,
    y_tr_enc: np.ndarray,
    *,
    device: torch.device,
    batch_size: int | None = None,
    val_frac: float = 0.10,
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """
    Stratify a provided *training* set into (train/val) and return DataLoaders.

    What it does
    ------------
    • Performs a single StratifiedShuffleSplit on integer-encoded labels
    • Builds ECGDataset instances for each split
    • Returns DataLoaders configured via utils.torch_loader_kwargs
      (correct num_workers/pin_memory/shuffle for the device type)
    • Also returns the index arrays (tr_pos, va_pos) for reproducibility and
      to construct a stable *train-eval* loader in Centralized.py.

    Parameters
    ----------
    train_paths : np.ndarray[str]
        WFDB record base paths for the training pool.
    y_tr_enc : np.ndarray[int]
        Integer-encoded labels aligned with train_paths.
    device : torch.device
        Used to derive DataLoader kwargs (workers/pinning).
    batch_size : Optional[int]
        Defaults to config.BATCH_SIZE (deep pipeline's BATCH_SIZE).
    val_frac : float
        Fraction of the training pool to use as validation.

    Returns
    -------
    (train_loader, val_loader, tr_pos, va_pos)

    Used by
    -------
    • src.Centralized to build train/val loaders for deep models
    """
    train_paths = np.asarray(train_paths)
    y_tr_enc = np.asarray(y_tr_enc, dtype=np.int64)
    assert train_paths.shape[0] == y_tr_enc.shape[0], "paths and labels must match in length"

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=SEED)
    all_pos = np.arange(len(y_tr_enc))
    tr_pos, va_pos = next(sss.split(all_pos, y_tr_enc))

    train_ds = ECGDataset(train_paths[tr_pos], y_tr_enc[tr_pos])
    val_ds   = ECGDataset(train_paths[va_pos],  y_tr_enc[va_pos])

    bs = int(batch_size or BATCH_SIZE)
    kw_tr = torch_loader_kwargs(True,  bs, device.type)   # shuffle True for training
    kw_va = torch_loader_kwargs(False, bs, device.type)   # shuffle False for validation

    return DataLoader(train_ds, **kw_tr), DataLoader(val_ds, **kw_va), tr_pos, va_pos


def make_predict_loader(
    paths: Iterable[str],
    *,
    device: torch.device,
    batch_size: int | None = None
) -> DataLoader:
    """
    Convenience builder for inference-only DataLoader (no labels).

    Parameters
    ----------
    paths : Iterable[str]
        WFDB record base paths (no extension).
    device : torch.device
        Used by torch_loader_kwargs to set workers/pin_memory.
    batch_size : Optional[int]
        Defaults to config.BATCH_SIZE (deep pipeline's default).

    Returns
    -------
    torch.utils.data.DataLoader

    Used by
    -------
    • src.results_visualization (TorchAdapter for model evaluation)
    """
    ds = ECGDataset(paths, y=None)
    bs = int(batch_size or BATCH_SIZE)
    kw = torch_loader_kwargs(False, bs, device.type)  # no shuffle for predict
    return DataLoader(ds, **kw)


__all__ = [
    "ECGDataset",
    "make_label_encoder",
    "make_train_val_loaders",
    "make_predict_loader",
]
