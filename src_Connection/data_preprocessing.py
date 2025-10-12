# src_Connection/data_preprocessing.py
"""
Lightweight dataset / dataloader adapters for PTB-XL deep models.

Exposes:
  • ECGDataset — returns (T, C) float32 tensors (models can permute as needed)
  • make_label_encoder — union-fit across train/test series
  • make_train_val_loaders — stratified split + DataLoaders
  • make_predict_loader — DataLoader for inference-only paths
"""

from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

from . import config as CFG
from .utils import torch_loader_kwargs
from .data_loader import load_waveform_np


class ECGDataset(Dataset):
    """
    Returns each sample as a float32 tensor shaped (T, C) where:
      • T = effective sequence length after downsampling
      • C = number of ECG leads (usually 12)

    CNNs using Conv1d usually want (B, C, T) — let the model permute.
    RNN/LSTM usually want (B, T, C) — already aligned here.
    """

    def __init__(self, paths: Iterable[str], y: Iterable[int] | None = None, *, seq_len: int | None = None, ds_factor: int | None = None):
        self.paths = np.asarray(list(paths))
        self.labels = None if y is None else np.asarray(list(y), dtype=np.int64)

        factor = int(ds_factor or CFG.DOWNSAMPLE_FACTOR)
        base_T = int(seq_len or CFG.SEQ_LEN)
        self.T = max(1, base_T // max(1, factor))
        self.factor = factor

    def __len__(self) -> int:
        return int(self.paths.shape[0])

    def __getitem__(self, i: int):
        path = str(self.paths[i])
        try:
            x = load_waveform_np(path, T=self.T, factor=self.factor)  # (T, C) float32 np.ndarray
            if x.dtype != np.float32:
                x = x.astype("float32", copy=False)
        except Exception:
            # Robustness: if a record is unreadable, return zeros of correct shape
            x = np.zeros((self.T, 12), dtype="float32")
        xt = torch.from_numpy(x)  # (T, C), float32
        if self.labels is None:
            return xt
        return xt, int(self.labels[i])


def make_label_encoder(y_series_train, y_series_test) -> LabelEncoder:
    """
    Fit a LabelEncoder on the union of train/test labels (strings).
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
    Stratified 90/10-ish split (configurable) on the provided *training* set
    and return DataLoaders + the indices for reproducibility.
    """
    train_paths = np.asarray(train_paths)
    y_tr_enc = np.asarray(y_tr_enc, dtype=np.int64)
    assert train_paths.shape[0] == y_tr_enc.shape[0], "paths and labels must match in length"

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=CFG.SEED)
    all_pos = np.arange(len(y_tr_enc))
    tr_pos, va_pos = next(sss.split(all_pos, y_tr_enc))

    train_ds = ECGDataset(train_paths[tr_pos], y_tr_enc[tr_pos])
    val_ds   = ECGDataset(train_paths[va_pos],  y_tr_enc[va_pos])

    bs = int(batch_size or CFG.BATCH_SIZE)
    kw_tr = torch_loader_kwargs(True,  bs, device.type)
    kw_va = torch_loader_kwargs(False, bs, device.type)

    return DataLoader(train_ds, **kw_tr), DataLoader(val_ds, **kw_va), tr_pos, va_pos


def make_predict_loader(paths: Iterable[str], *, device: torch.device, batch_size: int | None = None) -> DataLoader:
    """
    Convenience: DataLoader for inference-only paths (no labels).
    """
    ds = ECGDataset(paths, y=None)
    bs = int(batch_size or CFG.BATCH_SIZE)
    kw = torch_loader_kwargs(False, bs, device.type)
    return DataLoader(ds, **kw)


__all__ = [
    "ECGDataset",
    "make_label_encoder",
    "make_train_val_loaders",
    "make_predict_loader",
]
