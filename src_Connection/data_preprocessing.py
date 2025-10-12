# ---------
# ## 11) Data Pipeline (Dataset/DataLoader) — lightweight adapters (from notebook)
# ---------
from __future__ import annotations
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from . import config as CFG
from .utils import torch_loader_kwargs
from .data_loader import load_waveform_np

class ECGDataset(Dataset):
    """
    Returns each sample as a float32 tensor shaped (T, C) where:
      - T = effective sequence length after downsampling
      - C = number of ECG leads (usually 12)

    CNNs using Conv1d usually want (B, C, T) -> let the model permute.
    RNN/LSTM usually want (B, T, C) -> already aligned.
    """
    def __init__(self, paths, y=None, *, seq_len=None, ds_factor=None):
        self.paths = list(paths)
        self.labels = None if y is None else np.asarray(y, dtype=np.int64)
        self.T = max(1, (seq_len or CFG.SEQ_LEN) // max(1, (ds_factor or CFG.DOWNSAMPLE_FACTOR)))
        self.factor = int(ds_factor or CFG.DOWNSAMPLE_FACTOR)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            x = load_waveform_np(self.paths[i], T=self.T, factor=self.factor)  # (T, C) float32 np.ndarray
        except Exception:
            # Robustness: if a record is unreadable, return zeros of correct shape
            x = np.zeros((self.T, 12), dtype="float32")
        xt = torch.from_numpy(x).to(torch.float32)  # ensure float32 for torch
        if self.labels is None:
            return xt
        return xt, int(self.labels[i])

def make_label_encoder(y_series_train, y_series_test):
    union = np.unique(np.concatenate([
        np.asarray(y_series_train).astype(str),
        np.asarray(y_series_test).astype(str),
    ]))
    le = LabelEncoder().fit(union)
    return le

def make_train_val_loaders(train_paths, y_tr_enc, *, device, batch_size=None, val_frac=0.10):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=CFG.SEED)
    all_pos = np.arange(len(y_tr_enc))
    tr_pos, va_pos = next(sss.split(all_pos, y_tr_enc))

    train_ds = ECGDataset(train_paths[tr_pos], y_tr_enc[tr_pos])
    val_ds   = ECGDataset(train_paths[va_pos],  y_tr_enc[va_pos])

    kw_tr = torch_loader_kwargs(True,  batch_size or CFG.BATCH_SIZE, device.type)
    kw_va = torch_loader_kwargs(False, batch_size or CFG.BATCH_SIZE, device.type)
    return DataLoader(train_ds, **kw_tr), DataLoader(val_ds, **kw_va), tr_pos, va_pos
