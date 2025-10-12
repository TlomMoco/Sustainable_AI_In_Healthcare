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
    def __init__(self, paths, y=None, *, seq_len=None, ds_factor=None):
        self.paths = list(paths)
        self.labels = None if y is None else np.asarray(y, dtype=np.int64)
        self.T = max(1, (seq_len or CFG.SEQ_LEN) // max(1, (ds_factor or CFG.DOWNSAMPLE_FACTOR)))
        self.factor = int(ds_factor or CFG.DOWNSAMPLE_FACTOR)
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        x = load_waveform_np(self.paths[i], T=self.T, factor=self.factor)
        if self.labels is None: return torch.from_numpy(x)
        return torch.from_numpy(x), int(self.labels[i])

def make_label_encoder(y_series_train, y_series_test):
    union = np.unique(np.concatenate([np.asarray(y_series_train).astype(str), np.asarray(y_series_test).astype(str)]))
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
