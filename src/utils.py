from __future__ import annotations
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .config import RESULTS_DIR




def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_signal(sig: np.ndarray, title: str = "ECG (12xT)", save: str | None = None):
    # sig shape (12, T)
    plt.figure(figsize=(10, 6))
    offset = 2.5 * np.arange(sig.shape[0])
    for i in range(sig.shape[0]):
        plt.plot(sig[i] + offset[i])
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Leads (offset)")
    plt.tight_layout()
    if save:
        ensure_dir(RESULTS_DIR)
        plt.savefig(RESULTS_DIR / save, dpi=150)
    plt.close()




def compute_metrics(y_true, y_prob, labels):
    y_pred = y_prob.argmax(axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    try:
        auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "f1_macro": f1m, "roc_auc_ovr": auc}