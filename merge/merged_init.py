"""
src — Federated/connection-focused variant of the DSP5100 project.

What this package initializer is responsible for
-----------------------------------------------
• Lightweight import-time setup (safe to import even if heavy deps like torch/wfdb
  aren't installed yet).
• Exposes a minimal, stable public API (see __all__) that other modules/scripts
  can import from without paying heavy import costs.
• Resolves common project paths and (optionally) ensures results directories exist.
• Provides *lazy* convenience wrappers (set_seed, pick_device, load_metadata,
  create_model) so callers don't import heavy submodules at import time.

Where it connects
-----------------
• Centralized.py uses:
    - set_seed(), pick_device(), create_model()
• Client.py / Server.py (federated learning) often rely on:
    - create_model(), set_seed()
• EDA/training utilities may use:
    - get_paths() for filesystem locations, load_metadata() for PTB-XL tables
• The underlying implementations live in:
    - src.utils               (seeding, device, logging helpers)
    - src.data_loader         (metadata and feature-table builders)
    - src.models              (model factories: CNN/RNN/LSTM/ANN)
    - src.config              (project-wide paths and constants)

Import-time behavior & environment toggle
-----------------------------------------
By default, this module *creates* the results/ and results/viz/ directories
when imported. To disable that behavior (e.g., in read-only or packaging contexts),
set env var:  SAIH_AUTO_MKRESULTS=0

Public API (see __all__)
------------------------
__title__, __version__, __author__,
PROJ_ROOT, RESULTS_DIR, get_paths,
set_seed, pick_device, load_metadata, create_model
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple
import os

# --- Metadata -------------------------------------------------------------
__title__ = "sustainable-ai-healthcare"
__author__ = "DSP5100 Team"

# Prefer dynamic version if installed as a package; fall back otherwise.
try:
    from importlib.metadata import version as _pkg_version  # Py3.8+
    __version__ = _pkg_version(__name__)
except Exception:
    __version__ = "0.1.0"

# --- Paths / results dir --------------------------------------------------
try:
    from src.config import RESULTS_DIR, PROJ_ROOT, TUNING_DIR  # type: ignore
except Exception:
    PROJ_ROOT = Path(__file__).resolve().parent.parent
    RESULTS_DIR = PROJ_ROOT / "results"
    TUNING_DIR = RESULTS_DIR / "tuning"

# Create results directories unless opted out
if os.environ.get("SAIH_AUTO_MKRESULTS", "1") == "1":
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "viz").mkdir(parents=True, exist_ok=True)
        TUNING_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def get_paths() -> Tuple[Path, Path]:
    """Return common project paths (PROJ_ROOT, RESULTS_DIR)."""
    return PROJ_ROOT, RESULTS_DIR


# --- Lazy convenience wrappers (avoid heavy imports at import time) ------
def set_seed(seed: int | None = None) -> None:
    """Set global random seed across Python/NumPy/PyTorch.
    
    Parameters
    ----------
    seed : int, optional
        Random seed value. If None, uses SEED from config.
    """
    from .utils import set_seed as _set_seed
    _set_seed(seed)


def pick_device():
    """Select best available device (MPS -> CUDA -> CPU).
    
    Returns
    -------
    torch.device
    """
    from .utils import pick_device as _pick
    return _pick()


def load_metadata():
    """Load PTB-XL metadata and SCP aggregation table.
    
    Returns
    -------
    PTBXL
        Dataclass containing df (records) and agg (SCP mapping).
    """
    from .data_loader import load_metadata as _load
    return _load()


def create_model(model_type: str, n_classes: int, **kwargs):
    """Create a model instance (CNN/LSTM/RNN/ANN).
    
    Parameters
    ----------
    model_type : str
        Model architecture: 'cnn', 'lstm', 'rnn', or 'ann'.
    n_classes : int
        Number of output classes.
    **kwargs
        Additional model-specific parameters (e.g., hidden, layers, bidir).
        
    Returns
    -------
    torch.nn.Module
    """
    from .models import create_model as _create
    return _create(model_type, n_classes, **kwargs)


def log(msg: str) -> None:
    """Print a timestamped log message.
    
    Parameters
    ----------
    msg : str
        Message to log.
    """
    from .utils import log as _log
    _log(msg)


def ensure_dir(path: Path) -> None:
    """Create directory and parents if they don't exist.
    
    Parameters
    ----------
    path : Path
        Directory path to create.
    """
    from .utils import ensure_dir as _ensure
    _ensure(path)


# --- Type checking imports (zero runtime cost) ---
if TYPE_CHECKING:  # pragma: no cover
    from src import (  # noqa: F401
        config,
        models,
        data_loader,
        data_preprocessing,
        results_visualization,
        utils,
        Client,
        Server,
        Centralized,
        Experiments,
        tuning,
        eda,
    )

__all__ = [
    # Metadata
    "__title__",
    "__version__",
    "__author__",
    # Paths
    "PROJ_ROOT",
    "RESULTS_DIR",
    "TUNING_DIR",
    "get_paths",
    # Lazy utilities
    "set_seed",
    "pick_device",
    "load_metadata",
    "create_model",
    "log",
    "ensure_dir",
]
