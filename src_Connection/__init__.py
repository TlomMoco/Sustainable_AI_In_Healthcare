# src_Connection/__init__.py
"""
src_Connection — Federated/connection-focused variant of the DSP5100 project.

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
    - src_Connection.utils               (seeding, device, logging helpers)
    - src_Connection.data_loader         (metadata and feature-table builders)
    - src_Connection.models              (model factories: CNN/RNN/LSTM/ANN)
    - src_Connection.config              (project-wide paths and constants)

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
__title__ = "sustainable-ai-healthcare-connection"
__author__ = "DSP5100 Team"

# Prefer dynamic version if installed as a package; fall back otherwise.
# This avoids hardcoding versions and lets pip-installed wheels expose pkg version.
try:
    from importlib.metadata import version as _pkg_version  # Py3.8+
    __version__ = _pkg_version(__name__)
except Exception:
    # Fallback used in editable/dev environments where package metadata isn't present.
    __version__ = "0.1.0"

# --- Paths / results dir --------------------------------------------------
# We try to pull canonical paths from config (the single source of truth),
# but *without* importing heavy dependencies. If config is unavailable or raises,
# we fall back to safe defaults rooted at the package parent.
try:
    # Prefer the package's own config (doesn't pull torch/wfdb at import)
    from .config import RESULTS_DIR, PROJ_ROOT  # type: ignore
except Exception:
    # Safe fallbacks if config isn't ready yet (e.g., partial env or first-run).
    PROJ_ROOT = Path(__file__).resolve().parent.parent
    RESULTS_DIR = PROJ_ROOT / "results"

# On import, eagerly create results directories for convenience (opt-out supported).
# This makes plotting/saving artifacts "just work" across notebooks and scripts.
if os.environ.get("SAIH_AUTO_MKRESULTS", "1") == "1":
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "viz").mkdir(parents=True, exist_ok=True)
    except Exception:
        # Non-fatal (e.g., read-only env, container without perms); callers can create later.
        pass


def get_paths() -> Tuple[Path, Path]:
    """Return common project paths (PROJ_ROOT, RESULTS_DIR).

    Returns
    -------
    (PROJ_ROOT, RESULTS_DIR) : tuple[pathlib.Path, pathlib.Path]

    Used by
    -------
    • Any module or script that needs consistent base paths (e.g., saving artifacts)
      without importing config directly.

    Notes
    -----
    Keeping this here avoids importing .config in callers and stays lightweight.
    """
    return PROJ_ROOT, RESULTS_DIR


# --- Lazy convenience wrappers (avoid heavy imports at import time) ------
def set_seed(seed: int | None = None) -> None:
    """Project-wide seeding (LAZY import).

    Delegates to:
        src_Connection.utils.set_seed

    Why lazy?
        Avoids importing torch/numpy/loggers at package-import time. Callers pay
        the cost *only* when they actually need seeding.

    Typical callers
    ---------------
    • Centralized.py (centralized training)
    • Client.py / Server.py (federated clients/orchestrator)
    """
    from .utils import set_seed as _set_seed
    _set_seed(seed)


def pick_device():
    """Return best available torch device (LAZY import).

    Delegates to:
        src_Connection.utils.pick_device

    Returns
    -------
    torch.device or a lightweight proxy indicating 'cpu', 'cuda', or 'mps'.

    Typical callers
    ---------------
    • Centralized.py training loop setup
    • Client.py when preparing model/device per round
    """
    from .utils import pick_device as _pick
    return _pick()


def load_metadata():
    """Load PTB-XL metadata tables (LAZY import).

    Delegates to:
        src_Connection.data_loader.load_metadata

    Returns
    -------
    Dict/tuple of pandas.DataFrame (depends on your implementation), typically:
        • ptbxl_database.csv
        • scp_statements.csv
      loaded and preprocessed for downstream usage.

    Typical callers
    ---------------
    • EDA and preprocessing steps
    • Data module setup in training and evaluation flows
    """
    from .data_loader import load_metadata as _load
    return _load()


def create_model(model_type: str, n_classes: int, **kwargs):
    """Factory for deep models (CNN/LSTM/RNN/ANN) (LAZY import).

    Delegates to:
        src_Connection.models.create_model

    Parameters
    ----------
    model_type : {"cnn", "rnn", "lstm", "ann", ...}
        A key understood by your models factory.
    n_classes : int
        Number of output classes for classification (binary or multi-class).
    **kwargs :
        Additional model hyperparameters (e.g., channels, hidden sizes, dropout).

    Returns
    -------
    torch.nn.Module
        An initialized model ready to be moved to a device.

    Typical callers
    ---------------
    • Centralized.py during centralized training runs
    • Client.py (per-client model instantiation) and/or Server.py (global model)
    """
    from .models import create_model as _create
    return _create(model_type, n_classes, **kwargs)


# Optional: make type checkers happy without importing at runtime.
# This block is ignored at runtime but helps IDEs and static analyzers resolve symbols.
if TYPE_CHECKING:
    from . import (  # noqa: F401
        config,
        models,
        data_loader,
        data_preprocessing,
        results_visualization,
        utils,
        Client,
        Server,
        Centralized,
    )

# Public API surface; keeps imports stable for downstream code:
#   from src_Connection import create_model, set_seed, pick_device, get_paths, ...
__all__ = [
    "__title__", "__version__", "__author__",
    "PROJ_ROOT", "RESULTS_DIR", "get_paths",
    "set_seed", "pick_device", "load_metadata", "create_model",
]