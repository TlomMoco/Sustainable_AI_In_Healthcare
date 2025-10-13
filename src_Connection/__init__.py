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
try:
    from importlib.metadata import version as _pkg_version  # Py3.8+
    __version__ = _pkg_version(__name__)
except Exception:
    __version__ = "0.1.0"

# --- Paths / results dir --------------------------------------------------
try:
    from .config import RESULTS_DIR, PROJ_ROOT  # type: ignore
except Exception:
    PROJ_ROOT = Path(__file__).resolve().parent.parent
    RESULTS_DIR = PROJ_ROOT / "results"

# Create results directories unless opted out
if os.environ.get("SAIH_AUTO_MKRESULTS", "1") == "1":
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "viz").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


def get_paths() -> Tuple[Path, Path]:
    """Return common project paths (PROJ_ROOT, RESULTS_DIR)."""
    return PROJ_ROOT, RESULTS_DIR


# --- Lazy convenience wrappers (avoid heavy imports at import time) ------
def set_seed(seed: int | None = None) -> None:
    from .utils import set_seed as _set_seed
    _set_seed(seed)


def pick_device():
    from .utils import pick_device as _pick
    return _pick()


def load_metadata():
    from .data_loader import load_metadata as _load
    return _load()


def create_model(model_type: str, n_classes: int, **kwargs):
    from .models import create_model as _create
    return _create(model_type, n_classes, **kwargs)


if TYPE_CHECKING:  # pragma: no cover
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

__all__ = [
    "__title__", "__version__", "__author__",
    "PROJ_ROOT", "RESULTS_DIR", "get_paths",
    "set_seed", "pick_device", "load_metadata", "create_model",
]