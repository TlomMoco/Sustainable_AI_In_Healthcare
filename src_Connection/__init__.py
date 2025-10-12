# src_Connection/__init__.py
"""
src_Connection — Federated/connection-focused variant of the DSP5100 project.

Lightweight initializer:
  • Package metadata
  • Resolves project paths safely
  • Ensures results directory exists
  • Exposes a few convenience helpers via lazy imports (no heavy deps on import)
"""

from __future__ import annotations
from pathlib import Path

# --- Metadata -------------------------------------------------------------
__title__   = "sustainable-ai-healthcare-connection"
__version__ = "0.1.0"
__author__  = "DSP5100 Team"

# --- Paths / results dir --------------------------------------------------
try:
    # Prefer the package's own config (doesn't pull torch/wfdb at import)
    from .config import RESULTS_DIR, PROJ_ROOT  # type: ignore
except Exception:
    # Safe fallbacks if config isn't ready yet
    PROJ_ROOT  = Path(__file__).resolve().parent.parent
    RESULTS_DIR = PROJ_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Lazy convenience wrappers (avoid heavy imports on import) ------------
def set_seed(seed: int | None = None) -> None:
    """Project-wide seeding (lazy import)."""
    from .utils import set_seed as _set_seed
    _set_seed(seed)

def pick_device():
    """Return best available torch device (lazy import)."""
    from .utils import pick_device as _pick
    return _pick()

def load_metadata():
    """Load PTB-XL metadata (lazy import)."""
    from .data_loader import load_metadata as _load
    return _load()

def create_model(model_type: str, n_classes: int, **kwargs):
    """Factory for models (CNN/LSTM/RNN/ANN)."""
    from .models import create_model as _create
    return _create(model_type, n_classes, **kwargs)

__all__ = [
    "__title__", "__version__", "__author__",
    "PROJ_ROOT", "RESULTS_DIR",
    "set_seed", "pick_device", "load_metadata", "create_model",
]
