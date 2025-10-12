# src/__init__.py
"""
src — Sustainable AI in Healthcare (DSP5100)

Lightweight package initializer:
  • Defines package version/metadata
  • Ensures results directory exists
  • Exposes common config knobs and convenience helpers without heavy imports
"""

from __future__ import annotations

# --- Package metadata ----------------------------------------------------
__title__   = "sustainable-ai-healthcare"
__version__ = "0.1.0"
__author__  = "DSP5100 Team"

# --- Minimal, safe imports (no heavy deps like torch/wfdb here) ----------
from pathlib import Path

try:
    from .config import RESULTS_DIR, PROJ_ROOT  # type: ignore
except Exception:
    # Fallbacks if config can’t be imported during early setups
    PROJ_ROOT  = Path(__file__).resolve().parent.parent
    RESULTS_DIR = PROJ_ROOT / "results"

# Ensure results dir exists at import-time (cheap and helpful)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Tiny convenience wrappers (lazy heavy imports inside functions) -----
def set_seed(seed: int | None = None) -> None:
    """Project-wide seeding (lazy-imports utils to avoid torch cost on import)."""
    from .utils import set_seed as _set_seed
    _set_seed(seed)

def pick_device():
    """Return best available torch device (lazy import)."""
    from .utils import pick_device as _pick
    return _pick()

def load_metadata():
    """Load PTB-XL metadata (lazy import to avoid pandas/wfdb at package import)."""
    from .data_loader import load_metadata as _load
    return _load()

def create_model(model_type: str, n_classes: int, **kwargs):
    """Factory for models (CNN/LSTM/RNN/ANN)."""
    from .models import create_model as _create
    return _create(model_type, n_classes, **kwargs)

# --- What we publicly expose from the package ----------------------------
__all__ = [
    "__title__", "__version__", "__author__",
    "PROJ_ROOT", "RESULTS_DIR",
    "set_seed", "pick_device", "load_metadata", "create_model",
]
