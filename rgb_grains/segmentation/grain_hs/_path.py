"""Project root on ``sys.path`` so the vendored ``gala/`` package is importable."""

from __future__ import annotations

import sys
from pathlib import Path

_GALA_REGISTERED = False


def ensure_gala_path() -> None:
    """Prepend the repository root to ``sys.path`` (once).

    PyPI distributes a different package named ``gala`` (astronomy). This project
    vendors morphological helpers under ``./gala/`` (see ``gala/README.txt``).
    """
    global _GALA_REGISTERED
    if _GALA_REGISTERED:
        return
    root = Path(__file__).resolve().parent.parent
    r = str(root)
    if r not in sys.path:
        sys.path.insert(0, r)
    _GALA_REGISTERED = True
