"""
Hyperspectral grain extraction: reflectance normalization, watershed segmentation,
and 3-band NPZ export.

The vendored ``gala/morpho`` package (repository root) provides ``hminima`` and
watershed-with-dams used for splitting touching grains.
"""

from __future__ import annotations

from grain_hs._path import ensure_gala_path
from grain_hs.constants import set_parameters

ensure_gala_path()

__all__ = ["ensure_gala_path", "set_parameters"]
