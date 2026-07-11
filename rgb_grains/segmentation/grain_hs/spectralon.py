"""Spectralon ROI statistics: brightest band and per-band mean reference.

SPECTRALON USAGE IN THIS PROJECT:
=================================
The spectralon is a white reference panel with known reflectance properties.
In an ideal calibrated image, all bands would have the same mean over the spectralon.
In practice, we use these values for:

1. WATERSHED SEGMENTATION (in reflectance.py):
   - Row-wise normalization to create reflectance image
   - Values normalized such that spectralon ≈ 1.0

2. RGB WHITE BALANCE (for visualization):
   - Divide each RGB channel by its mean_over_spectralon
   - This corrects for illumination differences across bands
   - See script_hdr_to_jpg.py for the reference implementation
"""

from __future__ import annotations

from typing import Any

import numpy as np

from grain_hs.constants import SPECTRALON_COL_START, SPECTRALON_ROW_COUNT


def spectralon_mean_for_band(img: Any, band_index: int) -> float:
    """Mean raw value over the fixed spectralon ROI for a single band index."""
    rows = list(range(SPECTRALON_ROW_COUNT))
    cols = list(range(SPECTRALON_COL_START, img.shape[1]))
    band = np.asarray(img.read_subimage(rows, cols, [band_index]))
    return float(np.mean(band))

def band_brightness(img, k):
    """Compute mean brightness of band k over the spectralon ROI (rows 0-99, cols 150+)."""
    rows = list(range(SPECTRALON_ROW_COUNT))
    cols = list(range(SPECTRALON_COL_START, img.shape[1]))
    band = np.array(img.read_subimage(rows, cols, [k]))
    mean = np.mean(band)
    print(f"Band {k} brightness: {mean:.1f}")
    return mean




def brightest_band(img: Any, restrict_search:bool=False) -> tuple[int, float]:
    """Find the band with highest mean brightness over the spectralon ROI.

    Parameters
    ----------
    img
        Spectral ``SpyFile`` (or compatible) with ``read_subimage`` and ``shape``.

    Returns
    -------
    band_index
        Index of the brightest band.
    max_mean
        Mean value for that band over the ROI.
    """
    rows = list(range(SPECTRALON_ROW_COUNT))
    cols = list(range(SPECTRALON_COL_START, img.shape[1]))

    if restrict_search:
        ## restrict search to the RGB bands: 
        from grain_hs.constants import RGB_BANDS
        band_indices = list(RGB_BANDS)
    else:
        n_bands = int(img.shape[2])
        band_indices = range(n_bands)

    means: list[float] = []
    for k in band_indices:
        band = np.asarray(img.read_subimage(rows, cols, [k]))
        means.append(float(np.mean(band)))
    i = int(np.argmax(means))
    mean = means[i]
    if restrict_search: ## going back to absolute space bands
        i = list(RGB_BANDS)[i]
    return i, mean
