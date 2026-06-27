"""Single-band reflectance normalization using the spectralon strip."""

from __future__ import annotations

import cv2 as cv
import numpy as np


def reflectance(
    image: np.ndarray,
    crop_idx_dim1: int,
    thresh_lum_spectralon: float,
) -> np.ndarray:
    """Normalize one band to reflectance using the spectralon region left of the scene.

    The spectralon (columns 0..crop_idx_dim1-1) is used as white reference.
    Each row is divided by its mean spectralon value to get reflectance ratio.
    Output is float32 with spectralon ≈ 1.0.

    Parameters
    ----------
    image
        2D array, single spectral band (spatial dimensions as in the notebook pipeline).
    crop_idx_dim1
        Column index separating spectralon (left) from grain scene (right).
    thresh_lum_spectralon
        Threshold on raw intensity to seed the spectralon mask.

    Returns
    -------
    np.ndarray
        ``float32`` array of same shape as ``image``, row-wise divided by reference.
        Values are normalized such that spectralon ≈ 1.0 (100% reflectance).
        Typical output range: 0.0 to ~2.0 (grains may be brighter than spectralon).

    PIXEL VALUE TRANSFORMATION:
    ==========================
    Input:  raw sensor values (e.g., 0-65535 for int16, but typically ~10000-30000)
    Process: Each row x is divided by ref[x] (mean of bright spectralon pixels in that row)
    Output: reflectance ratio (float32) where 1.0 = spectralon reference

    EXAMPLE:
    - Input pixel: 15000, spectralon reference for that row: 20000
    - Output: 15000 / 20000 = 0.75 (75% reflectance)
    """
    imr = np.empty(image.shape, np.float32)
    im0 = image[:, :crop_idx_dim1]
    _, binary_image0 = cv.threshold(im0, thresh_lum_spectralon, 1, cv.THRESH_BINARY)
    binary_image0 = cv.erode(binary_image0, np.ones((10, 10), np.uint8))
    binary_image0 = cv.morphologyEx(
        binary_image0, cv.MORPH_CLOSE, np.ones((20, 20), np.uint8)
    )

    # Compute per-row reference from spectralon region
    # If <50 bright pixels, use default value 22000
    ref = np.zeros((image.shape[0],), dtype=image.dtype)
    for x in range(image.shape[0]):
        nz = binary_image0[x, :] != 0
        if np.sum(nz) > 50:
            ref[x] = np.mean(im0[x, nz])
        else:
            ref[x] = 22000
        imr[x, :] = image[x, :] / ref[x]

    return imr
