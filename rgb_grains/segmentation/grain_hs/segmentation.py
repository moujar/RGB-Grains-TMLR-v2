"""Watershed grain splitting and masked hyperspectral crop extraction."""

from __future__ import annotations

from typing import Any

import cv2 as cv
import numpy as np
from skimage.measure import label, regionprops
import time

from grain_hs._path import ensure_gala_path

ensure_gala_path()
from gala import morpho  # noqa: E402

from grain_hs.constants import RGB_BANDS

from numba import njit


def watershed_segmentation(
    binary_image: np.ndarray,
    area_range: tuple[int, int],
    solidity: float,
    *,
    show_debug: bool = False,
) -> tuple[list[Any], np.ndarray, list, np.ndarray]:
    """Split a binary foreground into grain regions via h-minima and watershed dams.

    Parameters
    ----------
    binary_image
        2D foreground grain_cleaned_background (values in ``{0,1}`` or comparable).
    area_range
        ``(min_area, max_area)`` in pixels for ``skimage.measure.regionprops``.
    solidity
        Minimum solidity for a region to count as a grain.
    show_debug
        If True, display the masked watershed image with matplotlib (notebook use).

    Returns
    -------
    grains
        List of ``RegionProperties`` for accepted grains.
    labeled_array
        Label image after watershed and masking.
    regions
        All regionprops from the label image (unfiltered).
    bw3
        Binary image after dams applied (watersheet boundaries zeroed).
    """
    d = -cv.distanceTransform(binary_image.astype(np.uint8), cv.DIST_L2, 3)
    t0 = time.time()
    print("        starting morpho.hminima")
    d2 = morpho.hminima(d, 3)
    print(f"        morpho.hminima() took {time.time() - t0:.2f} seconds")
    t0 = time.time()
    lab = morpho.watershed(d2, dams=True)
    print(f"        morpho.watershed() took {time.time() - t0:.2f} seconds")
    binary_image[lab == 0] = 0

    if show_debug:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 30))
        plt.imshow(binary_image)

    ## "label" is the grain number, detected by looking for simply connected components in the above-thresholds regions.
    labeled_array = label(binary_image, connectivity=1)
    regions = regionprops(labeled_array)
    grains = [
        r
        for r in regions
        if area_range[0] <= r.area <= area_range[1] and r.solidity > solidity
    ]
    return grains, labeled_array, regions, binary_image

def grain_segmentation(
    image: np.ndarray,
    labeled_array: np.ndarray,
    grain: Any,
    region_size: int = 252,
) -> np.ndarray:
    """Extract a single grain from the image using its label mask.

    This function:
    1. Crops a region_size x region_size square centered on the grain's centroid
    2. Masks out all pixels not belonging to this grain (sets background to 0)
    3. Bottom-aligns the grain in the output square (legacy behavior from notebooks)

    The output preserves the original pixel values from the input image.
    If input is raw int16 → output is raw int16.
    If input is normalized float → output is normalized float.

    Args:
        image: 3D array (H, W, n_bands) - the hyperspectral cube
        labeled_array: 2D integer label image where each grain has a unique ID
        grain: RegionProperties object for the grain to extract
        region_size: Size of output square (default 252x252 pixels)

    Returns:
        3D array (region_size, region_size, n_bands) with the extracted grain.
        Background pixels are set to 0. The grain is bottom-aligned in the square.
    """
    centroid_row, centroid_col = grain.centroid
    min_row = max(int(centroid_row - region_size / 2), 0)
    max_row = int(centroid_row + region_size / 2)
    min_col = max(int(centroid_col - region_size / 2), 0)
    max_col = int(centroid_col + region_size / 2)

    # Crop the region around the grain
    grain_cropped_region = image[min_row:max_row, min_col:max_col, :]
    height, width, depth = grain_cropped_region.shape

    # Create mask for this grain only (0 = background, non-zero = this grain)
    label_mask_local_to_one_grain = np.zeros((height, width))
    label_region_local_to_one_grain = labeled_array[min_row:max_row, min_col:max_col]
    centroid_label = labeled_array[int(centroid_row), int(centroid_col)]
    label_mask_local_to_one_grain[label_region_local_to_one_grain == centroid_label] \
        = label_region_local_to_one_grain[label_region_local_to_one_grain == centroid_label]

    # Create output array and bottom-align the grain
    grain_cleaned_background = np.zeros((region_size, region_size, depth), dtype=grain_cropped_region.dtype)
    start_row = max(region_size - height, 0)

    # Apply mask to all bands at once using broadcasting
    mask_bool = label_mask_local_to_one_grain != 0
    grain_cleaned_background[start_row:start_row + height, :width, :] = (
        grain_cropped_region * mask_bool[..., np.newaxis]
    )
    #     # Apply mask per band: keep only pixels belonging to this grain
    #     for d in range(depth):
    #         plane = np.zeros((height, width), dtype=grain_cropped_region.dtype)
    #         plane[label_mask_local_to_one_grain != 0] = grain_cropped_region[:, :, d]
    # [label_mask_local_to_one_grain != 0]
    #         grain_cleaned_background[start_row : start_row + height, :width, d] = plane
    return grain_cleaned_background
