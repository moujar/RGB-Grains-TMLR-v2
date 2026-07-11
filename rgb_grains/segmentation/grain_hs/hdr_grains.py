"""Load an HDR cube_image, run reflectance + watershed, return image_cropped_spectralon, labelled_cropped_spectralon and grain regions.

Raw int16 values are preserved for output; spectralon means are returned separately
for optional normalization (e.g., divide by means_over_spectralon for realistic colors).

PIXEL VALUE TRANSFORMATIONS in this module:
===========================================
1. Raw cube_image loading (spy_image.open_image): data remains in raw sensor values (typically int16)
2. RGB band extraction: cube_image loaded as int16, transposed to (H, W, 3)
3. Spectralon mean computation (spectralon_mean_for_band):
   - Computes mean over ROI: rows 0-99, cols 150-end
   - Returns raw average intensity for each band
4. Reflectance conversion (reflectance function):
   - INPUT: raw single-band image (float64)
   - PROCESSING: Each row divided by its spectralon reference value
   - OUTPUT: float32 reflectance image (values ~0-1, where spectralon ≈ 1.0)
   - Used ONLY for watershed segmentation (not for RGB output)
5. Final RGB image_cropped_spectralon: raw int16 values (NOT reflectance-calibrated)
   - The image_cropped_spectralon preserves original sensor values for ML training
   - means_over_spectralon returned separately for optional normalization later
"""

from __future__ import annotations

from pathlib import Path
import time

import cv2 as cv
import numpy as np
import spectral as sp

from grain_hs.reflectance import reflectance
from grain_hs.segmentation import watershed_segmentation
from grain_hs.spectralon import brightest_band, band_brightness



def segment_grains_from_hdr(
    hdr_path: Path,
    *,
    crop_idx_dim1: int,
    reflectance_crop_trim: int,
    watershed_crop_trim: int,
    area_range: tuple[int, int],
    solidity: float,
    binary_thresh: float,
    debug: bool = False,
) -> tuple[str, list, np.ndarray, np.ndarray, np.ndarray]:
    """Segment grains from one hyperspectral cube, reducing to 3 RGB bands.

    Returns:
        img_name: HDR stem (no extension)
        grains: List of RegionProperties for detected grains
        image_cropped_spectralon: Cropped cube (H, W, 3) as int16
        labelled_cropped_spectralon: 2D label image aligned with image_cropped_spectralon
        means_over_spectralon: Array of 3 mean values over spectralon ROI
    """
    stem = hdr_path.stem
    img_name = stem
    spy_image = sp.open_image(str(hdr_path))

    from grain_hs.constants import RGB_BANDS


    # Load only the 3 RGB bands to save memory (instead of all bands).
    # Data remains as raw int16 sensor values (not normalized).
    cube_image = np.zeros((spy_image.shape[1], spy_image.shape[0], len(RGB_BANDS)), dtype=np.int16)
    for i, b in enumerate(RGB_BANDS):
        cube_image[:, :, i] = np.asarray(spy_image.read_band(b), dtype=np.int16).T

    # Find brightest band over spectralon ROI for watershed segmentation
    band, max_ref = brightest_band(spy_image, restrict_search=True)
    print(f"    Using band {band} for segmentation (max_ref={max_ref:.2f})")

    # Compute mean brightness per RGB band over spectralon ROI (for white balance)
    means_over_spectralon = np.array([band_brightness(spy_image, b) for b in RGB_BANDS])
    print(f"    Spectralon means [R,G,B]: {[round(m, 1) for m in means_over_spectralon]}")


    ## Wathershed: preparation of the input values    
    # Threshold to identify spectralon pixels
    thresh_lum_spectralon = max_ref * 0.7  
    # Load single band for watershed segmentation (as float64 for precision)
    img_1b = np.asarray(spy_image.read_band(band), dtype=np.float64).T
    # Convert to reflectance: each row divided by its spectralon reference. 
    # This produces normalized values where spectralon ≈ 1.0
    img_reflectance = reflectance(img_1b, crop_idx_dim1 - reflectance_crop_trim, thresh_lum_spectralon)
    col0 = crop_idx_dim1 - watershed_crop_trim
    im1 = img_reflectance[:, col0:]
    ## we threshold the image (along the slected band) to obtain a binarized image of present/absent pixels
    _, binary_img = cv.threshold(im1, binary_thresh, 1, cv.THRESH_BINARY)  ## binary_thresh: default is 0.15
    # area_range: number of pixels (min, max) that should be grian-pixels. 
    # default: between 3000 and 20000 pixels.
    ## solidity: default to 0.75

    # Debug mode: reduce image size for faster processing
    if debug:
        binary_img = binary_img[:binary_img.shape[0]//2, :binary_img.shape[1]//2]
        print(f"    Debug mode: binary image shape {binary_img.shape}")

    # Run watershed segmentation
    t0 = time.time()
    print("    Starting watershed segmentation...")
    
    grains, labelled_cropped_spectralon, _, _ = watershed_segmentation(binary_img, area_range, solidity)
    print(f"    Watershed took {time.time()-t0:.2f}s, found {len(grains)} grains")

    # Extract scene region (raw int16 values, not normalized)
    image_cropped_spectralon = cube_image[:, col0:, :]
    # y_label = "u"  # Variety label (simplified)

    return img_name, grains, image_cropped_spectralon, labelled_cropped_spectralon, means_over_spectralon

    ## legacy code, abandoned, but stored here for reference
    # Determine which of the 3 RGB bands is brightest over the spectralon ROI
    # for use in reflectance calibration/segmentation.
    # if segmentation_band is not None:
    #     band = segmentation_band
    #     max_ref = spectralon_mean_for_band(spy_image, band)
    #     print(f"    1using band {band} (max_ref={max_ref:.2f})")
    # elif brightest_cache is not None and img_name in brightest_cache:
    #     band, max_ref = brightest_cache[img_name]
    #     print(f"    2using band {band} (max_ref={max_ref:.2f})")
    # else:


# def parse_variety_from_name(name: str) -> str | None:
#     """Return digits after ``var`` in the filename, if present (case-insensitive)."""
#     m = re.search(r"var(\d+)", name, re.I)
#     return m.group(1) if m else None
# def load_brightest_csv(path: Path) -> dict[str, tuple[int, float]]:
#     """Parse ``brightest_bands.csv`` (index column + band, max_ref)."""
#     out: dict[str, tuple[int, float]] = {}
#     with open(path, newline="") as f:
#         rows = list(csv.reader(f))
#     if len(rows) < 2:
#         return out
#     for parts in rows[1:]:
#         if len(parts) < 3:
#             continue
#         stem_raw = parts[0].strip()
#         stem = Path(stem_raw).stem if stem_raw.lower().endswith(".hdr") else stem_raw
#         out[stem] = (int(float(parts[1])), float(parts[2]))
#     return out