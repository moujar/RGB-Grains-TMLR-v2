#!/usr/bin/env python3
"""
REFERENCE IMPLEMENTATION: HDR → RGB with proper spectralon normalization.

This script produces REALISTIC COLORS by normalizing each RGB channel
by its mean brightness over the spectralon region (white reference).

PIXEL VALUE TRANSFORMATION (correct approach):
=============================================
1. Load raw HDR cube (int16 sensor values)
2. For each RGB band:
   - Read raw band values
   - Divide by mean_over_spectralon for that band
   - Result: normalized reflectance (spectalon ≈ 1.0)
3. Stack B, G, R channels (note: OpenCV-style BGR order used)
4. Output values are in range ~0-2, with realistic white balance

SPECTRALON NORMALIZATION FORMULA:
   normalized_band = raw_band / mean(spectralon_ROI)

This is the REFERENCE for how colors SHOULD look. Compare with:
- rgb_pipeline.py: Currently saves RAW values (no normalization)
- full_band_pipeline.py: Does apply normalization but may have broadcasting issues
- script_npz_to_jpg-2.py: Applies normalization when visualizing NPZ files
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import spectral as sp

from grain_hs.constants import SPECTRALON_COL_START, SPECTRALON_ROW_COUNT
from grain_hs import set_parameters

def band_brightness(img, k):
    """
    Compute mean brightness of band k over the spectralon ROI.

    This is the KEY FUNCTION for realistic color conversion.
    The spectralon (white reference panel) appears at top-left of image.
    Its mean brightness is used to normalize the entire band.

    ROI: rows 0-99, cols 150-end (matches constants.SPECTRALON_ROW_COUNT=100,
                                constants.SPECTRALON_COL_START=150)

    :param img: the spectral image (SpyFile from spectral library)
    :param k: the band index
    :return: the mean brightness of band k over spectralon (float)
    """
    rows = [i for i in range(SPECTRALON_ROW_COUNT)]  # SPECTRALON_ROW_COUNT
    cols = [i for i in range(SPECTRALON_COL_START, img.shape[1])]  # SPECTRALON_COL_START
    bands = [k]
    band = np.array(img.read_subimage(rows, cols, bands))
    mean = np.mean(band)  # This is the white reference value for this band

    print(f"Band {k} brightness: {mean}")
    return mean


def convert_to_rgb(data_dir, img_name):
    """
    Convert HDR hyperspectral image to RGB with proper white balance.

    KEY STEP: Each band divided by its spectralon mean for normalization.
    This corrects for illumination differences across bands.

    Note: Output is BGR order (Blue, Green, Red) for OpenCV compatibility.
    """
    img = sp.open_image(data_dir / img_name)
    RGB_BANDS: tuple[int, int, int] = (15, 52, 80) ## les valeurs du constructeur
    bands = list(RGB_BANDS)
    # CRITICAL: Divide each band by its spectralon mean for realistic colors
    # Without this, images have wrong white balance (typically too blue/green)
    img_r = img[:, :, bands[0]] / band_brightness(img, bands[0])  # Red channel
    img_g = img[:, :, bands[1]] / band_brightness(img, bands[1])  # Green channel
    img_b = img[:, :, bands[2]] / band_brightness(img, bands[2])  # Blue channel
    img_rgb = np.dstack((img_b, img_g, img_r))  # BGR order for OpenCV
    ## rotate the two first axis:
    img_rgb = np.rot90(img_rgb, k=1, axes=(0, 1))

    return img_rgb

def plot_hdr_to_jpg(data_dir, img_name, crop_idx_dim1, reflectance_trim, watershed_trim, show=False):
    img = convert_to_rgb(data_dir, img_name)

    ## image with rectangles
    plt.figure()
    plt.imshow(img)
    # Draw rectangles showing the processing regions
    # Reflectance region: columns 0 to (crop_idx_dim1 - reflectance_trim)
    # Watershed region: columns (crop_idx_dim1 - watershed_trim) to end
    col_reflectance_end = crop_idx_dim1 - reflectance_trim
    col_watershed_start = crop_idx_dim1 - watershed_trim
    height = img.shape[0]

    ## Solid rectangle yellow region delimited by SPECTRALON_COL_START, SPECTRALON_ROW_COUNT:
    spectralon_col_start = SPECTRALON_COL_START
    spectralon_row_count = SPECTRALON_ROW_COUNT
    plt.plot([spectralon_col_start, spectralon_col_start + spectralon_row_count, spectralon_col_start + spectralon_row_count, spectralon_col_start, spectralon_col_start], 
            [0, 0, height, height, 0], 'y-', linewidth=2, alpha=0.5, label='Spectralon region')

    # Solid rectangle for reflectance selected region (left part)
    plt.plot([0, col_reflectance_end, col_reflectance_end, 0, 0],
            [0, 0, height, height, 0], 'r-', linewidth=2, alpha=0.5, label='Reflectance-computing region')

    # Dashed rectangle for watershed excluded region (left part, larger)
    plt.plot([0, col_watershed_start, col_watershed_start, 0, 0],
            [0, 0, height, height, 0], 'b--', linewidth=2, alpha=0.5, label='Watershed excluded region')

    plt.legend(loc='upper right')
    plt.savefig(f"{data_dir.parent}/{img_name.stem}-complete.jpg", dpi=1200)
    plt.savefig(f"{data_dir.parent}/{img_name.stem}-complete-HQ.jpg", dpi=2400)
    if show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    ## input file:
    p.add_argument(
        "input_path",
        type=str,
        help="Input HDR file name",
    )
    p.add_argument(
        "--crop-idx-dim1",
        type=int,
        default=800, ## 1300 for first experiment, 1100 for second experiment
        help="Column index separating spectralon from scene (default: 1300)",
    )
    p.add_argument(
        "--reflectance-trim",
        type=int,
        default=700, # 350 for first experiment
        help="Reflectance uses columns [: crop_idx_dim1 - this] (default: 350)",
    )
    p.add_argument(
        "--watershed-trim",
        type=int,
        default=500,
        help="Watershed and crops use columns from crop_idx_dim1 - this (default: 500)",
    )
    args = p.parse_args()
    input_path = Path(args.input_path)
    data_dir = input_path.parent
    img_name = Path(input_path.name)

    args = set_parameters(args, input_path)
    plot_hdr_to_jpg(data_dir, img_name, args.crop_idx_dim1, args.reflectance_trim, args.watershed_trim, show=False)
