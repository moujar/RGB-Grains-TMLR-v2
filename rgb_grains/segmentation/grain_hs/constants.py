"""Shared numeric defaults for hyperspectral grain extraction.

CONSTANT VALUES AND THEIR MEANING:
===================================
RGB_BANDS: Selected bands for RGB visualization from the 240-band Hyspex cube.
   - Values (15, 52, 80) are the MANUFACTURER's recommended RGB bands
   - Previous version used (22, 53, 89) from earlier Phuoc preprocessing
   - Band 15 ≈ 650nm (Red), Band 52 ≈ 550nm (Green), Band 80 ≈ 450nm (Blue)

SPECTRALON_ROW_COUNT: Number of rows from top used for spectralon ROI
   - Spectralon appears at top-left of image (white reference panel)

SPECTRALON_COL_START: Column offset to skip dark regions at image edge
   - Actual spectralon starts around column 150
"""

from __future__ import annotations


# Three-band RGB-style export (same bands as ``npz_fullband_to_rgb.py`` / ``grain_hs.npz_rgb_reduce``).
# RGB_BANDS: tuple[int, int, int] = (22, 53, 89) ## les valeurs du preproc de Phuoc et des etudiants dans la version 1.
RGB_BANDS: tuple[int, int, int] = (15, 52, 80) ## les valeurs du constructeur

# Spectralon ROI for brightest-band and reference statistics (rows, col start).
SPECTRALON_ROW_COUNT = 100
SPECTRALON_COL_START = 150

def set_parameters(args, inp):
   if "SCOOP-R2022-bacs" in str(inp):
      args.crop_idx_dim1 = 2000
      args.reflectance_trim = 1250
      args.watershed_trim = 750
      args.area_min = 3000
      args.area_max = 20000
   elif "perfomix" in str(inp):
      args.crop_idx_dim1 = 2000
      args.reflectance_trim = 1500
      args.watershed_trim = 850
      # args.area_min = 3000
      # args.area_max = 20000
   return args
