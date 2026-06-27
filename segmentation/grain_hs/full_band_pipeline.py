"""HDR → per-grain NPZ with full spectral depth (keys ``X``).

PIXEL VALUE TRANSFORMATIONS in this pipeline:
=============================================
1. segment_grains_from_hdr() loads the full hyperspectral cube (all bands)
2. For segmentation: same reflectance normalization as rgb_pipeline (using single band)
3. Per-grain extraction via grain_segmentation()
4. **CRITICAL DIFFERENCE from rgb_pipeline**: This pipeline DOES normalize by spectralon
   - Line 51: seg /= means_over_spectralon  # APPLIES per-channel normalization
   - This produces values roughly in range 0-2 (relative to spectralon=1.0)
   - However, this is applied to the FULL BAND cube, not just RGB

NOTE: This normalization divides ALL bands by the RGB spectralon means, which may
not be appropriate for non-RGB bands. Consider reviewing this for spectroscopic analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from grain_hs.hdr_grains import segment_grains_from_hdr
from grain_hs.segmentation import grain_segmentation


def process_one_hdr_full_bands(
    hdr_path: Path,
    output_dir: Path,
    *,
    crop_idx_dim1: int,
    reflectance_crop_trim: int,
    watershed_crop_trim: int,
    area_range: tuple[int, int],
    solidity: float,
    binary_thresh: float,
) -> None:
    """Write one full-band NPZ per grain (``X``: H×W×n_bands)."""

    img_name, grains, image_cropped_spectralon, labelled_cropped_spectralon, means_over_spectralon = segment_grains_from_hdr(
        hdr_path,
        crop_idx_dim1=crop_idx_dim1,
        reflectance_crop_trim=reflectance_crop_trim,
        watershed_crop_trim=watershed_crop_trim,
        area_range=area_range,
        solidity=solidity,
        binary_thresh=binary_thresh,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, grain in enumerate(grains):
        seg = grain_segmentation(image_cropped_spectralon, labelled_cropped_spectralon, grain)
        out_path = out_dir / f"grain{i}_{img_name}.npz"
        np.savez_compressed(
            out_path,
            X=seg,
            means_over_spectralon=means_over_spectralon,
        )
