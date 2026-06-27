"""End-to-end HDR → 3-band grain NPZ export with optional JPG visualization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from grain_hs.hdr_grains import segment_grains_from_hdr
from grain_hs.segmentation import grain_segmentation

__all__ = ["process_one_hdr"]


def process_one_hdr(
    hdr_path: Path,
    output_dir: Path,
    *,
    crop_idx_dim1: int,
    reflectance_crop_trim: int,
    watershed_crop_trim: int,
    area_range: tuple[int, int],
    solidity: float,
    binary_thresh: float,
    debug: bool = False,
    output_jpg: bool = False,
) -> None:
    """Segment grains in one cube and write compressed NPZs with three bands each.

    Each grain gets a file like grain{i}_{img_name}.npz written directly to
    *output_dir* (i starts at 0 for each HDR file).

    Args:
        debug: If True, reduces image size for faster processing (testing only).
        output_jpg: If True, also saves JPG visualizations alongside NPZ files.
    """

    img_name, grains, image_cropped_spectralon, labelled_cropped_spectralon, means_over_spectralon = segment_grains_from_hdr(
        hdr_path,
        crop_idx_dim1=crop_idx_dim1,
        reflectance_crop_trim=reflectance_crop_trim,
        watershed_crop_trim=watershed_crop_trim,
        area_range=area_range,
        solidity=solidity,
        binary_thresh=binary_thresh,
        debug=debug,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, grain in enumerate(grains):
        if i==0 :
            print(f"Processing grain {i} of {len(grains)}")
        segmented_grain_image = grain_segmentation(image_cropped_spectralon, labelled_cropped_spectralon, grain)
        out_path = out_dir / f"grain{i}_{img_name}.npz"
        np.savez_compressed(
            out_path,
            x=segmented_grain_image,
            means_over_spectralon=means_over_spectralon,
        )
        if output_jpg:
            # Normalize by spectralon means for realistic colors, then flip to BGR for display
            image_to_save = segmented_grain_image.astype(np.float32) / means_over_spectralon
            image_to_save = image_to_save[:, :, ::-1]  # RGB -> BGR
            plt.figure(figsize=(4, 4))
            plt.imshow(image_to_save)
            plt.axis('off')
            plt.savefig(out_dir / f"grain{i}_{img_name}.jpg", bbox_inches='tight', pad_inches=0)
            plt.close()

    return None