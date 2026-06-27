"""Build ``brightest_bands.csv`` for a directory of ENVI HDR images (optional input to the RGB pipeline)."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import spectral as sp

from grain_hs.spectralon import brightest_band as brightest_band_core


def retrieve_all_brightest_bands_to_csv(img_dir: str | os.PathLike[str], csv_dir: str | os.PathLike[str]) -> None:
    """Write ``brightest_bands.csv`` with columns ``band``, ``max_ref`` indexed by image stem.

    For each ``*.hdr`` in ``img_dir``, opens the cube and runs :func:`grain_hs.spectralon.brightest_band`
    with a tqdm progress bar over bands when ``tqdm`` is installed.

    The CSV format matches :func:`grain_hs.hdr_grains.load_brightest_csv` (pandas-style index column).
    """
    img_dir = os.fspath(img_dir)
    csv_dir = os.fspath(csv_dir)

    list_fn = os.listdir(img_dir)
    list_hdr_fn = sorted(x for x in list_fn if "hdr" in x)

    os.makedirs(csv_dir, exist_ok=True)
    out_path = os.path.join(csv_dir, "brightest_bands.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["", "band", "max_ref"])
        for hdr_name in list_hdr_fn:
            stem = hdr_name[:-4]
            path = os.path.join(img_dir, hdr_name)
            img = sp.open_image(path)
            band, max_ref = brightest_band_core(img, use_progress_bar=True)
            w.writerow([stem, band, max_ref])


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m grain_hs.brightest_csv --img-dir DIR --out-dir DIR``."""
    p = argparse.ArgumentParser(
        description="Scan HDR files and write brightest_bands.csv (index + band, max_ref).",
    )
    p.add_argument(
        "--img-dir",
        type=Path,
        required=True,
        help="Directory containing .hdr (and matching .hyspex) files",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory where brightest_bands.csv will be written",
    )
    args = p.parse_args(argv)
    retrieve_all_brightest_bands_to_csv(args.img_dir.resolve(), args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
