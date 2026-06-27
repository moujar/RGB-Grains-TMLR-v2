"""HDR → full-band per-grain NPZ (keys X, y)"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from grain_hs.full_band_pipeline import process_one_hdr_full_bands


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hyspex/HDR cube → full-spectrum (all bands) grain NPZ files (keys X, y).",
    )
    p.add_argument(
        "input",
        type=Path,
        help="Path to .hdr file, or directory of .hdr files",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory for NPZ files",
    )
    # p.add_argument("--variety", type=str, default=None)
    # --start-index removed: indexing always starts from 0 per file
    p.add_argument("--crop-idx-dim1", type=int, default=1300)
    p.add_argument("--reflectance-trim", type=int, default=350)
    p.add_argument("--watershed-trim", type=int, default=500)
    p.add_argument("--area-min", type=int, default=9000) ## used to be at 3000 !!  but then sometimes a piece of the spectralon is inside, in the BACS dataset
    p.add_argument("--area-max", type=int, default=20000)
    p.add_argument("--solidity", type=float, default=0.75)
    p.add_argument("--binary-thresh", type=float, default=0.15)
    # p.add_argument("--segmentation-band", type=int, default=None)
    # p.add_argument(
    #     "--brightest-csv",
    #     type=Path,
    #     default=None,
    #     help="Optional brightest_bands.csv",
    # )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inp = args.input.expanduser().resolve()
    out = args.output.expanduser().resolve()
    area_range = (args.area_min, args.area_max)

    if inp.is_dir():
        hdr_files = sorted(inp.glob("*.hdr"))
        if not hdr_files:
            print(f"No .hdr files in {inp}", file=sys.stderr)
            return 1
    else:
        if inp.suffix.lower() != ".hdr":
            print("When input is a file, it must be a .hdr path.", file=sys.stderr)
            return 1
        hdr_files = [inp]

    for hdr in hdr_files:
        print(f"Processing {hdr.name} (full bands) …")
        process_one_hdr_full_bands(
            hdr,
            out,
            crop_idx_dim1=args.crop_idx_dim1,
            reflectance_crop_trim=args.reflectance_trim,
            watershed_crop_trim=args.watershed_trim,
            area_range=area_range,
            solidity=args.solidity,
            binary_thresh=args.binary_thresh,
        )
    print("Done.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())