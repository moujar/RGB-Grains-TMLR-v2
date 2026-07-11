"""Hyspex/HDR → 3-band grain NPZ only

PREPROCESSING PIPELINE - PIXEL VALUE FLOW:
==========================================
1. Raw HDR cube loading (int16 raw sensor values)
2. RGB band extraction (3 bands: R=15, G=52, B=80 - manufacturer values)
3. Spectralon analysis: compute mean brightness per RGB band over spectralon ROI
   - ROI: rows 0-99, cols 150-end (SPECTRALON_ROW_COUNT=100, SPECTRALON_COL_START=150)
4. Reflectance calibration (in reflectance.py): each row divided by its spectralon reference
   - This produces reflectance values (relative to spectralon=1.0)
5. Watershed segmentation on reflectance image
6. Per-grain cropping with grain_segmentation()
7. RGB values saved AS-IS (raw int16 values), NOT normalized by spectralon
   - The means_over_spectralon are saved separately but NOT applied to the output

To recover realistic colors (like script_hdr_to_jpg.py):
   - Divide each RGB channel by its mean_over_spectralon
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path
import time

from grain_hs.rgb_pipeline import process_one_hdr
from grain_hs import set_parameters
from control_script_hdr_to_jpg import plot_hdr_to_jpg

def _process_one_hdr_worker(
    hdr: Path,
    output_dir: Path,
    crop_idx_dim1: int,
    reflectance_trim: int,
    watershed_trim: int,
    area_min: int,
    area_max: int,
    solidity: float,
    binary_thresh: float,
    debug: bool,
    output_jpg: bool,
) -> tuple[str, int]:
    """Worker entry-point: process one HDR file.

    Writes NPZ files directly to *output_dir*.
    Returns (img_name, grain_count).
    """
    t0 = time.time()
    process_one_hdr(
        hdr,
        output_dir,
        crop_idx_dim1=crop_idx_dim1,
        reflectance_crop_trim=reflectance_trim,
        watershed_crop_trim=watershed_trim,
        area_range=(area_min, area_max),
        solidity=solidity,
        binary_thresh=binary_thresh,
        debug=debug,
        output_jpg=output_jpg,
    )
    grain_count = sum(1 for f in output_dir.glob(f"grain*_{hdr.stem}.npz"))
    img_name = hdr.stem
    return img_name, grain_count


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hyspex/HDR cube → compressed 3-band grain NPZ files.",
    )
    p.add_argument(
        "--input",
        type=Path,
        help="Path to .hdr file, or directory of .hdr files",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="Output directory for NPZ files. Default will create a folder of the form {input}_processed/.",
    )
    p.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Override the string matched against 'SCOOP-R2022-bacs'/'perfomix' in set_parameters() "
             "(default: derived from --input's path, which set_parameters() actually keys off of).",
    )
    # p.add_argument(
    #     "--variety",
    #     type=str,
    #     default=None,
    #     help="Label stored as y (default: parse varN from filename)",
    # )
    # --start-index removed: indexing always starts from 0 per file, filenames are non-overlapping
    p.add_argument(
        "--crop-idx-dim1",
        type=int,
        default=1300,
        help="Column index separating spectralon from scene (default: 1300)",
    )
    p.add_argument(
        "--reflectance-trim",
        type=int,
        default=350,
        help="Reflectance uses columns [: crop_idx_dim1 - this] (default: 350)",
    )
    p.add_argument(
        "--watershed-trim",
        type=int,
        default=500,
        help="Watershed and crops use columns from crop_idx_dim1 - this (default: 500)",
    )
    p.add_argument(
        "--area-min",
        type=int,
        default=3000,
        help="Minimum grain area in pixels (default: 3000)",
    )
    p.add_argument(
        "--area-max",
        type=int,
        default=20000,
        help="Maximum grain area in pixels (default: 20000)",
    )
    p.add_argument(
        "--solidity",
        type=float,
        default=0.75,
        help="Minimum solidity for regionprops (default: 0.75)",
    )
    p.add_argument(
        "--binary-thresh",
        type=float,
        default=0.10,
        help="Binary threshold on reflectance for watershed (default: 0.10 when restricting to RGB (selection of binary filter on band 80). Used to be 0.15 on band 106.)",
    )
    # p.add_argument(
    #     "--segmentation-band",
    #     type=int,
    #     default=None,
    #     help="Fixed band index for segmentation (skip brightest-band search)",
    # )
    # p.add_argument(
    #     "--brightest-csv",
    #     type=Path,
    #     default=None,
    #     help="Optional brightest_bands.csv (from python -m grain_hs.brightest_csv)",
    # )
    p.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Number of HDR files to process in parallel (default: 4)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug mode: reduce image size for faster processing",
    )
    p.add_argument(
        "--output-jpg",
        action="store_true",
        default=False,
        help="Also output JPG visualizations alongside NPZ files (default: False)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inp = args.input.expanduser().resolve()
    out = args.output
    if out is not None:
        out = Path(out)
    else: 
        out = inp.parent / (inp.name + "_processed")
    out.mkdir(parents=True, exist_ok=True)
    

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

    args = set_parameters(args, args.dataset_name if args.dataset_name else inp)

    ## plot the control for the very first hdr file:
    data_dir = inp.parent / inp.name
    img_name_0 = Path(hdr_files[0].name)
    plot_hdr_to_jpg(data_dir, img_name_0, args.crop_idx_dim1, args.reflectance_trim, args.watershed_trim, show=False)
    print("Control plots saved, please inspect them and adjust the trimming parameters accordingly")

    if args.jobs == 1 or len(hdr_files) == 1:
        for hdr in hdr_files:
            print(f"Processing {hdr.name} …")
            t0 = time.time()
            process_one_hdr(
                hdr,
                out,
                crop_idx_dim1=args.crop_idx_dim1,
                reflectance_crop_trim=args.reflectance_trim,
                watershed_crop_trim=args.watershed_trim,
                area_range=(args.area_min, args.area_max),
                solidity=args.solidity,
                binary_thresh=args.binary_thresh,
                debug=args.debug,
                output_jpg=args.output_jpg,
            )
            print(f"process_one_hdr() took {time.time() - t0:.2f} seconds")
    else:
        print(f"Processing {len(hdr_files)} HDR files with {args.jobs} workers …")
        futures = []
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.jobs,
        ) as executor:
            for hdr in hdr_files:
                futures.append(
                    executor.submit(
                        _process_one_hdr_worker,
                        hdr,
                        out,
                        args.crop_idx_dim1,
                        args.reflectance_trim,
                        args.watershed_trim,
                        args.area_min,
                        args.area_max,
                        args.solidity,
                        args.binary_thresh,
                        args.debug,
                        args.output_jpg,
                    )
                )
            for hdr, future in zip(hdr_files, futures):
                print(f"Waiting for {hdr.name} …")
                t0 = time.time()
                img_name, grain_count = future.result()
                print(f"Got {grain_count} grains for {img_name} in {time.time() - t0:.2f} seconds")
    total = sum(1 for f in out.glob("grain*.npz"))
    print(f"Done. Total grain NPZ files in output: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
