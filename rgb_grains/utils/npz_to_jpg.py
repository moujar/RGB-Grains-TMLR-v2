#!/usr/bin/env python3
"""Convert grain NPZ crops to JPG for quick visual inspection.

Consolidates what used to be two near-duplicate scripts
(``src/script_npz_to_jpg.py`` and
``segmentation/control_script_npz_to_jpg.py``) into one tool with two
normalization modes:

  * ``minmax``     (default) -- generic min/max normalization of the ``x`` array.
  * ``spectralon`` -- segmentation-pipeline specific: normalize by the
    ``means_over_spectralon`` array stored alongside ``x`` and swap to RGB
    channel order, matching how the raw hyperspectral RGB crops were meant
    to be viewed.

``--exclude-small`` additionally applies the TODO 5.5 by-size cleaning
criterion (see ``rgb_grains/data/cleaning.py``) and moves excluded files from
a ``..._processed`` folder into a sibling ``..._excluded`` folder.
"""
import numpy as np
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm

from rgb_grains.data.cleaning import grain_active_area, DEFAULT_MIN_AREA


def _load_and_normalize(npz_path, mode="minmax", transpose=False, normalize=True):
    data = np.load(npz_path)
    img = data["x"]

    if mode == "spectralon":
        means = data["means_over_spectralon"]
        img = img * 1.0 / means
        img = img[:, :, ::-1]  # BGR -> RGB
        return img

    if transpose and img.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
        img = np.transpose(img, (1, 2, 0))
    if normalize:
        img = img.astype(np.float32)
        img_min, img_max = img.min(), img.max()
        img = (img - img_min) / max(img_max - img_min, 1e-8)
    return img


def convert_npz_to_jpg(
    npz_path, output_path=None, dpi=150, cmap=None, transpose=False,
    normalize=True, mode="minmax",
):
    """Convert a single NPZ grain image file to JPG. Returns (output_path, img)."""
    img = _load_and_normalize(npz_path, mode=mode, transpose=transpose, normalize=normalize)

    fig = plt.figure(figsize=(img.shape[1] / dpi, img.shape[0] / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(img, cmap=cmap)

    if output_path is None:
        output_path = Path(npz_path).with_suffix(".jpg")
    fig.savefig(output_path, dpi=dpi, pad_inches=0)
    plt.close(fig)
    return output_path, img


def convert_folder(input_folder, output_folder=None, recursive=False, pattern="*.npz", **kwargs):
    """Convert all NPZ files in a folder to JPG. Returns (imgs, npz_files)."""
    input_path = Path(input_folder)
    npz_files = sorted(input_path.rglob(pattern) if recursive else input_path.glob(pattern))

    if not npz_files:
        print(f"No .npz files found in {input_folder}")
        return [], []
    print(f"Found {len(npz_files)} .npz files")

    imgs = []
    kept_files = []
    for npz_file in tqdm(npz_files, desc="Converting"):
        if output_folder is None:
            output_path = None
        else:
            output_path = Path(output_folder) / npz_file.relative_to(input_path).with_suffix(".jpg")
            output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            _, img = convert_npz_to_jpg(npz_file, output_path, **kwargs)
            imgs.append(img)
            kept_files.append(npz_file)
        except Exception as e:
            print(f"Error converting {npz_file}: {e}")
    return imgs, kept_files


def exclude_small_grains(npz_files, imgs, output_dir, min_area=None, max_area=None):
    """TODO 5.5: move outlier-area grains from a `_processed` folder to a sibling `_excluded` folder."""
    if not npz_files:
        return
    root_path = str(npz_files[0]).split("_processed")[0]
    excluded_folder = Path(root_path + "_excluded")
    excluded_folder.mkdir(exist_ok=True)

    areas = []
    for i, img in enumerate(imgs):
        area = grain_active_area(img)
        areas.append(area)
        too_small = min_area is not None and area < min_area
        too_large = max_area is not None and area > max_area
        if too_small or too_large:
            tag = "small" if too_small else "large"
            plt.figure()
            plt.imshow(img)
            plt.savefig(f"{output_dir}/{tag}Area={area}_{npz_files[i].stem}.png", dpi=300)
            plt.close()
            if too_small:
                shutil.move(str(npz_files[i]), excluded_folder / npz_files[i].name)

    plt.figure()
    plt.title(f"Histogram of grain areas for {len(areas)} images")
    counts, edges = np.histogram(areas, bins=50)
    plt.plot(edges[1:], counts)
    plt.xlabel("Area (pixels)")
    plt.ylabel("Frequency")
    plt.savefig(f"{output_dir}/histogram_of_areas_of_grains.png", dpi=300)
    plt.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Convert NPZ grain crops to JPG")
    parser.add_argument("input", help="Input NPZ file or folder")
    parser.add_argument("-o", "--output", help="Output file or folder")
    parser.add_argument("-r", "--recursive", action="store_true", help="Process folders recursively")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI (default: 150)")
    parser.add_argument("--mode", choices=["minmax", "spectralon"], default="minmax",
                         help="minmax: generic normalization. spectralon: divide by means_over_spectralon (segmentation QC view).")
    parser.add_argument("--transpose", action="store_true", help="Transpose (C,H,W) to (H,W,C) [minmax mode only]")
    parser.add_argument("--no-normalize", action="store_true", help="Skip image normalization [minmax mode only]")
    parser.add_argument("--cmap", help="Colormap (e.g. gray)")
    parser.add_argument("--exclude-small", action="store_true",
                         help="TODO 5.5: also apply by-size exclusion and move outliers to a sibling _excluded folder (folder input only).")
    parser.add_argument("--min-area", type=int, default=None, help="Override default min active-pixel area for --exclude-small.")
    parser.add_argument("--max-area", type=int, default=None, help="Optional max active-pixel area for --exclude-small.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    kwargs = dict(dpi=args.dpi, mode=args.mode, transpose=args.transpose,
                  normalize=not args.no_normalize, cmap=args.cmap)

    if input_path.is_file():
        if input_path.suffix != ".npz":
            print("Input file must be .npz")
            return
        output_path, _ = convert_npz_to_jpg(input_path, args.output, **kwargs)
        print(f"Converted: {output_path}")

    elif input_path.is_dir():
        output_folder = Path(args.output) if args.output else input_path.parent / (input_path.name + "_jpgs")
        output_folder.mkdir(parents=True, exist_ok=True)
        imgs, npz_files = convert_folder(input_path, output_folder, recursive=args.recursive, **kwargs)
        print("Conversion completed")
        if args.exclude_small:
            min_area = args.min_area
            if min_area is None and args.mode == "spectralon":
                # best-effort dataset detection from the folder name (perfomix vs SCOOP)
                for key, default in DEFAULT_MIN_AREA.items():
                    if key in str(input_path):
                        min_area = default
                        break
            exclude_small_grains(npz_files, imgs, output_folder, min_area=min_area, max_area=args.max_area)
    else:
        print(f"Input path does not exist: {args.input}")


if __name__ == "__main__":
    main()
