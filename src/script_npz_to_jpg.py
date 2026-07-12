#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm

# 5.5 by-size cleaning: default minimum active-pixel area per dataset.
# BACS: Martin's criterion (np.sum(img>0.01) < 110000 -> exclude; loses some
# half-grains, acceptable). perfomix: no committed value yet ("to be studied"),
# must be passed explicitly via --min-area.
DEFAULT_MIN_AREA = {"bacs": 110000, "SCOOP": 110000}


def grain_active_area(img):
    """Number of "active" (non-background) pixels, i.e. the 5.5 by-size cleaning criterion."""
    return int(np.sum(img > 0.01))


def convert_npz_to_jpg(
    npz_path, output_path=None, dpi=150, cmap=None, transpose=False, normalize=True
):
    """
    Convert a single NPZ grain image file to JPG.

    Args:
        npz_path: Path to input .npz file
        output_path: Path for output .jpg file (optional)
        dpi: DPI for output image
        cmap: Colormap (None for RGB, use 'gray' for single channel)
        transpose: If True, transpose axes from (C, H, W) to (H, W, C)
        normalize: If True, normalize image to 0-1 range
    """
    # Load NPZ file
    data = np.load(npz_path)
    img = data["x"]

    # Handle different channel orders
    if transpose:
        if img.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
            img = np.transpose(img, (1, 2, 0))

    # Normalize image
    if normalize:
        img = img.astype(np.float32)
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)

    # Create figure with no borders
    fig = plt.figure(figsize=(img.shape[1] / dpi, img.shape[0] / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    # Plot image
    if cmap is None:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap=cmap)

    # Determine output path
    if output_path is None:
        output_path = Path(npz_path).with_suffix(".jpg")

    # Save image
    fig.savefig(output_path, dpi=dpi, pad_inches=0)
    plt.close(fig)

    return output_path


def convert_folder(input_folder, output_folder=None, recursive=False, **kwargs):
    """
    Convert all NPZ files in a folder to JPG.

    Args:
        input_folder: Input folder containing .npz files
        output_folder: Output folder for .jpg files (optional)
        recursive: Search recursively in subdirectories
        **kwargs: Additional arguments passed to convert_npz_to_jpg
    """
    input_path = Path(input_folder)

    # Find all npz files
    if recursive:
        npz_files = list(input_path.rglob("*.npz"))
    else:
        npz_files = list(input_path.glob("*.npz"))

    if not npz_files:
        print(f"No .npz files found in {input_folder}")
        return

    print(f"Found {len(npz_files)} .npz files")

    # Process each file
    for npz_file in tqdm(npz_files, desc="Converting"):
        if output_folder is None:
            output_path = None
        else:
            output_path = Path(output_folder) / npz_file.relative_to(
                input_path
            ).with_suffix(".jpg")
            output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            convert_npz_to_jpg(npz_file, output_path, **kwargs)
        except Exception as e:
            print(f"Error converting {npz_file}: {e}")


def exclude_small_grains(npz_files, output_dir, min_area=None, max_area=None, key="x"):
    """5.5 (relatively easy) by-size cleaning: move outlier-area grain crops from a
    `_processed` folder to a sibling `_excluded` folder, plus a QC area histogram.
    Returns the list of excluded npz file paths."""
    if not npz_files:
        return []
    root_path = str(npz_files[0]).split("_processed")[0]
    excluded_folder = Path(root_path + "_excluded")
    excluded_folder.mkdir(exist_ok=True)

    areas, excluded = [], []
    for f in tqdm(npz_files, desc="Checking grain areas"):
        img = np.load(f)[key]
        area = grain_active_area(img)
        areas.append(area)
        too_small = min_area is not None and area < min_area
        too_large = max_area is not None and area > max_area
        if too_small or too_large:
            tag = "small" if too_small else "large"
            plt.figure()
            plt.imshow(img)
            plt.savefig(Path(output_dir) / f"{tag}Area={area}_{f.stem}.png", dpi=300)
            plt.close()
            shutil.move(str(f), excluded_folder / f.name)
            excluded.append(f)

    plt.figure()
    plt.title(f"Histogram of grain areas for {len(areas)} images")
    counts, edges = np.histogram(areas, bins=50)
    plt.plot(edges[1:], counts)
    plt.xlabel("Area (pixels)")
    plt.ylabel("Frequency")
    plt.savefig(Path(output_dir) / "histogram_of_areas_of_grains.png", dpi=300)
    plt.close()
    print(f"[*] By-size cleaning: excluded {len(excluded)}/{len(npz_files)} grain(s) -> {excluded_folder}")
    return excluded


def exclude_outliers_od(npz_files, contamination=0.02, key="x"):
    """5.5 (bigger work) by Outlier Detection: flag & move outlier grains (no labels
    used) using a hand-picked feature vector (active area + per-channel active-pixel
    mean/std), one sklearn IsolationForest fit per folder. Provisional -- a
    learned-embedding-based approach would likely do better; revisit if too coarse."""
    from sklearn.ensemble import IsolationForest

    if not npz_files:
        return []
    root_path = str(npz_files[0]).split("_processed")[0]
    excluded_folder = Path(root_path + "_excluded")
    excluded_folder.mkdir(exist_ok=True)

    feats = []
    for f in tqdm(npz_files, desc="Extracting OD features"):
        img = np.load(f)[key]
        mask = img > 0.01
        area = mask.sum()
        means = [img[..., c][mask[..., c]].mean() if mask[..., c].any() else 0.0 for c in range(img.shape[-1])]
        stds = [img[..., c][mask[..., c]].std() if mask[..., c].any() else 0.0 for c in range(img.shape[-1])]
        feats.append([area, *means, *stds])
    feats = np.asarray(feats, dtype=np.float64)

    clf = IsolationForest(contamination=contamination, random_state=42)
    is_outlier = clf.fit_predict(feats) == -1

    excluded = [f for f, outlier in zip(npz_files, is_outlier) if outlier]
    for f in excluded:
        shutil.move(str(f), excluded_folder / f.name)
    print(f"[*] OD cleaning: excluded {len(excluded)}/{len(npz_files)} grain(s) (contamination={contamination}) -> {excluded_folder}")
    return excluded


def apply_manual_exclusion(jpg_folder, npz_folder):
    """5.5 (easy but time-consuming) manual cleaning: export jpgs with this script's
    normal (non-cleaning) mode, delete the anomalous ones by hand in `jpg_folder`,
    then call this to move the corresponding (now jpg-less) .npz files from
    `npz_folder` to a sibling `_excluded` folder."""
    jpg_folder, npz_folder = Path(jpg_folder), Path(npz_folder)
    all_npz = {p.stem: p for p in npz_folder.glob("*.npz")}
    kept_jpg_stems = {p.stem for p in jpg_folder.glob("*.jpg")}
    to_exclude = [p for stem, p in all_npz.items() if stem not in kept_jpg_stems]

    if not to_exclude:
        print("[*] Nothing to exclude: every .npz still has a matching .jpg.")
        return []
    root_path = str(npz_folder).split("_processed")[0]
    excluded_folder = Path(root_path + "_excluded")
    excluded_folder.mkdir(exist_ok=True)
    for f in to_exclude:
        shutil.move(str(f), excluded_folder / f.name)
    print(f"[*] Manual cleaning: excluded {len(to_exclude)} grain(s) (jpg deleted by hand) -> {excluded_folder}")
    return to_exclude


def main():
    parser = argparse.ArgumentParser(description="Convert NPZ image files to JPG")
    parser.add_argument("input", help="Input NPZ file or folder")
    parser.add_argument("-o", "--output", help="Output file or folder")
    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Process folders recursively"
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="Output DPI (default: 150)"
    )
    parser.add_argument(
        "--transpose", action="store_true", help="Transpose (C,H,W) to (H,W,C)"
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="Skip image normalization"
    )
    parser.add_argument("--cmap", help="Colormap (e.g. gray)")
    parser.add_argument("--exclude-small", action="store_true",
                         help="5.5 by-size: also move outlier-area grains to a sibling _excluded folder (folder input only).")
    parser.add_argument("--min-area", type=int, default=None,
                         help="Min active-pixel area for --exclude-small (default per --dataset, else required).")
    parser.add_argument("--max-area", type=int, default=None, help="Optional max active-pixel area for --exclude-small.")
    parser.add_argument("--dataset", choices=list(DEFAULT_MIN_AREA.keys()), default=None,
                         help="Dataset name, to pick --min-area's default (perfomix has none yet: 'to be studied').")
    parser.add_argument("--exclude-outliers", action="store_true",
                         help="5.5 by OD: also move IsolationForest-flagged outlier grains to _excluded (folder input only).")
    parser.add_argument("--od-contamination", type=float, default=0.02,
                         help="Expected outlier fraction for --exclude-outliers (default: 0.02).")
    parser.add_argument("--apply-manual-exclusion", metavar="JPG_FOLDER",
                         help="5.5 manual: JPG_FOLDER is a jpg export (this script's normal mode) that a human has "
                              "pruned by hand; move the .npz files (from the folder given as `input`) that no longer "
                              "have a matching jpg into a sibling _excluded folder. Skips jpg conversion.")

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.apply_manual_exclusion:
        apply_manual_exclusion(args.apply_manual_exclusion, input_path)
        return

    if input_path.is_file():
        if input_path.suffix != ".npz":
            print(f"Input file must be .npz")
            return

        output_path = convert_npz_to_jpg(
            input_path,
            args.output,
            dpi=args.dpi,
            transpose=args.transpose,
            normalize=not args.no_normalize,
            cmap=args.cmap,
        )
        print(f"Converted: {output_path}")

    elif input_path.is_dir():
        output_folder = Path(args.output) if args.output else input_path.parent / (input_path.name + "_jpgs")
        convert_folder(
            input_path,
            output_folder,
            recursive=args.recursive,
            dpi=args.dpi,
            transpose=args.transpose,
            normalize=not args.no_normalize,
            cmap=args.cmap,
        )
        print("Conversion completed")

        if args.exclude_small or args.exclude_outliers:
            output_folder.mkdir(parents=True, exist_ok=True)
            npz_files = sorted(input_path.rglob("*.npz") if args.recursive else input_path.glob("*.npz"))
            if args.exclude_small:
                min_area = args.min_area
                if min_area is None and args.dataset is not None:
                    min_area = DEFAULT_MIN_AREA.get(args.dataset)
                if min_area is None:
                    print("[!] --exclude-small needs --min-area (or --dataset bacs/SCOOP); "
                          "perfomix's threshold is still 'to be studied'. Skipping.")
                else:
                    excluded = exclude_small_grains(npz_files, output_folder, min_area=min_area, max_area=args.max_area)
                    npz_files = [f for f in npz_files if f not in excluded]
            if args.exclude_outliers:
                exclude_outliers_od(npz_files, contamination=args.od_contamination)

    else:
        print(f"Input path does not exist: {args.input}")


if __name__ == "__main__":
    main()
