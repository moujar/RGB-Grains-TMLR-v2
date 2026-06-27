#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
from tqdm import tqdm


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

    args = parser.parse_args()

    input_path = Path(args.input)

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
        convert_folder(
            input_path,
            args.output,
            recursive=args.recursive,
            dpi=args.dpi,
            transpose=args.transpose,
            normalize=not args.no_normalize,
            cmap=args.cmap,
        )
        print("Conversion completed")

    else:
        print(f"Input path does not exist: {args.input}")


if __name__ == "__main__":
    main()
