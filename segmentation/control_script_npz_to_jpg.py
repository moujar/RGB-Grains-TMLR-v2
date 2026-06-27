#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil


def convert_npz_to_jpg(
    npz_path, output_path, means_over_spectralon=None, dpi=150, export_jpg=False
):
    """
    Convert a single NPZ grain image file to JPG.

    Args:
        npz_path: Path to input .npz file
        output_path: Path for output .jpg file (optional)
        dpi: DPI for output image
    """
    # Load NPZ file
    data = np.load(npz_path)
    img = data["x"]
    means_over_spectralon = data["means_over_spectralon"]
    img  = img*1.0 / means_over_spectralon
    img = img[:,:, ::-1]  ## BGR -> RGB conversion
    ## cursed line: this one is NOT equivalent to the previous (correct) one:
    # img[:,:,0] , img[:,:,2] = img[:,:,2] , img[:,:,0]

    if export_jpg:
        plt.imshow(img)
        plt.savefig(output_path, dpi=dpi, pad_inches=0)
        plt.close()

    return output_path, img


def convert_folder(input_folder, output_folder, means_over_spectralon=None, recursive=False, **kwargs):
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
        npz_files = list(input_path.rglob("grain*.npz"))
    else:
        npz_files = list(input_path.glob("grain*.npz"))
    if not npz_files:
        print(f"No .npz files found in {input_folder}")
        return
    print(f"Found {len(npz_files)} .npz files")

    # Process each file
    imgs = []
    for npz_file in tqdm(npz_files, desc="Reading npz files"):
        output_path = Path(output_folder) / npz_file.relative_to(
            input_path
        ).with_suffix(".jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _, img = convert_npz_to_jpg(npz_file, output_path, means_over_spectralon, **kwargs)
        imgs.append(img)

    imgs = np.array(imgs)
    return imgs, npz_files

# def main():
parser = argparse.ArgumentParser(description="Convert NPZ image files to JPG")
parser.add_argument("input", help="Input NPZ file or folder")
parser.add_argument("-o", "--output", default=None, help="Output file or folder")
parser.add_argument(
    "-r", "--recursive", action="store_true", help="Process folders recursively"
)
parser.add_argument(
    "--dpi", type=int, default=150, help="Output DPI (default: 150)"
)
parser.add_argument(
    "--export-jpg", action="store_true", default=False, help="Export JPG files. Default is False"
)
args = parser.parse_args()

input_path = Path(args.input)
if args.output is not None:
    output_path = Path(args.output)
else:
    output_path = input_path.parent / (input_path.name + "_jpgs")
output_path.mkdir(exist_ok=True)

if input_path.is_file():
    if input_path.suffix != ".npz":
        print(f"Input file must be .npz")
        raise SystemExit(1)
    output_path = convert_npz_to_jpg(
        input_path,
        args.output,
        dpi=args.dpi,
        export_jpg=args.export_jpg,
    )
    print(f"Converted: {output_path}")

elif input_path.is_dir():
    imgs, npz_files = convert_folder(
        input_path,
        output_path,
        recursive=args.recursive,
        dpi=args.dpi,
        export_jpg=args.export_jpg,
        )
    print("Conversion completed")

else:
    print(f"Input path does not exist: {args.input}")
    raise SystemExit(1)

## create the excluded folder:
rootPath = str(input_path).split("_processed")[0]
excluded_folder = Path(rootPath + "_excluded")
excluded_folder.mkdir(exist_ok=True)

## get the number of active pixels (area of grain) in each image:
areas = []
for i, img in enumerate(imgs):
    areas.append(np.sum(img > 0.01))
    if areas[-1] < 15000:
        print(f"Image {i}: {areas[-1]} pixels")
        plt.figure()
        plt.imshow(img)
        plt.savefig(f"{output_path}/smallArea={areas[-1]}_{npz_files[i].name[:-4]}.png", dpi=300)
        plt.close()
        ## move the npz file from its initial location in a _processed  folder into an _excluded folder:
        shutil.move(npz_files[i], excluded_folder / npz_files[i].name)


    if areas[-1] > 40000:
        print(f"Image {i}: {areas[-1]} pixels")
        plt.figure()
        plt.imshow(img)
        plt.savefig(f"{output_path}/largeArea={areas[-1]}_{npz_files[i].name[:-4]}.png", dpi=300)
        plt.close()


plt.figure()
plt.title(f"Histogram of grain areas for {len(areas)} images")
b,c = np.histogram(areas, bins=50)
plt.plot(c[1:], b)
plt.xlabel("Area (pixels)")
plt.ylabel("Frequency")
plt.savefig(f"{output_path}/histogram_of_areas_of_grains.png", dpi=300)
plt.show()
# plt.close()

img = imgs.reshape(-1,3)
plt.figure()
plt.title(f"Histogram of RGB values for {len(imgs)} images")
b, c = np.histogram(img[:,0].flatten(), bins=256)
plt.plot(c[10:], b[9:], label="Blue", color="blue")
b, c = np.histogram(img[:,1].flatten(), bins=256)
plt.plot(c[10:], b[9:], label="Green", color="green")
b, c = np.histogram(img[:,2].flatten(), bins=256)
plt.plot(c[10:], b[9:], label="Red", color="red")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.legend()
plt.savefig(f"{output_path}/histogram-of-colors.png", dpi=300)
# plt.show()
plt.close()

# if __name__ == "__main__":
#     main()
