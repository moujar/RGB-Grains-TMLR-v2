"""NPZ with full cube ``X`` → NPZ with 3 bands ``x`` (Lisheng-style second step)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from grain_hs.constants import RGB_BANDS


def run_reduce(src: Path, out: Path) -> int:
    """Convert each ``*.npz`` under ``src`` (key ``X``) to 3-band NPZ in ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    bands = list(RGB_BANDS)
    files = sorted(src.glob("*.npz"))
    n = len(files)
    if n == 0:
        print(f"No .npz files in {src}", file=sys.stderr)
        return 1

    print(f"Found {n} files.")
    step = max(1, n // 100)
    next_progress = step

    for i, npz_path in enumerate(files, 1):
        with np.load(npz_path) as data:
            cube = data["X"]
            y = data["y"]
            rgb_cube = cube[:, :, bands]

        out_path = out / npz_path.name
        np.savez_compressed(
            out_path,
            x=rgb_cube,
            y=y,
            original_filename=npz_path.name,
            bands=np.array(bands),
        )

        if i >= next_progress:
            print(f"{100 * i // n}% ({i}/{n})")
            next_progress += step

    print("Done.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
    )
    p.add_argument(
        "--src",
        type=Path,
        default=Path("clean_data"),
        help="Directory of input *.npz with key X (default: ./clean_data)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("clean_data_rgb"),
        help="Output directory (default: ./clean_data_rgb)",
    )
    args = p.parse_args(argv)
    return run_reduce(
        args.src.expanduser().resolve(),
        args.out.expanduser().resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
