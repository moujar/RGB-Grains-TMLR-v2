"""Run the RGB-Grains training CLI on a Modal GPU.

The repository code is baked into the Modal image. Processed datasets and run
artifacts live in separate persistent Volumes so they survive container exits.
"""

from __future__ import annotations

import shlex
import tarfile
from pathlib import Path, PurePosixPath

import modal


APP_NAME = "rgb-grains-training"
REMOTE_ROOT = PurePosixPath("/root/rgb-grains")
DATA_MOUNT = PurePosixPath("/mnt/rgb-grains-data")
OUTPUT_MOUNT = PurePosixPath("/mnt/rgb-grains-output")
DATA_VOLUME_NAME = "rgb-grains-data"
OUTPUT_VOLUME_NAME = "rgb-grains-output"

LOCAL_ROOT = Path(__file__).resolve().parent
REMOTE_DEPENDENCIES = [
    "torch>=2.1",
    "numpy>=1.23",
    "pandas>=1.5",
    "scikit-learn>=1.2",
    "seaborn>=0.12",
    "matplotlib>=3.7",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*REMOTE_DEPENDENCIES)
    .add_local_dir(
        LOCAL_ROOT / "src",
        remote_path=str(REMOTE_ROOT / "src"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "config.json",
        remote_path=str(REMOTE_ROOT / "config.json"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "config (debug).json",
        remote_path=str(REMOTE_ROOT / "config (debug).json"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "config (serious).json",
        remote_path=str(REMOTE_ROOT / "config (serious).json"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "perfomix_mixtures.csv",
        remote_path=str(REMOTE_ROOT / "perfomix_mixtures.csv"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "perfomix_mixtures.tsv",
        remote_path=str(REMOTE_ROOT / "perfomix_mixtures.tsv"),
        copy=True,
    )
    .add_local_file(
        LOCAL_ROOT / "microplot_exclusions.json",
        remote_path=str(REMOTE_ROOT / "microplot_exclusions.json"),
        copy=True,
    )
    .env(
        {
            "MPLBACKEND": "Agg",
            "PYTHONUNBUFFERED": "1",
            "PYTHONUTF8": "1",
        }
    )
    .workdir(str(REMOTE_ROOT))
)

data_volume = modal.Volume.from_name(
    DATA_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
output_volume = modal.Volume.from_name(
    OUTPUT_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
app = modal.App(APP_NAME, image=image)


def _replace_with_symlink(link: Path, target: Path, *, directory: bool) -> None:
    target.mkdir(parents=True, exist_ok=True) if directory else target.parent.mkdir(
        parents=True, exist_ok=True
    )
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"Cannot mount persistent storage: {link} already exists")
    link.symlink_to(target, target_is_directory=directory)


@app.function(
    cpu=4.0,
    memory=8192,
    timeout=2 * 60 * 60,
    volumes={str(DATA_MOUNT): data_volume},
)
def extract_data_archive(
    archive_name: str = "perfomix-data.tar",
    delete_archive: bool = True,
) -> dict[str, object]:
    """Safely extract a dataset TAR previously uploaded to the data Volume."""
    data_mount = Path(str(DATA_MOUNT)).resolve()
    archive_path = (data_mount / archive_name).resolve()
    if archive_path.parent != data_mount:
        raise ValueError("archive_name must name a file at the Volume root")
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        for member in members:
            destination = (data_mount / member.name).resolve()
            if destination != data_mount and data_mount not in destination.parents:
                raise ValueError(f"Unsafe path in archive: {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    f"Unsupported archive member type: {member.name!r}"
                )
        archive.extractall(data_mount, members=members)

    if delete_archive:
        archive_path.unlink()
    data_volume.commit()

    processed_dirs = sorted(path.name for path in data_mount.glob("*_processed"))
    extracted_files = sum(1 for member in members if member.isfile())
    return {
        "archive": archive_name,
        "extracted_files": extracted_files,
        "processed_directories": processed_dirs,
        "archive_deleted": delete_archive,
    }


@app.function(
    gpu="L4",
    cpu=8.0,
    memory=65536,
    timeout=23 * 60 * 60,
    max_containers=1,
    volumes={
        str(DATA_MOUNT): data_volume.with_mount_options(read_only=True),
        str(OUTPUT_MOUNT): output_volume,
    },
)
def run_training(train_args: list[str]) -> dict[str, object]:
    """Execute one training command and persist all generated artifacts."""
    import os
    import subprocess
    import sys

    remote_root = Path(str(REMOTE_ROOT))
    data_mount = Path(str(DATA_MOUNT))
    output_mount = Path(str(OUTPUT_MOUNT))

    processed_dirs = list(data_mount.glob("*_processed"))
    if not processed_dirs:
        raise RuntimeError(
            f"No *_processed dataset folders found in {data_mount}. "
            f"Upload data to the {DATA_VOLUME_NAME!r} Volume first."
        )

    _replace_with_symlink(remote_root / "data", data_mount, directory=True)
    _replace_with_symlink(
        remote_root / "expe", output_mount / "expe", directory=True
    )
    _replace_with_symlink(
        remote_root / "models", output_mount / "models", directory=True
    )
    _replace_with_symlink(
        remote_root / "overall_perf_summary.json",
        output_mount / "overall_perf_summary.json",
        directory=False,
    )

    command = [
        sys.executable,
        str(remote_root / "src" / "script_train_5_models_singleSplit.py"),
        *train_args,
    ]
    print("[*] Modal command:", shlex.join(command), flush=True)
    print("[*] CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

    result = None
    try:
        result = subprocess.run(command, cwd=remote_root, check=False)
    finally:
        output_volume.commit()

    if result is None or result.returncode != 0:
        return_code = None if result is None else result.returncode
        raise RuntimeError(f"Training command failed with exit code {return_code}")

    experiment_dirs = sorted(
        (path.name for path in (output_mount / "expe").iterdir() if path.is_dir())
    )
    return {
        "return_code": result.returncode,
        "output_volume": OUTPUT_VOLUME_NAME,
        "experiment_directories": experiment_dirs,
    }


@app.local_entrypoint()
def main(
    gpu: str = "L4",
    train_args: str = (
        '--config "config (debug).json" --debugMode 100 '
        "--splitting-choice random_year1only --yearChosen 2020 "
        "--tag modal_smoke"
    ),
    preview: bool = False,
) -> None:
    """Run one job; pass normal training flags inside --train-args."""
    allowed_gpus = {"T4", "L4", "A10", "L40S", "A100-40GB"}
    if gpu not in allowed_gpus:
        raise ValueError(
            f"Unsupported GPU {gpu!r}. Choose one of {sorted(allowed_gpus)}."
        )

    parsed_args = shlex.split(train_args)
    if preview:
        print("[*] Preview only; no Modal GPU will be started.")
        print("[*] GPU:", gpu)
        print("[*] Training arguments:", parsed_args)
        return
    result = run_training.with_options(gpu=gpu).remote(parsed_args)
    print("[*] Modal run completed:", result)
