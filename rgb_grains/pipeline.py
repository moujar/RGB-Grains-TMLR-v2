"""End-to-end RGB-Grains pipeline: segment raw hyperspectral cubes (optional),
clean the resulting grain crops (TODO 5.5), then fine-tune and validate the
ConvNeXt-Tiny classifier -- on GPU automatically if one is available.

This is the single entrypoint referenced by the README for running the whole
project from raw data to a trained, evaluated model:

    # Everything, starting from already-segmented data/ (the common case):
    python -m rgb_grains.pipeline --config configs/default.json \\
        --splitting-choice muPlot-3muTrain-1muTest --dataset-choice perfomix

    # Also segment raw .hdr cubes first (needs the hyperspectral extras, see
    # rgb_grains/segmentation/requirements.txt):
    python -m rgb_grains.pipeline --raw-hdr-dir /path/to/hdr_cubes \\
        --dataset-name perfomix_2020-2021_IE_HSI_var1-8 \\
        --config configs/default.json

    # Multiple folds/splits in one call (e.g. the 3 SCOOP/BACS folds):
    python -m rgb_grains.pipeline --dataset-choice SCOOP \\
        --splitting-choice bacs_2train_1test --folds 0 1 2

Each stage can be skipped independently (``--skip-segmentation``,
``--skip-cleaning``) so this also works as a plain training+validation
entrypoint on data that's already segmented and clean.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # ── Stage 1: segmentation (raw .hdr -> *_processed/*.npz) ──
    seg = p.add_argument_group("1. segmentation (optional)")
    seg.add_argument("--raw-hdr-dir", type=str, default=None,
                      help="Directory of raw .hdr hyperspectral cubes to segment. "
                           "If omitted, segmentation is skipped and --data-dir is assumed already populated.")
    seg.add_argument("--dataset-name", type=str, default=None,
                      help="Used to name the output '{dataset-name}_processed' folder, and forwarded to the "
                           "segmentation script's own --dataset-name override so segmentation parameters are picked "
                           "correctly (must contain 'perfomix' or 'SCOOP-R2022-bacs'); "
                           "see rgb_grains/segmentation/grain_hs/constants.py:set_parameters.")
    seg.add_argument("--segmentation-jobs", type=int, default=4, help="Parallel workers for segmentation.")
    seg.add_argument("--skip-segmentation", action="store_true", help="Force-skip stage 1 even if --raw-hdr-dir is given.")

    # ── Stage 2: cleaning (TODO 5.5, by-size exclusion) ──
    clean = p.add_argument_group("2. data cleaning (optional)")
    clean.add_argument("--clean-data", action="store_true",
                        help="Physically move outlier-area grain crops from *_processed/ to *_excluded/ before training "
                             "(one-time; see rgb_grains/data/cleaning.py). Independent of --skip-cleaning.")
    clean.add_argument("--skip-cleaning", action="store_true", help="Force-skip stage 2 even if --clean-data/--od-exclude are given.")
    clean.add_argument("--min-grain-area", type=int, default=None, help="Override the per-dataset default minimum active-pixel area.")
    clean.add_argument("--max-grain-area", type=int, default=None, help="Optional maximum active-pixel area.")
    clean.add_argument("--clean-dry-run", action="store_true", help="Report what --clean-data/--od-exclude would move, without moving anything.")
    clean.add_argument("--od-exclude", action="store_true",
                        help="TODO 5.5 'By OD': also move IsolationForest-flagged outlier grains (no labels used) "
                             "from *_processed/ to *_excluded/. Independent of --clean-data; runs after it if both are given.")
    clean.add_argument("--od-contamination", type=float, default=0.02,
                        help="Expected fraction of outliers per *_processed folder for --od-exclude (default: 0.02).")

    # ── Stage 3: train + validate ──
    train = p.add_argument_group("3. train + validate")
    train.add_argument("--config", type=str, default="configs/default.json", help="Path to a training config JSON (see configs/).")
    train.add_argument("--data-dir", type=str, default="data", help="Directory containing the *_processed dataset folders.")
    train.add_argument("--base-dir", type=str, default="./", help="Where expe/, models/, overall_perf_summary.json are written.")
    train.add_argument("--splitting-choice", type=str, default="muPlot-3muTrain-1muTest")
    train.add_argument("--dataset-choice", type=str, default="perfomix", choices=["perfomix", "SCOOP"])
    train.add_argument("--yearChosen", type=int, default=2020)
    train.add_argument("--restrict-classes", type=int, default=0)
    train.add_argument("--folds", type=int, nargs="+", default=[0], help="One training+validation run per fold listed here.")
    train.add_argument("--tag", type=str, default="pipeline")
    train.add_argument("--debugMode", type=int, default=0, help="Limit samples per class for a fast smoke run (0=off).")
    train.add_argument("--pretrained", type=int, default=1)
    train.add_argument("--backbone-impl", type=str, default="custom", choices=["custom", "torchvision"],
                        help="'custom' (default, no torchvision dep) or 'torchvision' for A/B-testing against "
                             "torchvision.models.convnext_tiny; see rgb_grains.train --help.")
    train.add_argument("--downsample-kernel", type=int, default=1, help="TODO 5.3 resolution ablation kernel (1=off).")
    train.add_argument("--downsample-mode", type=str, default="mean", choices=["mean", "max"])
    train.add_argument("--run-frozen-features", action="store_true",
                        help="Also run the frozen-feature (32-view) + logistic-regression C grid search, "
                             "reporting the held-out test result (see rgb_grains.models.classifier_head).")
    train.add_argument("--frozen-feature-c-grid", type=float, nargs="+",
                        default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                        help="C values for the frozen-feature logistic-regression grid search.")
    train.add_argument("--extra-train-args", type=str, nargs="*", default=[],
                        help="Additional raw CLI args forwarded verbatim to rgb_grains.train.")
    train.add_argument("--skip-training", action="store_true",
                        help="Force-skip stage 3, e.g. to run only segmentation and/or cleaning (such as --clean-dry-run).")
    return p


def _run_segmentation(args) -> None:
    if args.skip_segmentation or args.raw_hdr_dir is None:
        print("[pipeline] Stage 1/3 (segmentation): skipped.")
        return
    if args.dataset_name is None:
        raise SystemExit("--dataset-name is required when --raw-hdr-dir is given (used to pick segmentation parameters).")
    print(f"[pipeline] Stage 1/3 (segmentation): processing {args.raw_hdr_dir} ...")
    seg_script = Path(__file__).parent / "segmentation" / "script_hyspex_to_rgb.py"
    out_dir = Path(args.data_dir) / f"{args.dataset_name}_processed"
    cmd = [
        sys.executable, str(seg_script),
        "--input", args.raw_hdr_dir,
        "--output", str(out_dir),
        "--jobs", str(args.segmentation_jobs),
        "--dataset-name", args.dataset_name,
    ]
    print("[pipeline]  ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _run_cleaning(args) -> None:
    if args.skip_cleaning or not (args.clean_data or args.od_exclude):
        print("[pipeline] Stage 2/3 (cleaning): skipped (pass --clean-data and/or --od-exclude to enable).")
        return

    if args.clean_data:
        from rgb_grains.data.cleaning import move_excluded_files

        print(f"[pipeline] Stage 2/3 (cleaning, by-size): scanning {args.data_dir} for {args.dataset_choice} data "
              f"(dry_run={args.clean_dry_run}) ...")
        n_moved = move_excluded_files(
            data_dir=args.data_dir,
            dataset_choice=args.dataset_choice,
            min_area=args.min_grain_area,
            max_area=args.max_grain_area,
            dry_run=args.clean_dry_run,
        )
        print(f"[pipeline]   {n_moved} grain crop(s) {'would be' if args.clean_dry_run else 'were'} excluded.")

    if args.od_exclude:
        from rgb_grains.data.cleaning import move_outliers_od

        print(f"[pipeline] Stage 2/3 (cleaning, by-OD): scanning {args.data_dir} for {args.dataset_choice} data "
              f"(contamination={args.od_contamination}, dry_run={args.clean_dry_run}) ...")
        n_moved = move_outliers_od(
            data_dir=args.data_dir,
            dataset_choice=args.dataset_choice,
            contamination=args.od_contamination,
            dry_run=args.clean_dry_run,
        )
        print(f"[pipeline]   {n_moved} grain crop(s) {'would be' if args.clean_dry_run else 'were'} excluded.")


def _run_training(args):
    if args.skip_training:
        print("[pipeline] Stage 3/3 (train+validate): skipped.")
        return []

    from rgb_grains import train as train_module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[pipeline] Stage 3/3 (train+validate): device={device} "
          f"({torch.cuda.get_device_name(0) if device == 'cuda' else 'no GPU visible'})")

    results = []
    for fold in args.folds:
        train_argv = [
            "--config", args.config,
            "--data-dir", args.data_dir,
            "--base-dir", args.base_dir,
            "--splitting-choice", args.splitting_choice,
            "--dataset-choice", args.dataset_choice,
            "--yearChosen", str(args.yearChosen),
            "--restrict-classes", str(args.restrict_classes),
            "--fold", str(fold),
            "--tag", args.tag,
            "--debugMode", str(args.debugMode),
            "--pretrained", str(args.pretrained),
            "--backbone-impl", args.backbone_impl,
            "--downsample-kernel", str(args.downsample_kernel),
            "--downsample-mode", args.downsample_mode,
            "--run-frozen-features", "1" if args.run_frozen_features else "0",
            "--frozen-feature-c-grid", *[str(c) for c in args.frozen_feature_c_grid],
            # the loader-side filter mirrors stage 2 so a single --clean-data
            # flag governs both the one-time move and any remaining in-loader filtering
            "--clean-data", "1" if args.clean_data else "0",
        ]
        if args.min_grain_area is not None:
            train_argv += ["--min-grain-area", str(args.min_grain_area)]
        if args.max_grain_area is not None:
            train_argv += ["--max-grain-area", str(args.max_grain_area)]
        train_argv += args.extra_train_args

        print(f"\n[pipeline] === fold {fold} ===")
        print("[pipeline]   rgb_grains.train " + " ".join(train_argv))
        perf_dict = train_module.main(train_argv)
        results.append(perf_dict)
    return results


def main(argv=None):
    args = build_parser().parse_args(argv)

    _run_segmentation(args)
    _run_cleaning(args)
    results = _run_training(args)

    if not results:
        return results

    print("\n[pipeline] Summary (test_bal_acc per fold):")
    for r in results:
        print(f"[pipeline]   fold {r['split_number']}: {r['test_bal_acc']:.4f}  ({r['experiment_short_name']})")
    if len(results) > 1:
        import numpy as np
        accs = [r["test_bal_acc"] for r in results]
        print(f"[pipeline]   mean +/- std: {np.nanmean(accs):.4f} +/- {np.nanstd(accs):.4f}")
    return results


if __name__ == "__main__":
    main()
