"""Fine-tune (or evaluate) the ConvNeXt-Tiny grain classifier on one train/test split.

This is the historical ``script_train_5_models_singleSplit.py`` entrypoint,
now wrapped in ``build_parser()`` / ``main()`` so it can be imported (e.g. by
``rgb_grains.pipeline``) as well as run as a script:

    python -m rgb_grains.train --config configs/debug.json --debugMode 100 \\
        --splitting-choice random_year1only --yearChosen 2020 --tag smoke
"""
from __future__ import annotations
import os
import math, os, time
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json
import argparse

from rgb_grains.data.dataset import *
from rgb_grains.models.convnext import *
from rgb_grains.utils.tools import *
from rgb_grains.viz.plots import *
from rgb_grains.models.classifier_head import *

DEFAULT_RESTRICTED_CLASSES = [0, 1, 2, 3, 5, 6, 7]


def get_experiment_short_name(args, config, reload=False):
    """Define experiment short name combining args.expe folder name and fold number."""
    if reload:
        return config.get("expe", "expe")
    base_name = config.get("expe", "expe")
    tag = args.tag
    if tag != "":
        base_name = tag + "_" + base_name
    experiment_name = base_name + f"_{args.splitting_choice}"
    if args.dataset_choice != "perfomix":
        experiment_name += f"_{args.dataset_choice}"
    experiment_name += f"_cmbMxPur={args.combineMixedAndPureInTest}"
    if args.yearChosen != "all":
        experiment_name += f"_year={args.yearChosen}"
    if args.restrict_classes != 0:
        experiment_name += f"_rstCls={args.restrict_classes}"
    if args.testOnMixedOnly != 0:
        experiment_name += f"_tstMxNly={args.testOnMixedOnly}"
    if args.equalize_classes != 0:
        experiment_name += f"_eqCls={args.equalize_classes}"
    if args.trainOnMixedAndPure != 0:
        experiment_name += f"_trainOnMixedAndPure={args.trainOnMixedAndPure}"
    if args.testOnWholePureOnly != 0:
        experiment_name += f"_ststPurNly={args.testOnWholePureOnly}"
    if args.kickstart != 0:
        experiment_name += f"_kickstart={args.kickstart}"
    if args.pretrained == 0:
        experiment_name += "_pretrained=0"
    if args.NsamplesYear2 != 0:
        experiment_name += f"_NsamplesYear2={args.NsamplesYear2}"
    if args.downsample_kernel != 1:
        experiment_name += f"_downsample={args.downsample_kernel}{args.downsample_mode[0]}"
    if args.clean_data:
        experiment_name += "_cleaned=1"
    experiment_name += f"_fold={args.fold}"
    return experiment_name


def setup_logging(OUTPUT_DIR):
    """Setup logging to file, similar to convnext.py"""
    log_path = OUTPUT_DIR / "training.log"
    log_file = open(log_path, "a", encoding="utf-8")
    print(f"[*] Log file: {log_path}")

    def _log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    return _log, log_file


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--tag", type=str, default="", help="tag to add to experiment name (prefix), to avoid duplicate names of folders"
    )
    p.add_argument("--config", type=str, default="configs/default.json", help="Path to config file")
    p.add_argument(
        "--npz_path", type=str, default=None, help="Path to npz file to LOAD predictions from (and thus, not load any model)"
    )
    p.add_argument("--reload", type=int, default=0, help="wether to re-load model")
    p.add_argument("--reload_path", type=str, default=None, help="location of experiment")
    p.add_argument("--modelPath", type=str, default=None, help="Path to the model to be pre-loaded")
    p.add_argument("--debugMode", type=int, default=0, help="Debug mode: limit samples per class for fast iteration (0=off)")
    p.add_argument("--test_ratio", type=float, default=0.5, help="Test ratio, for the case where we split at random (0-1)")
    p.add_argument(
        "--restrict-classes", type=int, default=0,
        help=f"Restrict classes to a few classes, namely only classes: {DEFAULT_RESTRICTED_CLASSES}",
    )
    p.add_argument("--yearChosen", type=int, default=2020, help="Year to use for single year splits (2020/2021)")
    p.add_argument(
        "--splitting-choice", type=str, default="muPlot-3muTrain-1muTest",
        choices=[
            "muPlot_year1only",
            "muPlot-2muTrain-2muTest",
            "muPlot-2muTrain-2muTest-yearGeneralization_year1only",
            "muPlot-3muTrain-1muTest",
            "muPlot-1muTrain-1muTest-crossYears",
            "random_1muPlot",
            "random_3muPlot",
            "random_year1only",
            "random_4muPlot",
            "inferenceMode_4muTrain-0muTest",
            "trainOnMixedOnly",
            "muPlots_mvblNY2",
            "bacs_2train_1test",
        ],
        help="Dataset splitting strategy",
    )
    p.add_argument(
        "--dataset-choice", type=str, default="perfomix", choices=["perfomix", "SCOOP"],
        help="Dataset to use: perfomix or SCOOP/BACS.",
    )
    p.add_argument("--fold", type=int, default=0, help="Fold number for cross-validation (optional)")
    p.add_argument("--equalize-classes", type=int, default=0, help="Equalize classes. 0 is better.")
    p.add_argument("--testOnMixedOnly", type=int, default=0, help="Test on mixed stands only")
    p.add_argument(
        "--trainOnMixedAndPure", type=int, default=0,
        help="Test on a mix of: some of the pure stand data (fraction chosen according to splitting-choice) and all of the mixed data",
    )
    p.add_argument("--testOnWholePureOnly", type=int, default=0, help="Test on pure stands only")
    p.add_argument("--combineMixedAndPureInTest", type=int, default=1, help="Combine mixed and pure data in test set")
    p.add_argument(
        "--NsamplesYear2", type=int, default=0,
        help="Number of samples to use for year 2, when the choice of split is muPlots_mvblNY2",
    )
    p.add_argument(
        "--kickstart", type=int, default=0,
        help="LP-FT kickstart: 1=enable, 0=disable. When enabled, before fine-tuning, "
             "we extract 32-view features from the pretrained backbone over the train "
             "set, fit a logistic regression on them, and use the resulting (W_raw, "
             "b_raw) as the initial weights of the model's classifier head.",
    )
    p.add_argument(
        "--pretrained", type=int, default=1,
        help="Use ImageNet pretrained ConvNeXt-Tiny backbone weights (1) or train from random initialization (0).",
    )
    p.add_argument(
        "--kickstart_path", type=str, default=None,
        help="Optional path to an .npz file with keys W_raw, b_raw. If provided and "
             "the file exists, kickstart weights are loaded from it. If provided and "
             "the file does not exist, weights are computed and saved there. If not "
             "provided, weights are computed live and not cached.",
    )
    p.add_argument("--kickstart_recompute", type=int, default=0, help="If 1, ignore the cache at --kickstart_path and recompute.")
    p.add_argument("--run-frozen-features", type=int, default=0, help="Run the additional frozen-feature logistic-regression learning curve.")
    p.add_argument(
        "--frozen-feature-c-grid", type=float, nargs="+",
        default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        help="C values for frozen-feature logistic-regression grid search.",
    )
    # TODO 5.3: resolution ablation
    p.add_argument(
        "--downsample-kernel", type=int, default=1,
        help="TODO 5.3: pool crops by this kernel size (mean/max) then upsample back "
             "to crop size, to test how much accuracy depends on input resolution. "
             "1 = no-op (full resolution).",
    )
    p.add_argument("--downsample-mode", type=str, default="mean", choices=["mean", "max"], help="Pooling used by --downsample-kernel.")
    # TODO 5.5: data cleaning
    p.add_argument("--clean-data", type=int, default=0, help="TODO 5.5: drop grain crops whose active area is outside [--min-grain-area, --max-grain-area] before splitting.")
    p.add_argument("--min-grain-area", type=int, default=None, help="Override the per-dataset default minimum active-pixel area (see rgb_grains/data/cleaning.py).")
    p.add_argument("--max-grain-area", type=int, default=None, help="Optional maximum active-pixel area.")
    # Paths (overridable so rgb_grains.pipeline / cluster jobs can redirect output)
    p.add_argument("--base-dir", type=str, default="./", help="Base directory for expe/, models/, and overall_perf_summary.json")
    p.add_argument("--data-dir", type=str, default=None, help="Directory containing the *_processed dataset folders (default: <base-dir>/data)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    BASE_DIR = Path(args.base_dir)
    MODELS_DIR = BASE_DIR / "models"
    RGB_DIR = Path(args.data_dir) if args.data_dir is not None else BASE_DIR / "data"
    print(f"Base Directory: {BASE_DIR}")
    print(f"Dataset Directory: {RGB_DIR}")

    n_views = 32  # for the kickstarting only. In the end, it's not helping.

    reload = args.reload
    reload_path = args.reload_path
    if reload_path:
        config_path = Path(reload_path) / "config.json"
    else:
        config_path = Path(args.config)
    fold_number = args.fold
    if args.debugMode:
        print(f"[*] Debug mode: {args.debugMode}")
        exist_ok = True
        limit_per_year = args.debugMode
    elif args.npz_path is not None:
        exist_ok = True
        limit_per_year = None
    else:
        exist_ok = False
        limit_per_year = None
    if args.NsamplesYear2 > 0:
        limit_per_year = 1450

    ## load config file + create experiment dir. + backup the config file there
    with open(config_path, "r") as f:
        config = json.load(f)
    if args.dataset_choice == "SCOOP":
        config["nc"] = 4
    if (
        "year1only" not in args.splitting_choice
        and args.splitting_choice not in ["muPlots_mvblNY2", "bacs_2train_1test"]
    ):
        args.yearChosen = "all"

    experiment_short_name = get_experiment_short_name(args, config, reload=reload)
    config["expe"] = experiment_short_name
    config["pretrained"] = bool(args.pretrained)
    config["downsample_kernel"] = args.downsample_kernel
    config["downsample_mode"] = args.downsample_mode
    print(f"Experiment name: {experiment_short_name}")

    num_classes = config.get("nc", 8)
    if args.restrict_classes:
        RESTRICTED_CLASSES = DEFAULT_RESTRICTED_CLASSES
    else:
        RESTRICTED_CLASSES = np.arange(num_classes, dtype=int)

    OUTPUT_DIR = BASE_DIR / "expe" / experiment_short_name
    print(f"Output directory for this experiment: {OUTPUT_DIR}")
    if reload == False:
        os.makedirs(OUTPUT_DIR, exist_ok=exist_ok)
        os.makedirs(MODELS_DIR, exist_ok=True)
        config_save_path = OUTPUT_DIR / "config.json"
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to: {config_save_path}")

    _log_fn, _log_file = setup_logging(OUTPUT_DIR)
    _log_fn(f"Experiment short name: {experiment_short_name}")

    model_name = "ConvNeXt-Tiny"
    modelSingle = Model_ConvNeXt(config=config, restricted_classes=RESTRICTED_CLASSES, base_dir=str(BASE_DIR))

    _log_fn(f"args:{args}")

    if args.npz_path is None:
        train_data, test_data = load_datasets_microplot_split(
            dataset_path=RGB_DIR,
            _log_fn=_log_fn,
            limit_per_year=limit_per_year,
            test_ratio=args.test_ratio,
            output_dir=OUTPUT_DIR,
            restrict_classes=bool(args.restrict_classes),
            restricted_classes=RESTRICTED_CLASSES,
            yearChosen=args.yearChosen,
            splitting_choice=args.splitting_choice,
            NUM_CLASSES=num_classes,
            fold_number=fold_number,
            testOnMixedOnly=args.testOnMixedOnly,
            trainOnMixedAndPure=args.trainOnMixedAndPure,
            equalize_classes=args.equalize_classes,
            NsamplesYear2=args.NsamplesYear2,
            testOnWholePureOnly=args.testOnWholePureOnly,
            combineMixedAndPureInTest=args.combineMixedAndPureInTest,
            dataset_choice=args.dataset_choice,
            clean_data=bool(args.clean_data),
            min_grain_area=args.min_grain_area,
            max_grain_area=args.max_grain_area,
        )
        true_y_test = np.asarray(test_data["y"])  # one-hot or BOW vectors, of shape: (Ntest, nc)
        augmentation_plot_path = plot_augmentation_examples(
            train_data,
            OUTPUT_DIR,
            crop_size=config.get("crop", 176),
            seed=config.get("seed", 42),
        )
        _log_fn(f"Augmentation examples saved to: {augmentation_plot_path}")

        if reload:
            _log_fn("\n\n## LOADING ALREADY FINE-TUNED MODEL\n\n")
            safe_name = model_name.replace("/", "-").replace(" ", "_")
            _log_fn(f"loading model {model_name} already fine-tuned")
            model_filename_best = f"{safe_name}_Y1_model_{experiment_short_name}_bestmodel.pth"
            model_path = MODELS_DIR / model_filename_best
            if args.modelPath is not None:
                model_path = Path(args.modelPath)
            if model_path.exists():
                if hasattr(modelSingle, "net"):
                    modelSingle.set_selected_model(
                        "loaded_best_model",
                        torch.load(model_path, map_location="cpu"),
                    )
                    modelSingle.net.eval()
                    _log_fn(f"Selected model loaded from: {model_path}")
            else:
                raise FileNotFoundError(f"Model weights not found: {model_path}")

        elif reload == 0:  ## FINE-TUNING THE MODEL:
            safe_name = model_name.replace("/", "-").replace(" ", "_")
            _log_fn(f"\n{'=' * 70}")

            if bool(args.kickstart):
                _log_fn("\n--- LP-FT kickstart: computing/loading head weights ---")
                modelSingle.compute_kickstart_head(
                    train_data,
                    cache_path=args.kickstart_path,
                    recompute=bool(args.kickstart_recompute),
                    n_views=n_views,
                )

            _log_fn(f"\n--- Training (fine-tuning) {model_name} on Year 1 (Source Domain) ---")
            modelSingle.fit(train_data)

            os.makedirs(MODELS_DIR, exist_ok=True)
            model_filename_best = f"{safe_name}_Y1_model_{experiment_short_name}_bestmodel.pth"
            if modelSingle.selected_state_dict is None:
                raise RuntimeError("Training finished without a selected model to save")
            model_path = MODELS_DIR / model_filename_best
            torch.save(modelSingle.selected_state_dict, str(model_path))
            _log_fn(f"Selected model '{modelSingle.selected_model_name}' saved to {model_path}")

        train_bal_acc_32views, train_recall_per_class_32views, train_probabilities_32views = (
            modelSingle.evaluate_balanced_accuracy_32views(train_data, dataset_name="train")
        )
        modelSingle.train_bal_acc_32views = train_bal_acc_32views
        modelSingle.train_recall_per_class_32views = train_recall_per_class_32views
        if modelSingle.selected_val_bal_acc is not None:
            _log_fn(
                "Selected-model train-validation balanced accuracy gap: "
                f"{train_bal_acc_32views - modelSingle.selected_val_bal_acc:.4f}"
            )
        np.savez(
            OUTPUT_DIR / "train_selected_model_32view_predictions.npz",
            probabilities=train_probabilities_32views,
            true_y_train=np.asarray(train_data["y"]),
            ids=train_data["ids"],
        )
        _log_fn(
            "Selected-model train 32-view predictions saved to: "
            f"{OUTPUT_DIR / 'train_selected_model_32view_predictions.npz'}"
        )

    safe_name = model_name.replace("/", "-").replace(" ", "_")
    if args.npz_path is None:
        npz_path = OUTPUT_DIR / "predictions.npz"
    else:
        npz_path = Path(args.npz_path)
        OUTPUT_DIR = npz_path.parent

    if args.npz_path is not None:
        if not npz_path.exists():
            raise FileNotFoundError(f"Prediction archive not found: {npz_path}")
        _log_fn(f"[*] Loading predictions from {npz_path}")
        pred_flow = np.load(npz_path, allow_pickle=True)
        logits_test_data = pred_flow["logits_test_data"]
        true_y_test = pred_flow["true_y_test"]
        ids = pred_flow["ids"]
        y_pred_test = logits_test_data.argmax(1)
    else:
        _log_fn(f"[*] Computing predictions for {model_name}")
        _log_fn(f"[{model_name}] Testing Model (by default, In-Domain, Upper Bound)...")
        logits_test_data = modelSingle.predict_logits(test_data)
        y_pred_test = logits_test_data.argmax(1)
        np.savez(
            npz_path,
            logits_test_data=logits_test_data,
            true_y_test=true_y_test,
            ids=test_data["ids"],
        )
        _log_fn(f"Predictions saved to: {npz_path}")

    if args.npz_path is None:
        failed_plot_path = plot_failed_predictions(test_data, logits_test_data, OUTPUT_DIR)
        if failed_plot_path is None:
            _log_fn("No failed predictions to plot.")
        else:
            _log_fn(f"Failed prediction examples saved to: {failed_plot_path}")
    else:
        _log_fn("Skipping failed-prediction plot: original test images are unavailable.")

    ## divide the score measures in 2: the pure data, first, then the mixed data:
    keep_pure = (((true_y_test > 0).sum(axis=1)) == 1)
    keep_mixed = (((true_y_test > 0).sum(axis=1)) > 1)
    assert (keep_pure.sum() + keep_mixed.sum()) == len(true_y_test), "Inconsistent split of pure and mixed data"

    y_true = true_y_test[keep_pure].argmax(1)

    if keep_pure.sum() > 0:
        _log_fn("\n[*] Evaluating on pure stand data")
        test_acc = accuracy_score(y_true, y_pred_test[keep_pure])
        test_bal_acc = balanced_accuracy_score(y_true, y_pred_test[keep_pure])
        metric_labels = np.arange(num_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_test[keep_pure], labels=metric_labels, average=None, zero_division=0,
        )
        _log_fn(f"\n--- Balanced Accuracy for {model_name} ---")
        _log_fn(f"Test (un-balanced) acc (Native, pure samples): {test_acc:.4f}")
        _log_fn(f"Test (balanced) acc (Native, pure samples): {test_bal_acc:.4f}")
        plot_cm(
            y_true,
            y_pred_test[keep_pure],
            f"{model_name}\nTest Y1 (Native)\n{experiment_short_name}",
            "test_pure",
            OUTPUT_DIR,
            labels=metric_labels,
        )

        test_bal_acc_pure, precision_per_class_pure, recall_per_class_pure = val_bal_acc_per_class_offline(
            y_pred_test[keep_pure], true_y_test[keep_pure], num_classes, strict=False
        )
        _log_fn(f"Balanced accuracy (pure, home-made): {test_bal_acc_pure:.4f}")
        _log_fn(f"Per-class precision (pure): {precision_per_class_pure}")
        _log_fn(f"Per-class recall (pure): {recall_per_class_pure}")
        assert np.allclose(precision_per_class_pure, precision), "Inconsistent precision values"
        assert np.allclose(recall_per_class_pure, recall), "Inconsistent recall values"

        cm_pure = soft_confusion_matrix(y_pred_test[keep_pure], true_y_test[keep_pure])
        plot_soft_confusion_matrix(cm_pure, "softCM_pure", OUTPUT_DIR, title="pure samples: soft conf. mat. (unif PLL prior)")
    else:
        test_acc = np.nan
        test_bal_acc = np.nan
        precision, recall, f1 = np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        print("No pure data in test set")

    if keep_mixed.sum() > 0:
        _log_fn("\n[*] Evaluating on mixed stand data")
        test_bal_acc_mixed, precision_per_class_mixed, recall_per_class_mixed = val_bal_acc_per_class_offline(
            y_pred_test[keep_mixed], true_y_test[keep_mixed], num_classes, strict=False
        )
        _log_fn(f"Balanced accuracy (mixed, home-made): {test_bal_acc_mixed:.4f}")
        _log_fn(f"Per-class precision (mixed): {precision_per_class_mixed}")
        _log_fn(f"Per-class recall (mixed): {recall_per_class_mixed}")
        cm_mixed = soft_confusion_matrix(y_pred_test[keep_mixed], true_y_test[keep_mixed])
        plot_soft_confusion_matrix(cm_mixed, "softCM_mixed", OUTPUT_DIR, title="Mixed samples: soft conf. mat. (unif PLL prior)")
    else:
        test_bal_acc_mixed = np.nan
        recall_per_class_mixed, precision_per_class_mixed = np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
        print("No mixed data in test set")

    _log_fn("\n[*] Global metrics: collecting")
    split_number = args.fold
    num_epochs = config.get("max_ep", 20)
    year = args.yearChosen
    num_classes = config.get("nc", 8)
    restricted_classes = RESTRICTED_CLASSES if args.restrict_classes else None
    frozen_feature_result = None
    perf_dict = {
        "experiment_short_name": experiment_short_name,
        "split_number": split_number,
        "num_epochs": num_epochs,
        "limit_per_year": limit_per_year,
        "year": str(year),
        "num_classes": num_classes,
        "downsample_kernel": args.downsample_kernel,
        "downsample_mode": args.downsample_mode,
        "clean_data": bool(args.clean_data),
        "selected_model": modelSingle.selected_model_name,
        "selected_val_bal_acc": (
            float(modelSingle.selected_val_bal_acc) if modelSingle.selected_val_bal_acc is not None else None
        ),
        "train_bal_acc_32views": (
            float(modelSingle.train_bal_acc_32views) if modelSingle.train_bal_acc_32views is not None else None
        ),
        "train_recall_per_class_32views": (
            [float(r) for r in modelSingle.train_recall_per_class_32views]
            if modelSingle.train_recall_per_class_32views is not None
            else None
        ),
        "train_val_bal_acc_gap": (
            float(modelSingle.train_bal_acc_32views - modelSingle.selected_val_bal_acc)
            if modelSingle.train_bal_acc_32views is not None and modelSingle.selected_val_bal_acc is not None
            else None
        ),
        "train_acc": float(
            modelSingle.train_acc[-1]
            if modelSingle.is_fitted and hasattr(modelSingle, "train_acc") and len(modelSingle.train_acc) > 0
            else 0.0
        ),
        "test_acc": float(test_acc),
        "test_bal_acc": float(test_bal_acc),
        "per_class_precision": [float(p) for p in np.array(precision)],
        "per_class_recall": [float(r) for r in np.array(recall)],
        "test_bal_acc_mixed": float(test_bal_acc_mixed),
        "per_class_precision_mixed": [float(p) for p in np.array(precision_per_class_mixed)],
        "per_class_recall_mixed": [float(r) for r in np.array(recall_per_class_mixed)],
        "restricted_classes": restricted_classes,
    }

    calibration_plot(logits_test_data, true_y_test, OUTPUT_DIR, experiment_short_name)

    if bool(args.run_frozen_features):
        _log_fn("Running frozen-feature logistic-regression analysis")
        model = modelSingle.net
        X_train_lr, y_train_lr_aug, X_test_lr, y_test_lr, X_test_lr_aug = (
            extract_features_from_dataset(model, train_data, test_data)
        )
        frozen_features_path = OUTPUT_DIR / "frozen_features_32views.npz"
        np.savez_compressed(
            frozen_features_path,
            X_train_lr=X_train_lr,
            y_train_lr_aug=y_train_lr_aug,
            X_test_lr=X_test_lr,
            y_test_lr=y_test_lr,
            X_test_lr_aug=X_test_lr_aug,
            train_ids=train_data["ids"],
            test_ids=test_data["ids"],
        )
        _log_fn(f"Frozen features saved to: {frozen_features_path}")

        frozen_feature_result = frozen_feature_logreg_grid_search(
            X_train_lr, y_train_lr_aug, X_test_lr, y_test_lr, X_test_lr_aug, OUTPUT_DIR,
            c_grid=np.asarray(args.frozen_feature_c_grid, dtype=float),
        )
        _log_fn(
            "Frozen-feature LR grid search: "
            f"best_C={frozen_feature_result['best_C']} "
            f"test_bal={frozen_feature_result['test_bal_acc_avg_features']:.4f} "
            f"test_bal_viewavg={frozen_feature_result['test_bal_acc_view_average']:.4f}"
        )
        Ncaps, acc_lr_train_list, acc_lr_test_list, acc_lr_test_views_list, W_raw, b_raw = (
            learning_curve_exactFit(X_train_lr, y_train_lr_aug, X_test_lr, y_test_lr, X_test_lr_aug, OUTPUT_DIR)
        )
    else:
        _log_fn("Skipping frozen-feature experiment; enable with --run-frozen-features 1")

    if frozen_feature_result is not None:
        perf_dict.update(
            {
                "frozen_feature_best_C": float(frozen_feature_result["best_C"]),
                "frozen_feature_cv_mean_scores": [float(x) for x in frozen_feature_result["cv_mean_scores"]],
                "frozen_feature_cv_std_scores": [float(x) for x in frozen_feature_result["cv_std_scores"]],
                "frozen_feature_test_bal_acc": float(frozen_feature_result["test_bal_acc_avg_features"]),
                "frozen_feature_test_bal_acc_viewavg": float(frozen_feature_result["test_bal_acc_view_average"]),
                "frozen_feature_test_recall": [float(x) for x in frozen_feature_result["test_recall_avg_features"]],
            }
        )

    summary_json_path = BASE_DIR / "overall_perf_summary.json"
    with open(summary_json_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(perf_dict, allow_nan=True) + "\n")
    _log_fn(f"Performance summary saved to: {summary_json_path}")

    if config and reload == False:
        config_save_path = OUTPUT_DIR / "config_end_check.json"
        with open(config_save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        _log_fn(f"Config saved to: {config_save_path}")

    return perf_dict


if __name__ == "__main__":
    main()
