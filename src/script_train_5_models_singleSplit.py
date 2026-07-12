from __future__ import annotations
import os
import shutil
import math, os, time
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# import seaborn as sns
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

## home made modules:
from dataset import *
# from model_others import Model_EfficientNet
# from model_others import *
from model_ConvNeXt import *
from tools import *
from module_plots import *
from module_exactfit_classifierHead import *

if 'ipython' in globals():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

# limit_per_year = None  ## disabled feature, to be removed or restored at some point
# exist_ok = False
train_on_year_2 = False  ## to be cleaned up at some point

## DEBUG MODE:
# limit_per_year = 300  #DEBUG Set to e.g., 100 for fast debugging
# exist_ok = True
RESTRICTED_CLASSES = [5, 7]  ##TODO: repasser ce choix en convetion nouvelle, c.a.d. 4,6, je crois ?
RESTRICTED_CLASSES = [0,1,2,3,5,6,7]  ##TODO: repasser ce choix en convetion nouvelle, c.a.d. 4,6, je crois ?

# ── CONSTANTS ──
BASE_DIR = Path("./")
MODELS_DIR = BASE_DIR / "models"


def get_experiment_short_name(args, config, reload=False):
    """Define experiment short name combining args.expe folder name and fold number."""
    if reload:
        return config.get("expe", "expe")
    base_name = config.get("expe", "expe")
    #     testOnMixedOnly=args.testOnMixedOnly,
    # trainOnMixedAndPure=args.trainOnMixedAndPure,
    # equalize_classes=args.equalize_classes,
    # NsamplesYear2=args.NsamplesYear2,
    # testOnWholePureOnly=args.testOnWholePureOnly,
    # combineMixedAndPureInTest=args.combineMixedAndPureInTest,
    experiment_name =  base_name + f"_{args.splitting_choice}_year={args.yearChosen}_rstCls={args.restrict_classes}_tstMxNly={args.testOnMixedOnly}_eqCls={args.equalize_classes}_trainOnMixedAndPure={args.trainOnMixedAndPure}_ststPurNly={args.testOnWholePureOnly}_cmbMxPur={args.combineMixedAndPureInTest}"
    if args.NsamplesYear2!=0:
        experiment_name += f"_NsamplesYear2={args.NsamplesYear2}"
    experiment_name += f"_fold={args.fold}"
    return experiment_name


def setup_logging(OUTPUT_DIR):
    """Setup logging to file, similar to model_ConvNeXt.py"""
    log_path = OUTPUT_DIR / "training.log"
    log_file = open(log_path, "a")
    print(f"[*] Log file: {log_path}")

    def _log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    return _log, log_file


MODELS_DIR = BASE_DIR / "models"

# ── CONSTANTS ──
BASE_DIR = Path("./")
RGB_DIR = BASE_DIR / "data"
CH_SCALE = [1567.0, 8316.0, 18126.0]  # spectral bands [22, 53, 89]
IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]
print(f"Base Directory: {BASE_DIR}")
print(f"Dataset Directory: {RGB_DIR}")

### ARGUMENTS ####
argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--config", type=str, default="./config.json", help="Path to config file"
)
argparser.add_argument(
    "--npz_path", type=str, default=None, help="Path to npz file to LOAD predictions from (and thus, not load any model)"
)
argparser.add_argument("--reload", type=int, default=0, help="wether to re-load model")
argparser.add_argument(
    "--reload_path", type=str, default=None, help="location of experiment"
)
argparser.add_argument(
    "--modelPath",
    type=str,
    default=None, # models/ConvNeXt-Tiny_Y1_model_debug_muPlot_year1only_year=2020_restrClass=1.pth
    help="Path to the model to be pre-loaded"
)
argparser.add_argument(
    "--debugMode",
    type=int,
    default=0,
    help="Debug mode (0/1)",
)
argparser.add_argument(
    "--test_ratio",
    type=float,
    default=0.5,
    help="Test ratio, for the case where we split at random (0-1)",
)

argparser.add_argument(
    "--restrict-classes",
    type=int,
    default=0,
    help=f"Restrict classes to a few classes, namely only classes: {RESTRICTED_CLASSES}",
)
argparser.add_argument(
    "--yearChosen",
    type=int,
    default=2020,
    help="Year to use for single year splits (2020/2021)",
)
argparser.add_argument(
    "--splitting-choice",
    type=str,
    default="muPlot-3muTrain-1muTest",
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
    ],
    help="Dataset splitting strategy",
)
argparser.add_argument(
    "--fold",
    type=int,
    default=0,
    help="Fold number for cross-validation (optional)",
)
argparser.add_argument(
    "--equalize-classes",
    type=int,
    default=1,
    help="Equalize classes",
)

argparser.add_argument(
    "--testOnMixedOnly",
    type=int,
    default=0,
    help="Test on mixed stands only",
)
argparser.add_argument(
    "--trainOnMixedAndPure",
    type=int,
    default=0,
    help="Test on a mix of: some of the pure stand data (fraction chosen according to splitting-choice) and all of the mixed data",
)
argparser.add_argument(
    "--testOnWholePureOnly",
    type=int,
    default=0,
    help="Test on pure stands only",
)
argparser.add_argument(
    "--combineMixedAndPureInTest",
    type=int,
    default=1,
    help="Combine mixed and pure data in test set",
)
argparser.add_argument(
    "--NsamplesYear2",
    type=int,
    default=0,
    help="Number of samples to use for year 2, when the choice of split is muPlotsTrain_year1_plusVariableNumberOn1muPlotYear2_1muPlotTestYear2",
)
argparser.add_argument(
    "--dataset-choice",
    type=str,
    default="perfomix",
    choices=["perfomix", "SCOOP"],
    help="5.2bis: 'perfomix' (8 varieties, pure+mixed) or 'SCOOP' (Martin's SCOOP/BACS data, "
         "4 varieties, pure only -- use with --splitting-choice muPlot_year1only and the bacs' "
         "collection year for --yearChosen).",
)
argparser.add_argument(
    "--run-frozen-features",
    type=int,
    default=0,
    help="5.1 kickstart: also extract frozen (32-view) features, save them to disk, and run a CV "
         "grid search over the logistic-regression C, reporting the held-out test result.",
)
argparser.add_argument(
    "--frozen-feature-c-grid",
    type=float,
    nargs="+",
    default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    help="C values for --run-frozen-features's grid search.",
)
args = argparser.parse_args()
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
    # limit_per_year = None
elif args.npz_path is not None:
    exist_ok = True
    limit_per_year=None
else:
    exist_ok = False
    limit_per_year = None
if args.NsamplesYear2 > 0 :
    limit_per_year=1450
## load config file + create experiment dir. + backup the config file there
with open(config_path, "r") as f:
    config = json.load(f)
# experiment_name = config.get("expe", "expe/")
if "year1only" not in args.splitting_choice and args.splitting_choice != "muPlots_mvblNY2":
    args.yearChosen = "all"
# experiment_name = (
#     experiment_name
#     + f"_{args.splitting_choice}_year={args.yearChosen}_restrClass={args.restrict_classes}"
# )

experiment_short_name = get_experiment_short_name(args, config, reload=reload)
config["expe"] = experiment_short_name
print(f"Experiment name: {experiment_short_name}")


num_classes = config.get("nc", 8)
if args.restrict_classes:
    RESTRICTED_CLASSES = RESTRICTED_CLASSES
else:
    RESTRICTED_CLASSES = np.arange(num_classes, dtype=int)

# Get output directory from config (default to "expe/")
OUTPUT_DIR = BASE_DIR / "expe" / experiment_short_name
print(f"Output directory for this experiment: {OUTPUT_DIR}")
if reload == False:
    os.makedirs(
        OUTPUT_DIR, exist_ok=exist_ok
    )  ## set to False to avoid overwritting ?? But, maybe not. For dev it's nice to overwrite
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Save config file to output directory, for reproducibility
    config_save_path = OUTPUT_DIR / "config.json"
    with open(config_save_path, "a") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_save_path}")

# Setup logging and get experiment short name
_log_fn, _log_file = setup_logging(OUTPUT_DIR)
_log_fn(f"Experiment short name: {experiment_short_name}")


model_name = "ConvNeXt-Tiny"
modelSingle = Model_ConvNeXt(config=config,restricted_classes=RESTRICTED_CLASSES)

_log_fn(f"args:{args}")
# print(f"args:{args}")

## defaults for the case args.npz_path re-loads logits without recomputing these (no train_data then):
train_bal_acc_32views = float("nan")
frozen_feature_result = None

if args.npz_path is None:

    #####################################
    ### Loading the train/test splits: ##
    # train_data, test_data, train_y2, test_y2 = load_datasets(dataset_path=str(RGB_DIR), limit_per_year=limit_per_year, test_ratio=0.2, base_dir=BASE_DIR)
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
        fold_number=fold_number,
        dataset_choice=args.dataset_choice,
        testOnMixedOnly=args.testOnMixedOnly,
        trainOnMixedAndPure=args.trainOnMixedAndPure,
        equalize_classes=args.equalize_classes,
        NsamplesYear2=args.NsamplesYear2,
        testOnWholePureOnly=args.testOnWholePureOnly,
        combineMixedAndPureInTest=args.combineMixedAndPureInTest,
    )
    true_y_test = np.asarray(test_data["y"]) ## one-hot or BOW vectors, of shape: (Ntest, nc)

    ## 5.1 (easy): display examples of inputs after data augmentation (5 grains x 10 rows: 1 unperturbed + 9 random augs)
    plot_augmentation_examples(train_data["X"], modelSingle.crop, OUTPUT_DIR, suffix=f"_{experiment_short_name}")

    if reload:
        _log_fn("\n\n## LOADING ALREADY FINE-TUNED MODEL\n\n")
        # for model_name in models_.keys():
        safe_name = model_name.replace("/", "-").replace(" ", "_")
        _log_fn(f"loading model {model_name} already fine-tuned")
        model_filename_best = f"{safe_name}_Y1_model_{experiment_short_name}_bestmodel.pth"
        # model_filename_net = f"{safe_name}_Y1_model_{experiment_short_name}_net.pth"
        model_path = MODELS_DIR / model_filename_best
        if args.modelPath is not None:
            model_path = Path(args.modelPath)
        if model_path.exists():
            if hasattr(modelSingle, "net"):
                modelSingle.net.load_state_dict(
                    torch.load(model_path, map_location="cpu")
                )
                modelSingle.net.eval()
            # if hasattr(modelSingle, "all_models"):
            #     modelSingle.all_models = [
            #         ("loaded", torch.load(model_path, map_location="cpu"))
            #     ]
            modelSingle.is_fitted = True


    elif reload == 0:  ## FINE-TUNING THE MODEL:
        safe_name = model_name.replace("/", "-").replace(" ", "_")
        _log_fn(f"\n{'=' * 70}")

        _log_fn(
            f"\n--- Training (fine-tuning) {model_name} on Year 1 (Source Domain) ---"
        )
        modelSingle.fit(train_data)

        # Export Weights to /models/ directory with experiment short name suffix
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_filename_best = f"{safe_name}_Y1_model_{experiment_short_name}_bestmodel.pth"
        model_filename_net = f"{safe_name}_Y1_model_{experiment_short_name}_net.pth"
        if (
            hasattr(modelSingle, "all_models")
            and len(modelSingle.all_models) > 0
        ):
            _log_fn("saving model.all_models parameters")
            _, lw = modelSingle.all_models[-1]
            torch.save(lw, str(MODELS_DIR / model_filename_best))
        elif hasattr(modelSingle, "net"):
            _log_fn("saving model.net parameters")
            torch.save(modelSingle.net.cpu().state_dict(), str(MODELS_DIR / model_filename_net))
        _log_fn(f"Model weights saved to {MODELS_DIR}.")

        ### END OF MODEL FINE TUNING
        #############################

    ## 5.1 (relatively easy): train balanced accuracy on the best model, using minimal
    ## augmentations (32-view TTA) only, to assess overfitting (should be >= val bal acc, hopefully not by much).
    _log_fn("\n[*] Computing train balanced accuracy (32-view TTA) on the best model...")
    logits_train_data = modelSingle.predict_logits(train_data)
    train_bal_acc_32views, train_precision_32views, train_recall_32views = val_bal_acc_per_class_offline(
        logits_train_data.argmax(1), np.asarray(train_data["y"]), num_classes, strict=False
    )
    _log_fn(f"Train balanced accuracy (32-view TTA): {train_bal_acc_32views:.4f}")

    ## 5.1 "kickstart loop" recycling: extract frozen (32-view) features on pure-stand
    ## samples only (logistic regression doesn't handle partial/mixed labels), save them
    ## to disk, CV grid search over C, report the held-out test set's result.
    if args.run_frozen_features:
        _log_fn("\n[*] 5.1 kickstart: extracting frozen features (32 views) for logistic-regression CV grid search...")
        train_pure_mask = (np.asarray(train_data["y"]) > 0).sum(axis=1) == 1
        test_pure_mask = (np.asarray(test_data["y"]) > 0).sum(axis=1) == 1
        train_pure = {"X": train_data["X"][train_pure_mask], "y": train_data["y"][train_pure_mask].argmax(1)}
        test_pure = {"X": test_data["X"][test_pure_mask], "y": test_data["y"][test_pure_mask].argmax(1)}
        X_train_lr, y_train_lr_aug, X_test_lr, y_test_lr, X_test_lr_aug = extract_features_from_dataset(
            modelSingle.net, train_pure, test_pure
        )
        frozen_features_path = OUTPUT_DIR / "frozen_features_32views.npz"
        save_frozen_features(frozen_features_path, X_train_lr, y_train_lr_aug, X_test_lr, y_test_lr, X_test_lr_aug)
        frozen_feature_result = frozen_feature_gridsearch_C(
            X_train_lr, y_train_lr_aug, X_test_lr, y_test_lr, c_grid=args.frozen_feature_c_grid
        )
        _log_fn(
            f"[*] Frozen-feature result: best_C={frozen_feature_result['best_C']} "
            f"test_bal_acc={frozen_feature_result['test_bal_acc']:.4f}"
        )

safe_name = model_name.replace("/", "-").replace(" ", "_")
if args.npz_path is None:
    ## savving predictions that have just been computed
    npz_path = OUTPUT_DIR / f"{safe_name}_predictions_{experiment_short_name}.npz"
else:
    ## loading predictions that are already sitting on the disk
    npz_path = Path(args.npz_path)
    OUTPUT_DIR = npz_path.parent

if npz_path.exists():  ## reloading predictions from saved file
    _log_fn(f"[*] Loading predictions from {npz_path}")
    pred_flow = np.load(npz_path, allow_pickle=True)
    # NpzFile 'expe/try2/ConvNeXt-Tiny_Y2=False_predictions.npz' with keys: logits_test_data, true_y_test, logits_y1_on_y2, true_y2_test
    logits_test_data = pred_flow["logits_test_data"]
    true_y_test = pred_flow["true_y_test"]
    ids = pred_flow["ids"]
    y_pred_test = logits_test_data.argmax(1)

else:  ## computing predictions using the models
    _log_fn(f"[*] Computing predictions for {model_name}")
    _log_fn(f"[{model_name}] Testing Model (by default, In-Domain, Upper Bound)...")
    
    logits_test_data = modelSingle.predict_logits(test_data)

    y_pred_test = logits_test_data.argmax(1)

    np.savez(npz_path,
        logits_test_data=logits_test_data,
        true_y_test=true_y_test,
        ids=test_data['ids'], ## ideally we can do without the ids.
    )
    _log_fn(f"Predictions saved to: {npz_path}")



## divide the score measures in 2: the pure data, first, then the mixed data:
keep_pure  = (((true_y_test>0).sum(axis=1))==1)
keep_mixed = (((true_y_test>0).sum(axis=1))>1) ## indices of the mixed ones
assert (keep_pure.sum() + keep_mixed.sum()) == len(true_y_test), "Inconsistent split of pure and mixed data"

y_true = true_y_test[keep_pure].argmax(1)

## pure stand data:
if keep_pure.sum() > 0:
    _log_fn(f"\n[*] Evaluating on pure stand data")
    test_acc = accuracy_score(y_true, y_pred_test[keep_pure])
    test_bal_acc = balanced_accuracy_score(y_true, y_pred_test[keep_pure])
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_test[keep_pure], average=None, zero_division=0 )
    _log_fn(f"\n--- Balanced Accuracy for {model_name} ---")
    _log_fn(f"Test (un-balanced) acc (Native, pure samples): {test_acc:.4f}")
    _log_fn(f"Test (balanced) acc (Native, pure samples): {test_bal_acc:.4f}")
    plot_cm(
        y_true , # true_y_test.argmax(1),
        y_pred_test[keep_pure], # y_pred_test, # preds_dict["standard"],
        f"{model_name}\nTest Y1 (Native)\n{experiment_short_name}",
        f"Y1-Y1_{experiment_short_name}",
        OUTPUT_DIR,
    )

    ## 5.1 (easy): display examples of failed predictions (imshow the original grains).
    ## Only possible when test_data (raw npz images) was actually loaded this run, i.e.
    ## not when --npz_path re-loads already-computed logits without the underlying images.
    if args.npz_path is None:
        plot_failed_predictions(
            test_data["X"][keep_pure], y_true, y_pred_test[keep_pure],
            OUTPUT_DIR, suffix=f"_{experiment_short_name}",
        )

    # same logic as for the mixed, fbut for the pure data, using soft logits:
    test_bal_acc_pure, precision_per_class_pure, recall_per_class_pure = val_bal_acc_per_class_offline(y_pred_test[keep_pure], true_y_test[keep_pure], num_classes, strict=False)
    _log_fn(f"Balanced accuracy (pure, home-made): {test_bal_acc_pure:.4f}") ## there can be some nasty zeros lowering this one.
    _log_fn(f"Per-class precision (pure): {precision_per_class_pure}")
    _log_fn(f"Per-class recall (pure): {recall_per_class_pure}")
    assert (precision_per_class_pure-precision).sum() == 0, "Inconsistent precision values"
    assert (recall_per_class_pure-recall).sum() == 0, "Inconsistent recall values"

    cm_pure = soft_confusion_matrix(y_pred_test[keep_pure], true_y_test[keep_pure])
    plot_soft_confusion_matrix(cm_pure,       f"Y1-Y1_{experiment_short_name}_softCM_pure",   OUTPUT_DIR, title="pure samples: soft conf. mat. (unif PLL prior)")
    ## not very intersting, in the end, but can be computed: soft recall, precision, f1:
    # precision_pure_soft, recall_pure_soft, f1_pure_soft  = soft_class_metrics(logits_test_data[keep_pure], true_y_test[keep_pure])
    # _log_fn(f"Soft precision (pure): {precision_pure_soft}")
    # _log_fn(f"Soft recall (pure): {recall_pure_soft}")
    # _log_fn(f"Soft f1 (pure): {f1_pure_soft}")

else:
    test_acc=np.nan
    test_bal_acc=np.nan
    precision, recall, f1 = np.array([np.nan, np.nan, np.nan]),  np.array([np.nan, np.nan, np.nan]),  np.array([np.nan, np.nan, np.nan])
    print("No pure data in test set")

## mixed stand data:
if keep_mixed.sum() > 0:
    _log_fn(f"\n[*] Evaluating on mixed stand data")
    test_bal_acc_mixed, precision_per_class_mixed, recall_per_class_mixed = val_bal_acc_per_class_offline(y_pred_test[keep_mixed], true_y_test[keep_mixed], num_classes, strict=False)
    _log_fn(f"Balanced accuracy (mixed, home-made): {test_bal_acc_mixed:.4f}")
    _log_fn(f"Per-class precision (mixed): {precision_per_class_mixed}")
    _log_fn(f"Per-class recall (mixed): {recall_per_class_mixed}")
    cm_mixed = soft_confusion_matrix(y_pred_test[keep_mixed], true_y_test[keep_mixed])
    plot_soft_confusion_matrix(cm_mixed,       f"Y1-Y1_{experiment_short_name}_softCM_mixed",   OUTPUT_DIR, title="Mixed samples: soft conf. mat. (unif PLL prior)")
    ## not very intersting, in the end, but can be computed: soft recall, precision, f1:
    # precision_mixed_soft, recall_mixed_soft, f1_mixed_soft  = soft_class_metrics(logits_test_data[keep_mixed], true_y_test[keep_mixed])
    # _log_fn(f"Soft precision (mixed): {precision_mixed_soft}")
    # _log_fn(f"Soft recall (mixed): {recall_mixed_soft}")
    # _log_fn(f"Soft f1 (mixed): {f1_mixed_soft}")
else:
    test_bal_acc_mixed=np.nan
    recall_per_class_mixed, precision_per_class_mixed = np.array([np.nan, np.nan, np.nan]), np.array([np.nan, np.nan, np.nan])
    print("No mixed data in test set")



_log_fn(f"\n[*] Global metrics: saving")
split_number = args.fold
num_epochs = config.get("max_ep", 20)
year = args.yearChosen
num_classes = config.get("nc", 8)
restricted_classes = RESTRICTED_CLASSES if args.restrict_classes else None
## TODO [optionnal]: add the other toggles to the perf dict entries.
# try: 
perf_dict = {
    "experiment_short_name": experiment_short_name,
    "split_number": split_number,
    "num_epochs": num_epochs,
    "limit_per_year": limit_per_year,
    "year": str(year),
    "num_classes": num_classes,
    "train_acc": float(
        modelSingle.train_acc[-1]
        if modelSingle.is_fitted
        and hasattr(modelSingle, "train_acc")
        and len(modelSingle.train_acc) > 0
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
    ## 5.1: train balanced accuracy (32-view TTA, best model) -- to assess overfitting.
    "train_bal_acc_32views": float(train_bal_acc_32views),
    ## 5.1 kickstart: best C + held-out test bal acc from the frozen-feature logistic regression, if run.
    "frozen_feature_best_C": float(frozen_feature_result["best_C"]) if frozen_feature_result else None,
    "frozen_feature_test_bal_acc": float(frozen_feature_result["test_bal_acc"]) if frozen_feature_result else None,
}

# _log_fn(f"Performance dict: {perf_dict}")
# ## nice display print for a dict:
# import pprint 
# pp = pprint.PrettyPrinter(indent=4)
# pp.pprint(perf_dict)

summary_json_path = BASE_DIR / "overall_perf_summary.json"
with open(summary_json_path, "a") as f:
    f.write((f"{perf_dict}\n".replace("'", '"')).replace("None", "null" )) ## 1 line per dict, to allow easy read and parsing
_log_fn(f"Performance summary saved to: {summary_json_path}")




## TODO: revive this, with the the new one-hot thing(simplest is to just never use the mixed data in this):
calibration_plot(logits_test_data, true_y_test, OUTPUT_DIR, experiment_short_name)



# _log_fn("transfer learning")
# model  = models_['ConvNeXt-Tiny'].net
# model = modelSingle.net
# X_train_lr, y_train_lr_aug , X_test_lr, y_test_lr, X_test_lr_aug \
#     = extract_features_from_dataset(model, train_y2, test_y2)
# Ncaps, acc_lr_train_list, acc_lr_test_list, acc_lr_test_views_list \
#     = learning_curve_exactFit(X_train_lr, y_train_lr_aug , X_test_lr, y_test_lr, X_test_lr_aug, OUTPUT_DIR)
# flow = np.load("expe/transferlearning10/transfer_learning.npz")
# Ncaps = flow['Ncaps']
# acc_lr_train_list = flow['acc_lr_train_list']
# acc_lr_test_list = flow['acc_lr_test_list']
# acc_lr_test_views_list = flow['acc_lr_test_views_list']
# plot_transfer_learning_learningCurve(Ncaps, acc_lr_train_list, acc_lr_test_list, acc_lr_test_views_list, OUTPUT_DIR)

if config and reload == False:
    config_save_path = OUTPUT_DIR / "config_end_check.json"
    with open(config_save_path, "a") as f:
        json.dump(config, f, indent=2)
    _log_fn(f"Config saved to: {config_save_path}")
