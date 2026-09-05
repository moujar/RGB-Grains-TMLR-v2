# Daryl Branch Work Summary

This branch contains engineering and experiment-support changes for the RGB grain classification pipeline. The work focuses on making training/evaluation runs more reproducible, adding requested diagnostic plots, and adding frozen-feature and scratch-training experiment modes.

See [COMMANDS.md](COMMANDS.md) for the maintained command reference, argument
descriptions, and PowerShell usage examples.

## Main Changes

### Model Selection And Persistence

- Cleaned up checkpoint selection between best validation model, SWA model, and last model.
- The selected model is now explicit:
  - Best validation balanced-accuracy checkpoint is used by default.
  - SWA is selected only if it improves validation balanced accuracy.
- The same selected checkpoint is used for inference, saving, and reloading.
- Experiment summaries now record:
  - `selected_model`
  - `selected_val_bal_acc`

### Diagnostics And Plots

- Added `augmentation_examples.jpg/pdf`.
  - Shows original center crops and random training augmentations.
  - Useful for visually checking the augmentation pipeline.

- Added `failed_predictions.jpg/pdf`.
  - Shows the most confident incorrect predictions using original test grains.
  - Handles partial/mixed labels by treating a prediction as correct if it is in the allowed label set.

- Improved confusion matrices.
  - Standard and soft confusion matrices now include:
    - Recall as a right-side column.
    - Precision as a bottom row.
  - Soft confusion matrices now handle empty rows without runtime warnings.

- Shortened generated artifact names for Windows path safety.
  - The experiment folder already stores the long experiment name, so files inside the folder now use shorter names such as `predictions.npz`, `fine_tuning_monitoring.jpg`, and `calibration_reliability.png`.

### Training Metrics

- Added selected-model training-set balanced accuracy using the deterministic 32-view inference protocol.
- Saved train-set 32-view probabilities to:

```text
train_selected_model_32view_predictions.npz
```

- Experiment summaries now include:
  - `train_bal_acc_32views`
  - `train_recall_per_class_32views`
  - `train_val_bal_acc_gap`

### Frozen-Feature Logistic Regression

- Made the frozen-feature analysis optional:

```powershell
--run-frozen-features 1
```

- Added 32-view frozen ConvNeXt feature extraction and saving:

```text
frozen_features_32views.npz
```

- Added logistic-regression `C` grid search using grouped cross-validation.
  - All 32 augmented views of the same grain stay in the same fold.
  - This avoids leakage between train and validation folds.

- Grid-search results are saved to:

```text
frozen_feature_logreg_grid_search.npz
```

- The grid can be configured with:

```powershell
--frozen-feature-c-grid 0.01 0.03 0.1 0.3 1 3 10 30
```

- Fixed feature/label ordering in the 32-view feature extraction path.
  - Features are view-major.
  - Labels are now tiled in the same view-major order.

### Scratch Training Option

- Added:

```powershell
--pretrained 0
```

- This disables ImageNet ConvNeXt-Tiny weight loading and trains from random initialization.
- Scratch runs are marked in experiment names with:

```text
pretrained=0
```

This feature was added but not fully benchmarked yet.

### Windows And JSON Robustness

- Log files now use UTF-8 encoding.
- Config snapshots are written as valid JSON instead of appended JSON fragments.
- Fresh model runs recompute predictions instead of silently reusing stale prediction archives.
- Explicit `--npz_path` is still supported for loading saved predictions.

## Smoke-Test Commands

Run the standard debug smoke test:

```powershell
python src\script_train_5_models_singleSplit.py `
  --config "config (debug).json" `
  --debugMode 100 `
  --splitting-choice random_year1only `
  --yearChosen 2020 `
  --tag smoke
```

Run the frozen-feature debug smoke test:

```powershell
python src\script_train_5_models_singleSplit.py `
  --config "config (debug).json" `
  --debugMode 100 `
  --splitting-choice random_year1only `
  --yearChosen 2020 `
  --tag smoke_frozen `
  --run-frozen-features 1
```

Run a scratch-training smoke test:

```powershell
python src\script_train_5_models_singleSplit.py `
  --config "config (debug).json" `
  --debugMode 100 `
  --splitting-choice random_year1only `
  --yearChosen 2020 `
  --tag smoke_scratch `
  --pretrained 0
```

## Notes On Artifacts

The following directories/files should generally not be committed:

```text
data/
expe/
models/
*.npz
*.pth
*.pt
overall_perf_summary.json
```

The processed data can live in the workspace under `data/`, but it should stay ignored by Git.

## Verification Performed

The following checks were run during development:

```text
compileall: OK
selected-state helper: OK
UTF-8 logging: OK
augmentation plot generated from real grains
failed-prediction plot generated from real grains
confusion matrix plots generated
32-view metric helper: OK
exact-fit robust class sampling: OK
frozen-feature grid search: OK
```

The frozen-feature smoke run completed successfully on CPU with `--debugMode 100`.

## Remaining Professor TODOs

- Fully benchmark `--pretrained 0` scratch training.
- Add Mengtsu et al. dataset support.
- Add SCOOP/BACS dataset mode with bac-based splitting.
- Add image downsampling experiments.
- Optional: refactor code structure.
- Optional: data cleaning and artifact filtering.
