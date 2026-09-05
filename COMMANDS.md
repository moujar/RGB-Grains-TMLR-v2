# Command Reference

Last verified against the command-line parsers: 2026-09-05.

Run commands from the repository root unless a section says otherwise. Activate
the project environment first:

```powershell
conda activate grainrgb
cd C:\Users\royka\Documents\GrainRGB\RGB-Grains-TMLR-v2
```

Use the program's built-in help as the final source of truth:

```powershell
python src\script_train_5_models_singleSplit.py --help
python src\eda.py --help
python src\script_npz_to_jpg.py --help
```

## Training

The main entry point trains and evaluates one model on one split:

```powershell
python src\script_train_5_models_singleSplit.py [OPTIONS]
```

### Common runs

Fast Perfomix smoke test:

```powershell
python src\script_train_5_models_singleSplit.py `
  --config "config (debug).json" `
  --debugMode 100 `
  --splitting-choice random_year1only `
  --yearChosen 2020 `
  --tag smoke
```

Standard full Perfomix run, with three microplots for training and one for test:

```powershell
python src\script_train_5_models_singleSplit.py `
  --config "config (serious).json" `
  --splitting-choice muPlot-3muTrain-1muTest `
  --fold 0 `
  --equalize-classes 0 `
  --trainOnMixedAndPure 0 `
  --tag baseline
```

Train with pure and PLL/mixed data:

```powershell
python src\script_train_5_models_singleSplit.py `
  --config "config (serious).json" `
  --splitting-choice muPlot-3muTrain-1muTest `
  --fold 0 `
  --trainOnMixedAndPure 1 `
  --combineMixedAndPureInTest 1 `
  --tag pure_plus_pll
```

Train on PLL/mixed data only and evaluate on pure plus mixed data:

```powershell
python src\script_train_5_models_singleSplit.py `
  --config "config (serious).json" `
  --splitting-choice trainOnMixedOnly `
  --fold 0 `
  --combineMixedAndPureInTest 1 `
  --tag pll_only
```

Train from random initialization:

```powershell
python src\script_train_5_models_singleSplit.py `
  --config "config (serious).json" `
  --splitting-choice muPlot-3muTrain-1muTest `
  --pretrained 0 `
  --tag scratch
```

Run frozen-feature extraction and logistic-regression model selection:

```powershell
python src\script_train_5_models_singleSplit.py `
  --config "config (serious).json" `
  --splitting-choice muPlot-3muTrain-1muTest `
  --run-frozen-features 1 `
  --frozen-feature-c-grid 0.01 0.03 0.1 0.3 1 3 10 30 `
  --tag frozen_features
```

Run the three SCOOP/BACS folds as three commands by changing `--fold`:

```powershell
python src\script_train_5_models_singleSplit.py `
  --config "config (serious).json" `
  --dataset-choice SCOOP `
  --splitting-choice bacs_2train_1test `
  --fold 0 `
  --tag scoop
```

Cross-year learning-curve point using `N` samples from the second year:

```powershell
python src\script_train_5_models_singleSplit.py `
  --config "config (serious).json" `
  --splitting-choice muPlots_mvblNY2 `
  --yearChosen 2021 `
  --NsamplesYear2 100 `
  --fold 0 `
  --tag cross_year_n100
```

### Training options

| Argument | Default | Purpose |
|---|---:|---|
| `--config PATH` | `./config.json` | Training configuration JSON. |
| `--tag TEXT` | empty | Prefix added to the experiment directory name. |
| `--debugMode N` | `0` | Limit data for a quick run; `0` uses all available data. |
| `--dataset-choice {perfomix,SCOOP}` | `perfomix` | Select the dataset layout and label parser. |
| `--splitting-choice MODE` | `muPlot-3muTrain-1muTest` | Select the train/test split strategy. |
| `--fold N` | `0` | Select the held-out split or microplot/bac. |
| `--yearChosen YEAR` | `2020` | Year used by single-year and cross-year modes. |
| `--test_ratio FLOAT` | `0.5` | Test fraction for random split modes. |
| `--restrict-classes 0\|1` | `0` | Restrict training to the hardcoded subset when set to `1`. |
| `--equalize-classes 0\|1` | `0` | Resample training classes; the project default should remain `0`. |
| `--testOnMixedOnly 0\|1` | `0` | Evaluate only on PLL/mixed stands. |
| `--trainOnMixedAndPure 0\|1` | `0` | Add all PLL/mixed grains to the pure training data. |
| `--testOnWholePureOnly 0\|1` | `0` | Evaluate only on all pure grains; intended with `trainOnMixedOnly`. |
| `--combineMixedAndPureInTest 0\|1` | `1` | Include both pure and PLL/mixed samples in evaluation. |
| `--NsamplesYear2 N` | `0` | Number of second-year samples for `muPlots_mvblNY2`. |
| `--pretrained 0\|1` | `1` | Use ImageNet weights (`1`) or random initialization (`0`). |
| `--kickstart 0\|1` | `0` | Initialize the classifier head from a 32-view linear probe. |
| `--kickstart_path PATH` | none | Load or save cached kickstart weights. |
| `--kickstart_recompute 0\|1` | `0` | Ignore and overwrite an existing kickstart cache. |
| `--run-frozen-features 0\|1` | `0` | Run the additional frozen-feature experiment. |
| `--frozen-feature-c-grid VALUES...` | eight values | Logistic-regression `C` candidates. |
| `--microplot-exclusions PATH` | `microplot_exclusions.json` | Exclude configured microplots before splitting. |
| `--reload 0\|1` | `0` | Load a previously trained model instead of fitting. |
| `--reload_path PATH` | none | Existing experiment directory whose config should be reused. |
| `--modelPath PATH` | none | Explicit model checkpoint to load. |
| `--npz_path PATH` | none | Re-evaluate an existing prediction archive without loading a model. |

Microplot exclusions are read from `microplot_exclusions.json`. Each rule must
contain `microplot` and may narrow the match with `dataset_choice`, `mixed`,
`mix`, `label`, or `year`. Pass an empty string to disable the file explicitly.

### Split modes

| Mode | Intended use |
|---|---|
| `muPlot_year1only` | Train/test by microplot within one year. |
| `muPlot-2muTrain-2muTest` | One train and one test microplot from each year. |
| `muPlot-2muTrain-2muTest-yearGeneralization_year1only` | Train on one year and test on the other. |
| `muPlot-3muTrain-1muTest` | Standard Perfomix split. |
| `muPlot-1muTrain-1muTest-crossYears` | One train and test microplot across years. |
| `random_1muPlot` | Random split using one microplot. |
| `random_3muPlot` | Random split using three microplots. |
| `random_year1only` | Random split within one year. |
| `random_4muPlot` | Random split over four microplots. |
| `inferenceMode_4muTrain-0muTest` | Train on all pure microplots for PLL inference. |
| `trainOnMixedOnly` | Train only on PLL/mixed data. |
| `muPlots_mvblNY2` | Cross-year learning curve with variable second-year data. |
| `bacs_2train_1test` | SCOOP only: two bacs train, one bac test. |

### Configuration files

The CLI selects the JSON file; model and optimization settings live inside it:

| Key | Meaning |
|---|---|
| `seed` | Random seed. |
| `nc` | Number of classes; SCOOP overrides this to 4. |
| `crop` | Model crop size. |
| `val_ratio` | Validation fraction taken from training data. |
| `time_budget`, `predict_t` | Runtime limits. |
| `wd`, `lr_bb`, `lr_hd` | Weight decay, backbone LR, and head LR. |
| `warmup_ep`, `max_ep` | Warmup and maximum epochs. |
| `bs` | Batch size. |
| `dp`, `head_drop` | Dropout settings. |
| `swa_start`, `swa_keep` | SWA start epoch and retained snapshots. |
| `trainAugmentations` | Enable normal training augmentations. |
| `cutMix_mixUp`, `alpha` | Enable and configure CutMix/MixUp. |
| `expe` | Base experiment name. |

Experiments write to `expe/<experiment-name>/`, model checkpoints to `models/`,
and one JSON object per run to `overall_perf_summary.json`.

## Modal GPU Runs

`modal_runner.py` executes the normal training CLI on a Modal GPU. Code is
included in the container image, while datasets and outputs use persistent
Volumes. The runner allows only one active container to limit accidental spend.

Install and authenticate the Modal CLI:

```powershell
python -m pip install modal
modal setup
modal profile current
```

Create the two Volume v2 stores once in the active Modal profile:

```powershell
modal volume create --version=2 rgb-grains-data
modal volume create --version=2 rgb-grains-output
```

Upload the four Perfomix folders. Ending the remote path with `/` keeps each
local folder name at the Volume root, which is the layout expected by the
dataloader:

```powershell
modal volume put rgb-grains-data `
  data\perfomix_2019-2020_IE_HSI_var1-8_processed /
modal volume put rgb-grains-data `
  data\perfomix_2020-2021_IE_HSI_var1-8_processed /
modal volume put rgb-grains-data `
  data\perfomix_2019-2020_IE_HSI_mix_processed /
modal volume put rgb-grains-data `
  data\perfomix_2020-2021_IE_HSI_mix_processed /
```

For large folders containing thousands of small files, prefer one TAR upload.
This avoids a long series of individual network requests. The TAR is not
compressed because the NPZ files are already compressed. This example stages
the archive in the Windows temporary directory so it cannot be committed:

```powershell
$dataset = "perfomix_2019-2020_IE_HSI_var1-8_processed"
$archive = Join-Path $env:TEMP "$dataset.tar"

tar.exe -cf $archive -C data $dataset
modal volume put rgb-grains-data $archive /perfomix-data.tar
modal run modal_runner.py::extract_data_archive `
  --archive-name perfomix-data.tar
```

The extraction function uses CPU only, validates archive paths, commits the
extracted files, and deletes the remote TAR after a successful extraction.
Delete the local temporary archive after Modal reports success:

```powershell
Remove-Item -LiteralPath $archive
```

Confirm the upload before spending GPU time:

```powershell
modal volume ls rgb-grains-data
```

Preview argument parsing without calling the remote training function:

```powershell
modal run modal_runner.py --preview
```

Run the default smoke test on an L4:

```powershell
modal run modal_runner.py
```

Pass any normal training arguments as one quoted `--train-args` value:

```powershell
modal run modal_runner.py `
  --gpu L4 `
  --train-args '--config "config (serious).json" --splitting-choice muPlot-3muTrain-1muTest --fold 0 --equalize-classes 0 --trainOnMixedAndPure 0 --tag modal_baseline_fold0'
```

Use a unique `--tag` for every run. Outputs persist in the
`rgb-grains-output` Volume even after the GPU container exits. List and download
them with:

```powershell
modal volume ls rgb-grains-output expe
modal volume get rgb-grains-output expe\EXPERIMENT_NAME modal_results
```

The runner supports `T4`, `L4`, `A10`, `L40S`, and `A100-40GB`. Begin with L4;
move to a larger GPU only if memory use or runtime justifies it. A job has a
23-hour timeout and requests 64 GB of CPU memory because the current loader
materializes the selected images in RAM.

### Modal profiles and workspaces

Modal resources and billing belong to the active workspace profile. Inspect and
switch only between profiles/workspaces you are authorized to use:

```powershell
modal profile list
modal profile current
modal profile activate PROFILE_NAME
modal token info
```

Each profile has separate Volumes. After switching, create/upload its Volumes
again, and retrieve important results before changing profiles. Do not put Modal
tokens in this repository or use additional accounts to evade service limits,
credit restrictions, or platform terms.

For an already configured profile, Modal also exposes `--profile` directly on
the runner:

```powershell
modal run modal_runner.py --profile PROFILE_NAME --preview
modal run modal_runner.py --profile PROFILE_NAME --gpu L4
```

## PLL Diagnostics

`control_labels.py` reports candidate-set consistency for every mix and
microplot, including its expected classes:

```powershell
python src\control_labels.py
```

This script does not have CLI arguments yet. Set `input_file` and `name_tag`
near the top of the file before running it. Use its output to decide which PLL
microplots belong in `microplot_exclusions.json`.

## Exploratory Data Analysis

Generate metadata, figures, montages, and an EDA report:

```powershell
python src\eda.py `
  --data-dir data `
  --out eda_outputs `
  --mix-csv perfomix_mixtures.csv `
  --max-per-class 400 `
  --montage-n 8 `
  --seed 42
```

| Argument | Default | Purpose |
|---|---:|---|
| `--data-dir PATH` | `data` | Root containing `*_processed` folders. |
| `--out PATH` | `eda_outputs` | Report and figure output directory. |
| `--mix-csv PATH` | `perfomix_mixtures.csv` | Perfomix composition table. |
| `--max-per-class N` | `400` | Samples per class for pixel statistics. |
| `--montage-n N` | `8` | Example grains per montage class. |
| `--seed N` | `42` | Sampling seed. |

## NPZ Image Conversion

Convert one crop:

```powershell
python src\script_npz_to_jpg.py data\example.npz -o example.jpg
```

Convert a directory recursively:

```powershell
python src\script_npz_to_jpg.py data -o grain_jpgs -r --dpi 200
```

Additional switches are `--transpose`, `--no-normalize`, and `--cmap NAME`.

`segmentation/control_script_npz_to_jpg.py` is a separate experimental cleaning
utility. It can move small crops from `_processed` to `_excluded` automatically,
so inspect its thresholds and use a data backup before running it.

## Segmentation

Install `segmentation/requirements.txt` before using these commands.

Convert raw HDR data directly to three-band grain crops:

```powershell
python segmentation\script_hyspex_to_rgb.py `
  --input path\to\hdr_folder `
  --output data\dataset_name_processed `
  --jobs 4 `
  --output-jpg
```

Its tuning switches are `--crop-idx-dim1`, `--reflectance-trim`,
`--watershed-trim`, `--area-min`, `--area-max`, `--solidity`,
`--binary-thresh`, and `--debug`.

Two-stage full-spectrum conversion:

```powershell
python segmentation\hyspex_to_full_band.py path\to\hdr_folder `
  -o path\to\full_band_npz

python segmentation\full_band_to_rgb.py `
  --src path\to\full_band_npz `
  --out path\to\rgb_npz
```

Inspect one HDR acquisition as a high-resolution JPG:

```powershell
python segmentation\control_script_hdr_to_jpg.py path\to\image.hdr
```

Optional tuning arguments are `--crop-idx-dim1`, `--reflectance-trim`, and
`--watershed-trim`.

Generate `brightest_bands.csv` from inside the segmentation directory:

```powershell
cd segmentation
python -m grain_hs.brightest_csv `
  --img-dir path\to\hdr_folder `
  --out-dir path\to\output
cd ..
```

## Legacy Analysis Scripts

These scripts run, but important inputs are currently hardcoded in their source:

| Command | Current limitation |
|---|---|
| `python src\script_simulate_varietal_proportions.py --boot 100` | Prediction archive path and experiment settings are hardcoded. |
| `python src\mini-script-read-overall-perf-summary.py` | Summary filename and expected experiment-name patterns are hardcoded. |
| `python src\quickcheckgather.py` | Demonstration script, not a project workflow. |
| `python src\import_dataset.py` | Untracked local helper, not part of the repository CLI. Never store access tokens in it. |

`sbatch.sh` is a legacy SLURM loop with a hardcoded environment path. Update the
activation path and experiment matrix before submitting it with `sbatch sbatch.sh`.

## Maintenance Rule

Keep this document synchronized with the code:

1. When adding, renaming, or removing an `argparse` argument, update this file in
   the same commit.
2. Add one runnable example for every new workflow, not every minor switch.
3. Mark destructive commands and scripts with hardcoded paths explicitly.
4. Verify affected interfaces with `python <script> --help` and update the
   `Last verified` date above.
5. Prefer moving repeated hardcoded settings into CLI arguments rather than
   documenting source edits as the permanent workflow.
