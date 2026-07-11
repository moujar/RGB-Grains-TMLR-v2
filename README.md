# RGB-Grains

Code release for **"Estimating Varietal Proportions of Wheat Grain Varieties
with Fine-Tuned Deep Vision Models"** (Houmed, Ho, Moujar, Oloruntobi, Okou,
Lu, Khuong, Ecarnot, Flutre, Sun-Hosoya, Landes — Université Paris-Saclay /
LISN, INRIA / GQE, INRAE, CNRS, AgroParisTech / AGAP Institut / DATAIA).

Agroecology is increasingly relying on multivarietal cropping (sowing several
varieties of the same species in one plot), which requires estimating each
variety's yield after harvest — i.e. classifying grains. This repository
contains (1) a dataset of RGB(-reduced hyperspectral) images of individual
wheat grains from 8 varieties collected in a field trial in Saclay, France,
and (2) the full pipeline to fine-tune a ConvNeXt-Tiny classifier on them,
evaluate it, and simulate mixed-cropping proportion estimation from the
predictions. See `paper/` for the full manuscript.

## Repository layout

```
rgb_grains/                  installable Python package
  data/
    dataset.py                dataset loading + train/test splitting (by microplot/bac)
    cleaning.py                TODO 5.5: by-size grain exclusion
    perfomix_mixtures.csv      variety composition of each "perfomix" mixture
  models/
    convnext.py                 ConvNeXt-Tiny (custom, no torchvision dependency) + training loop
    classifier_head.py          frozen-feature logistic-regression / LP-FT kickstart
  viz/
    plots.py                    augmentation/failed-prediction/confusion-matrix/calibration plots
    eda.py                       exploratory data analysis (used by eda.ipynb)
  utils/
    tools.py                    balanced-accuracy / soft-label metrics
    npz_to_jpg.py                grain crop -> JPG viewer/QC tool
    manual_tag.py                TODO 5.5: manual by-hand exclusion tagging tool
  segmentation/                raw hyperspectral .hdr cubes -> RGB grain crops (optional extra)
  train.py                      single train/test split: fine-tune + evaluate ConvNeXt-Tiny
  pipeline.py                   end-to-end entrypoint: segment (optional) -> clean -> train -> validate

configs/                     training configs (debug / default / serious)
scripts/                     standalone analysis CLIs (not part of the core pipeline)
tests/                       pytest tests for the newer utilities (cleaning, resolution ablation)
docs/                        TODO backlog, changelog, segmentation notes
archive/                     legacy run logs / stray output files, kept for provenance only
eda.ipynb, eda_outputs/      interactive + generated exploratory data analysis
paper/                       the manuscript PDF
```

## Installation

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

This installs the core dependencies (PyTorch, numpy, pandas, scikit-learn,
seaborn, matplotlib) and registers console scripts `rgb-grains-train`,
`rgb-grains-pipeline`, `rgb-grains-eda`, `rgb-grains-npz2jpg`, `rgb-grains-tag`.
If you only
need `requirements.txt`-style installs (no local package build), `pip
install -r requirements.txt` gives the same core dependencies, and you'd run
modules as `python -m rgb_grains.train ...` instead.

To also run the raw-hyperspectral segmentation pipeline (`rgb_grains/segmentation/`,
`.hdr` → RGB grain crops), install the extra dependencies (OpenCV, `spectral`,
scikit-image, numba, ...):

```bash
pip install -e ".[segmentation]"
```

GPU: PyTorch is used with automatic CUDA detection (`rgb_grains/models/convnext.py`) —
training runs on GPU whenever one is visible to `torch.cuda.is_available()`,
and falls back to CPU (with a warning; CPU training is very slow) otherwise.
No extra flag is needed; just run on a CUDA-enabled machine, or via `sbatch.sh`
on a SLURM cluster with `--gres=gpu:1` (edit the venv-activation line at the
top of that script first).

## Data layout

Place (or symlink) the processed grain crops under `data/` at the repo root
(gitignored — this is per-user data, not tracked):

```
data/
  perfomix_2019-2020_IE_HSI_var1-8_processed/*.npz   # pure-stand, 8 varieties
  perfomix_2020-2021_IE_HSI_var1-8_processed/*.npz
  perfomix_2019-2020_IE_HSI_mix_processed/*.npz      # mixed-stand
  perfomix_2020-2021_IE_HSI_mix_processed/*.npz
  SCOOP-R2022-bacs_processed/*.npz                    # SCOOP/BACS, 4 varieties
```

Each `.npz` holds a `(252, 252, 3) int16` array `x` (3 spectral bands) plus a
`means_over_spectralon` calibration reference. These crops are produced by
`rgb_grains/segmentation/` from raw Hyspex `.hdr` cubes — see that folder's
docstrings, or run the full pipeline with `--raw-hdr-dir` (below) to generate
them as part of a single invocation.

Run the EDA to sanity-check whatever data you've placed there before training:

```bash
python -m rgb_grains.viz.eda --data-dir data --out eda_outputs
# or open eda.ipynb for the interactive version
```

## Quickstart: the end-to-end pipeline

`rgb_grains/pipeline.py` is the single entrypoint that takes you from raw
data (optional) to a trained, validated model:

```bash
# Fast smoke test on CPU (small debug config, tiny per-class sample cap):
python -m rgb_grains.pipeline \
  --config configs/debug.json --data-dir data \
  --splitting-choice random_year1only --yearChosen 2021 \
  --debugMode 60 --pretrained 0 --tag smoke

# A real training + validation run (GPU used automatically if available):
python -m rgb_grains.pipeline \
  --config configs/serious.json --data-dir data \
  --splitting-choice muPlot-3muTrain-1muTest --dataset-choice perfomix

# SCOOP/BACS, all 3 microplot-holdout folds in one call:
python -m rgb_grains.pipeline \
  --config configs/serious.json --dataset-choice SCOOP \
  --splitting-choice bacs_2train_1test --folds 0 1 2

# Also segment raw .hdr cubes first (needs `pip install -e ".[segmentation]"`):
python -m rgb_grains.pipeline \
  --raw-hdr-dir /path/to/hdr_cubes --dataset-name perfomix_2020-2021_IE_HSI_var1-8 \
  --config configs/serious.json

# One-time physical cleanup of outlier-area grain crops (TODO 5.5, by size), without training:
python -m rgb_grains.pipeline --clean-data --clean-dry-run --data-dir data

# One-time physical cleanup of outlier grain crops via unsupervised outlier
# detection (TODO 5.5, "by OD"; no labels used), without training:
python -m rgb_grains.pipeline --od-exclude --clean-dry-run --data-dir data
```

Each stage can be skipped independently: `--skip-segmentation` (or simply
omit `--raw-hdr-dir`), and cleaning only runs when `--clean-data` and/or
`--od-exclude` is passed. Run `python -m rgb_grains.pipeline --help` for the
full flag reference, organized by stage.

For manual by-hand cleaning (TODO 5.5, "manually" — catching artifacts the
automatic filters miss), export grain crops to JPG, delete the bad ones
yourself, then apply the review:

```bash
rgb-grains-tag export data/perfomix_..._processed reviews/perfomix_review
# ... delete anomalous JPGs from reviews/perfomix_review in Finder/Preview/etc ...
rgb-grains-tag apply data/perfomix_..._processed reviews/perfomix_review
```

Outputs land under `<base-dir>/expe/<experiment_name>/` (config snapshot,
training log, augmentation/failed-prediction/confusion-matrix/calibration
plots, saved predictions) and the fine-tuned weights under
`<base-dir>/models/`. Every run also appends one JSON line to
`<base-dir>/overall_perf_summary.json`; aggregate several runs with:

```bash
python scripts/read_perf_summary.py --summary-json overall_perf_summary.json
```

### Just training (no segmentation/cleaning stages)

`rgb_grains/train.py` is what `pipeline.py` calls under the hood for each
fold; use it directly if you don't need the orchestration:

```bash
python -m rgb_grains.train --config configs/debug.json --debugMode 100 \
  --splitting-choice random_year1only --yearChosen 2021 --tag smoke
```

See `python -m rgb_grains.train --help` for the full set of experiment
toggles (LP-FT kickstart, frozen-feature logistic-regression grid search,
class restriction/equalization, mixed-vs-pure test composition, etc.).

To A/B-test the hand-rolled ConvNeXt-Tiny against torchvision's reference
implementation (see "Project status / TODO" below), add `--backbone-impl
torchvision` (needs `pip install -e ".[torchvision]"`) to either command above.

### Mixture-proportion simulation

Once you have a `predictions.npz` from a pure-stand test run, you can
simulate binary-mixture proportion estimation error and sanity-check labels:

```bash
python scripts/simulate_varietal_proportions.py --predictions-npz expe/<run>/predictions.npz
python scripts/control_labels.py --predictions-npz expe/<run>/predictions.npz --name-tag <run>
```

## Configs

`configs/default.json`, `configs/debug.json` (fast, no augmentation, 3
epochs), `configs/serious.json` (full-scale, 50 epochs, batch size 128) share
one schema (see any file for all keys: seed, crop size, learning rates, SWA
settings, mixup/cutmix, etc.) and are selected via `--config <path>`.

## Project status / TODO

`docs/TODO.md` is the original backlog from the lab; `docs/CHANGELOG_source_daryl_branch.md`
documents engineering work done on an earlier branch. As of this restructuring:

- **Done**: checkpoint-selection cleanup (best-val vs SWA vs last), augmentation-example /
  failed-prediction / confusion-matrix-with-recall-precision plots, train-set 32-view
  balanced accuracy, SCOOP/BACS dataset mode with bac-based splitting, image-resolution
  ablation (`--downsample-kernel`/`--downsample-mode`), by-size data cleaning
  (`rgb_grains/data/cleaning.py`, `--clean-data`), manual by-hand cleaning
  (`rgb_grains/utils/manual_tag.py`, `rgb-grains-tag`), and a basic unsupervised
  outlier-detection cleaning pass (`--od-exclude`, `IsolationForest`-based, provisional).
- **Blocked**: the Mengtsu et al. dataset adapter — no data for it is present in this
  repository yet; add a new `dataset_choice` following the `SCOOP` branch in
  `rgb_grains/data/dataset.py` once the files are available.
- **Not the default, opt-in for A/B testing**: `--backbone-impl torchvision` swaps in
  `torchvision.models.convnext_tiny` (requires `pip install -e ".[torchvision]"`) behind the
  same `Model_ConvNeXt` training loop, so the hand-rolled `custom` backbone (default; see the
  module docstring in `rgb_grains/models/convnext.py` for why it was written from scratch —
  no torchvision dependency, in case of a sandboxed submission environment with no network)
  can be benchmarked against the reference torchvision implementation. Both pull the exact
  same pretrained checkpoint (`download.pytorch.org/models/convnext_tiny-983f1562.pth`), so
  any accuracy delta on GPU should reflect implementation details (init recipe, stochastic
  depth, etc.), not different weights. CPU-verified for correct wiring (shapes, checkpoint
  save/load, experiment-name tagging via `_backbone=torchvision`); accuracy comparison
  between the two still needs an actual GPU run — not done as part of this change.

## Testing

```bash
pip install -e ".[dev]"
pytest tests/
```

Covers the newer pieces of logic added during this restructuring
(`rgb_grains/data/cleaning.py` incl. outlier detection, `rgb_grains/utils/manual_tag.py`,
and the resolution-ablation option in `GrainDataset_ConvNeXt`). The rest of the pipeline
is validated by actually running it end-to-end on real data (see "Quickstart" above) —
there's no substitute for that with a model this size.

## License

MIT — see `LICENSE`. The vendored watershed helpers under
`rgb_grains/segmentation/gala/` are BSD-3-Clause (see the `NOTICE.md` there).

## Citation

If you use this code or data, please cite the paper (see `paper/` for the PDF):

```
Houmed, O.A., Ho, Q.P., Moujar, A., Oloruntobi, O.P., Okou, G.D.M., Lu, R.,
Khuong, T.G.H., Ecarnot, M., Flutre, T., Sun-Hosoya, L., Landes, F.P.
"Estimating Varietal Proportions of Wheat Grain Varieties with Fine-Tuned
Deep Vision Models."
```
