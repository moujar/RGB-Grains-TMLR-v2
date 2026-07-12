# Changelog

## 2026-07 — Audit fixes: pipeline frozen-feature flag, restored color histogram

Audited every claim in `docs/TODO.md` against the actual code (not just the
prose). Everything reported "Done" checked out correctly (best/SWA selection,
augmentation grid, failed-prediction viz, confusion-matrix recall/precision
placement, 32-view train balanced accuracy, SCOOP/BACS splitting, downsampling,
torchvision A/B, the three 5.5 cleaning modes). Two real gaps found and fixed:

- `rgb_grains/pipeline.py` now exposes `--run-frozen-features` /
  `--frozen-feature-c-grid` as first-class flags (previously only reachable
  through the `--extra-train-args` escape hatch).
- `rgb_grains/utils/npz_to_jpg.py`: restored the per-channel RGB pixel-value
  histogram diagnostic that the original `segmentation/control_script_npz_to_jpg.py`
  produced, dropped when it was consolidated with `src/script_npz_to_jpg.py`.
  New `plot_color_histogram()`, opt-in via `--color-histogram`.

The perfomix by-size cleaning threshold remains a genuine open item (needs a
real area-distribution study on data not available in this environment), and
5.2 (Mengtsu), the SCOOP/BACS full run, the resolution sweep, and the
torchvision GPU comparison remain blocked on data/compute, not code.

## 2026-07 — Package restructuring + pipeline entrypoint + TODO 5.3/5.5

- Restructured the flat `src/`/`segmentation/` scripts into an installable
  package, `rgb_grains/` (`data/`, `models/`, `viz/`, `utils/`, `segmentation/`),
  plus `configs/`, `scripts/`, `tests/`, `docs/`, `archive/`. See `README.md`.
- Added `rgb_grains/pipeline.py`: a single entrypoint orchestrating
  segmentation (optional) -> data cleaning (optional) -> training -> validation,
  with automatic GPU/CPU device selection (unchanged from the prior automatic
  `torch.cuda.is_available()` behavior, now easier to reach as one command).
- Implemented TODO 5.3 (image-resolution ablation): `--downsample-kernel`/
  `--downsample-mode` on `rgb_grains.train`/`rgb_grains.pipeline`, pooling
  crops then upsampling back to the model's input size.
- Implemented TODO 5.5 (by-size data cleaning): `rgb_grains/data/cleaning.py`,
  used both as a non-destructive in-loader filter (`--clean-data` on
  `rgb_grains.train`) and a one-time physical `_processed` -> `_excluded`
  move (`rgb_grains.pipeline --clean-data`).
- Consolidated the two near-duplicate `npz_to_jpg` scripts
  (`src/script_npz_to_jpg.py`, `segmentation/control_script_npz_to_jpg.py`)
  into `rgb_grains/utils/npz_to_jpg.py` with a `--mode {minmax,spectralon}` flag.
- Fixed a bug introduced by the move itself: `Model_ConvNeXt` used to derive
  its output directory from its own file's location on disk; after moving it
  out of `src/`, that pointed at the wrong place. It now takes an explicit
  `base_dir` (threaded from `rgb_grains.train`'s `--base-dir`), matching the
  experiment output directory exactly. Caught via an actual end-to-end run.
- Added a clear assertion (instead of an opaque pandas internals crash) when
  `--yearChosen` selects a year with zero samples in the loaded dataset —
  found while validating with real data: the currently-checked-in `perfomix`
  pure-stand crops are all dated 2021, so `--yearChosen 2020` (the CLI
  default) has nothing to train on with this data snapshot; use `2021` or a
  year-agnostic `--splitting-choice`.
- Moved `perfomix_mixtures.{csv,tsv}` into `rgb_grains/data/` as package data;
  `_load_mix_dict()` now defaults to that bundled path instead of assuming CWD.
- Archived stray root-level run artifacts (`slurm-*.out`, `ys_for_debug.npz`,
  duplicate `overall_perf_summary*.json` variants, a leftover confusion-matrix
  PNG) into `archive/legacy_outputs/`; archived the cluster job scripts under
  `segmentation/jobs and logs/` (hardcoded personal paths) into
  `archive/segmentation_job_scripts/`.
- Added `pyproject.toml`, `requirements.txt`, `LICENSE` (MIT), and
  `tests/` (pytest) covering the two new pieces of logic above.
- `docs/TODO.md` now carries a status table on top of the original request;
  `docs/CHANGELOG_source_daryl_branch.md` preserves the prior branch's notes.

See `docs/CHANGELOG_source_daryl_branch.md` for engineering work done before
this restructuring (checkpoint-selection cleanup, diagnostic plots, SCOOP/BACS
dataset mode, frozen-feature analysis, etc.).
