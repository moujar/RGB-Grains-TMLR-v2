"""Data-cleaning / outlier-exclusion utilities (TODO 5.5, "by size").

Some grain crops produced by the segmentation pipeline are dust or other
watershed artifacts rather than actual grains. This module implements the
"easy" by-size criterion from ``docs/TODO.md`` #5.5: a grain crop's active
(non-background) pixel count -- summed over the 3 stored channels, following
the convention used by the original QC script
(``rgb_grains/segmentation/control_script_npz_to_jpg.py``) -- must fall
within a plausible range, otherwise the file is excluded before it ever
reaches ``load_datasets_microplot_split``.

Defaults differ by dataset because grain size/crop conventions differ:
  * ``perfomix``: min_area=15000  (matches the threshold used historically by
    the segmentation QC script for this dataset).
  * ``SCOOP``:    min_area=110000 (matches the value specified in
    ``docs/TODO.md`` #5.5 for the BACS data).
These are provisional heuristics ("Perfomix: to be studied" per the TODO) --
override them with ``--min-grain-area``/``--max-grain-area`` if you have a
better-justified cutoff (e.g. from ``rgb_grains/viz/eda.py``'s area
histograms).
"""
from __future__ import annotations

import glob
import shutil
from pathlib import Path

import numpy as np

DEFAULT_MIN_AREA = {
    "perfomix": 15000,
    "SCOOP": 110000,
}
DEFAULT_MAX_AREA = None  # no upper bound by default; large grains are flagged, not dropped, upstream


def grain_active_area(x: np.ndarray, threshold: float = 0.01) -> int:
    """Active-pixel area of a grain crop, summed over channels.

    Matches the formula used by the original segmentation QC script and by
    ``docs/TODO.md`` #5.5: ``np.sum(img > threshold)`` over the whole (H, W, C)
    array. Background pixels are exactly 0, so this is robust to whether
    ``x`` is raw sensor counts or spectralon-normalized reflectance.
    """
    return int(np.sum(np.asarray(x) > threshold))


def filter_excluded_files(
    files,
    dataset_choice="perfomix",
    min_area=None,
    max_area=None,
    enabled=True,
    log_fn=print,
):
    """Drop grain ``.npz`` files whose active area is outside [min_area, max_area].

    Parameters
    ----------
    files : list[str]
        Paths to grain ``.npz`` files (each must contain array ``x``).
    dataset_choice : str
        Used only to pick a default ``min_area`` when one isn't given explicitly.
    min_area, max_area : int or None
        Explicit overrides. ``None`` disables that bound (falls back to the
        per-dataset default for ``min_area``; ``max_area`` is unbounded by default).
    enabled : bool
        If False, this is a no-op (returns ``files`` unchanged) -- useful to
        keep the old, uncleaned behavior available for comparison.
    log_fn : callable
        Logging function, e.g. the experiment's ``_log_fn``.

    Returns
    -------
    list[str]
        The subset of ``files`` that passed the area filter.
    """
    if not enabled:
        return list(files)

    if min_area is None:
        min_area = DEFAULT_MIN_AREA.get(dataset_choice, None)
    if max_area is None:
        max_area = DEFAULT_MAX_AREA

    if min_area is None and max_area is None:
        return list(files)

    kept = []
    dropped = []
    for f in files:
        try:
            area = grain_active_area(np.load(f)["x"])
        except Exception as e:
            log_fn(f"  [cleaning] Warning: could not read {f} ({e}); keeping it.")
            kept.append(f)
            continue
        if min_area is not None and area < min_area:
            dropped.append((f, area))
            continue
        if max_area is not None and area > max_area:
            dropped.append((f, area))
            continue
        kept.append(f)

    if dropped:
        log_fn(
            f"  [cleaning] Excluded {len(dropped)}/{len(files)} grain(s) outside "
            f"area range [{min_area}, {max_area}] (dataset={dataset_choice})."
        )
    return kept


def move_excluded_files(
    data_dir,
    dataset_choice="perfomix",
    min_area=None,
    max_area=None,
    dry_run=False,
):
    """One-time physical cleanup: move outlier-area grains out of every
    ``*_processed`` folder under ``data_dir`` into a sibling ``*_excluded``
    folder, matching the "or directly move from the _processed folder to an
    _excluded folder" wording of TODO 5.5.

    Unlike ``filter_excluded_files`` (which filters an in-memory file list at
    load time, non-destructively), this function moves files on disk. Used by
    ``rgb_grains.pipeline``'s cleaning stage.

    Returns the number of files moved (or that would be moved, if dry_run).
    """
    if min_area is None:
        min_area = DEFAULT_MIN_AREA.get(dataset_choice, None)
    if max_area is None:
        max_area = DEFAULT_MAX_AREA

    pattern = "SCOOP-R2022-bacs_processed" if "SCOOP" in dataset_choice else "perfomix*_processed"
    processed_dirs = sorted(Path(data_dir).glob(pattern))

    n_moved = 0
    for processed_dir in processed_dirs:
        excluded_dir = processed_dir.parent / (processed_dir.name.replace("_processed", "_excluded"))
        for npz_path in sorted(processed_dir.glob("*.npz")):
            try:
                area = grain_active_area(np.load(npz_path)["x"])
            except Exception as e:
                print(f"  [cleaning] Warning: could not read {npz_path} ({e}); skipping.")
                continue
            too_small = min_area is not None and area < min_area
            too_large = max_area is not None and area > max_area
            if too_small or too_large:
                n_moved += 1
                if dry_run:
                    print(f"  [cleaning] would move {npz_path.name} (area={area}) -> {excluded_dir.name}/")
                else:
                    excluded_dir.mkdir(exist_ok=True)
                    shutil.move(str(npz_path), str(excluded_dir / npz_path.name))
    return n_moved
