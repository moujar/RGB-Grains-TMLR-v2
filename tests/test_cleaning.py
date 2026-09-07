"""Tests for TODO 5.5 data-cleaning (rgb_grains/data/cleaning.py)."""
import numpy as np

from rgb_grains.data.cleaning import (
    grain_active_area,
    filter_excluded_files,
    detect_outliers_od,
    move_outliers_od,
)


def _make_grain(area_pixels, size=64, channel_value=1.0):
    """A (size, size, 3) crop with exactly `area_pixels` foreground pixels (value `channel_value`)."""
    x = np.zeros((size, size, 3), dtype=np.float32)
    flat = x.reshape(-1, 3)
    flat[:area_pixels] = channel_value
    return x


def test_grain_active_area_counts_nonzero_across_channels():
    x = _make_grain(area_pixels=100)
    # each foreground pixel contributes 3 (one per channel) to the sum-over-channels area
    assert grain_active_area(x) == 300


def test_filter_excluded_files_drops_small_grains(tmp_path):
    small = tmp_path / "small.npz"
    big = tmp_path / "big.npz"
    np.savez(small, x=_make_grain(10))
    np.savez(big, x=_make_grain(10000))

    kept = filter_excluded_files(
        [str(small), str(big)],
        dataset_choice="perfomix",
        min_area=1000,
        enabled=True,
        log_fn=lambda *a, **k: None,
    )
    assert kept == [str(big)]


def test_filter_excluded_files_noop_when_disabled(tmp_path):
    small = tmp_path / "small.npz"
    np.savez(small, x=_make_grain(1))
    files = [str(small)]
    assert filter_excluded_files(files, enabled=False) == files


def test_detect_outliers_od_flags_grossly_different_grains(tmp_path):
    rng = np.random.RandomState(0)
    files = []
    for i in range(18):
        area = 5000 + int(rng.randint(-50, 50))
        f = tmp_path / f"normal{i}.npz"
        np.savez(f, x=_make_grain(area, channel_value=1.0))
        files.append(str(f))
    for i in range(2):
        f = tmp_path / f"outlier{i}.npz"
        np.savez(f, x=_make_grain(40000, channel_value=3.0))
        files.append(str(f))

    outliers = detect_outliers_od(files, contamination=0.1, log_fn=lambda *a, **k: None)
    outlier_names = {f.split("/")[-1] for f in outliers}
    assert outlier_names == {"outlier0.npz", "outlier1.npz"}


def test_detect_outliers_od_skips_when_too_few_grains():
    files = [f"grain{i}.npz" for i in range(3)]
    assert detect_outliers_od(files, log_fn=lambda *a, **k: None) == []


def test_move_outliers_od_moves_flagged_files(tmp_path):
    processed_dir = tmp_path / "perfomix_test_processed"
    processed_dir.mkdir()
    rng = np.random.RandomState(0)
    for i in range(18):
        area = 5000 + int(rng.randint(-50, 50))
        np.savez(processed_dir / f"normal{i}.npz", x=_make_grain(area, channel_value=1.0))
    for i in range(2):
        np.savez(processed_dir / f"outlier{i}.npz", x=_make_grain(40000, channel_value=3.0))

    n_moved = move_outliers_od(data_dir=tmp_path, dataset_choice="perfomix", contamination=0.1)
    assert n_moved == 2

    excluded_dir = tmp_path / "perfomix_test_excluded"
    assert (excluded_dir / "outlier0.npz").exists()
    assert (excluded_dir / "outlier1.npz").exists()
    assert not (processed_dir / "outlier0.npz").exists()
    assert (processed_dir / "normal0.npz").exists()
