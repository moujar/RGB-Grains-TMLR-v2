"""Tests for TODO 5.5 data-cleaning (rgb_grains/data/cleaning.py)."""
import numpy as np

from rgb_grains.data.cleaning import grain_active_area, filter_excluded_files


def _make_grain(area_pixels, size=64):
    """A (size, size, 3) crop with exactly `area_pixels` foreground pixels (value 1.0)."""
    x = np.zeros((size, size, 3), dtype=np.float32)
    flat = x.reshape(-1, 3)
    flat[:area_pixels] = 1.0
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
