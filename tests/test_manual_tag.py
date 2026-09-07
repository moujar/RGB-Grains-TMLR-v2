"""Tests for TODO 5.5 manual tagging (rgb_grains/utils/manual_tag.py)."""
import numpy as np

from rgb_grains.utils.manual_tag import export_for_review, apply_review


def _make_grain(size=32):
    x = np.random.rand(size, size, 3).astype(np.float32)
    means = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return x, means


def test_export_then_apply_excludes_deleted_jpgs(tmp_path):
    processed_dir = tmp_path / "foo_processed"
    processed_dir.mkdir()
    for i in range(3):
        x, means = _make_grain()
        np.savez(processed_dir / f"grain{i}.npz", x=x, means_over_spectralon=means)

    review_dir = tmp_path / "review"
    npz_files = export_for_review(processed_dir, review_dir)
    assert len(npz_files) == 3
    jpgs = sorted(review_dir.glob("*.jpg"))
    assert len(jpgs) == 3

    # user "deletes" grain1.jpg as an artifact
    (review_dir / "grain1.jpg").unlink()

    excluded = apply_review(processed_dir, review_dir)
    assert [p.name for p in excluded] == ["grain1.npz"]

    excluded_dir = tmp_path / "foo_excluded"
    assert (excluded_dir / "grain1.npz").exists()
    assert not (processed_dir / "grain1.npz").exists()
    assert (processed_dir / "grain0.npz").exists()
    assert (processed_dir / "grain2.npz").exists()

    exclusion_list = (review_dir / "manual_exclusion_list.txt").read_text()
    assert "grain1.npz" in exclusion_list


def test_apply_review_dry_run_moves_nothing(tmp_path):
    processed_dir = tmp_path / "foo_processed"
    processed_dir.mkdir()
    x, means = _make_grain()
    np.savez(processed_dir / "grain0.npz", x=x, means_over_spectralon=means)

    review_dir = tmp_path / "review"
    export_for_review(processed_dir, review_dir)
    (review_dir / "grain0.jpg").unlink()

    excluded = apply_review(processed_dir, review_dir, dry_run=True)
    assert len(excluded) == 1
    assert (processed_dir / "grain0.npz").exists()
    assert not (tmp_path / "foo_excluded").exists()
