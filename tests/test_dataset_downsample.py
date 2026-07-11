"""Tests for the TODO 5.3 resolution-ablation option in GrainDataset_ConvNeXt."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rgb_grains.data.dataset import GrainDataset_ConvNeXt


def _fake_batch(n=2, size=200):
    rng = np.random.RandomState(0)
    X = rng.randint(0, 5000, size=(n, size, size, 3)).astype(np.int16)
    return X


def test_downsample_kernel_1_is_noop():
    X = _fake_batch()
    ds_plain = GrainDataset_ConvNeXt(X, augment=False, crop_size=64, downsample_kernel=1)
    ds_k1 = GrainDataset_ConvNeXt(X, augment=False, crop_size=64, downsample_kernel=1, downsample_mode="mean")
    assert torch.allclose(ds_plain[0], ds_k1[0])


def test_downsample_preserves_shape():
    X = _fake_batch()
    ds = GrainDataset_ConvNeXt(X, augment=False, crop_size=64, downsample_kernel=4, downsample_mode="mean")
    out = ds[0]
    assert out.shape == (3, 64, 64)


def test_downsample_reduces_effective_resolution():
    X = _fake_batch(size=64)
    ds = GrainDataset_ConvNeXt(X, augment=False, crop_size=64, downsample_kernel=8, downsample_mode="mean")
    out = ds[0]
    # after 8x8 block pooling + nearest upsample, each 8x8 block should be constant
    block = out[:, :8, :8]
    assert torch.allclose(block, block[:, :1, :1].expand_as(block), atol=1e-5)


def test_invalid_downsample_mode_raises():
    X = _fake_batch()
    with pytest.raises(ValueError):
        GrainDataset_ConvNeXt(X, augment=False, crop_size=64, downsample_kernel=2, downsample_mode="bogus")
