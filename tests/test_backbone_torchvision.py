"""Tests for the optional torchvision ConvNeXt-Tiny backbone (--backbone-impl torchvision).

Skipped entirely when the `torchvision` extra isn't installed (it's optional,
see pyproject.toml's `torchvision` extra) -- the default `custom` backbone
path is unaffected either way.
"""
import pytest
import torch

pytest.importorskip("torchvision")

from rgb_grains.models.convnext import ConvNeXtTinyTorchvision, Model_ConvNeXt


def test_convnext_tiny_torchvision_forward_shapes():
    m = ConvNeXtTinyTorchvision(num_classes=8, pretrained=False)
    x = torch.randn(2, 3, 176, 176)
    feats = m.forward_features(x)
    assert feats.shape == (2, 768)
    logits = m(x)
    assert logits.shape == (2, 8)


def test_model_convnext_selects_torchvision_backbone(tmp_path):
    model = Model_ConvNeXt(
        config={"pretrained": False, "backbone_impl": "torchvision", "expe": "smoke", "max_ep": 1},
        restricted_classes=list(range(8)),
        base_dir=str(tmp_path),
    )
    assert isinstance(model.net, ConvNeXtTinyTorchvision)
    assert model.backbone_impl == "torchvision"


def test_model_convnext_rejects_invalid_backbone_impl(tmp_path):
    with pytest.raises(ValueError, match="backbone_impl"):
        Model_ConvNeXt(
            config={"pretrained": False, "backbone_impl": "bogus", "expe": "smoke2"},
            restricted_classes=list(range(8)),
            base_dir=str(tmp_path),
        )
