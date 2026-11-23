"""Tests for building_segmentation.py — CPU only, no dataset downloads."""
import pytest
import torch
import numpy as np
from backend.models.building_segmentation import (
    UNet, BuildingSegmentationModel, SegmentationConfig,
)


def _rand_batch(b=2, c=3, h=64, w=64):
    return torch.randn(b, c, h, w)


class TestUNet:
    def test_forward_shape(self):
        model = UNet(num_classes=2)
        x = _rand_batch()
        out = model(x)
        assert out.shape == (2, 2, 64, 64)

    def test_forward_different_num_classes(self):
        model = UNet(num_classes=3)
        x = _rand_batch()
        out = model(x)
        assert out.shape[1] == 3

    def test_cpu_inference(self):
        model = UNet(num_classes=2).eval()
        x = _rand_batch(b=1)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2, 64, 64)


class TestBuildingSegmentationModel:
    def test_predict_returns_binary_mask(self):
        cfg = SegmentationConfig(num_classes=2, device="cpu")
        m = BuildingSegmentationModel(cfg)
        x = _rand_batch()
        pred = m.predict(x)
        assert pred.shape == (2, 64, 64)
        assert set(pred.unique().tolist()).issubset({0, 1})

    def test_predict_proba_shape(self):
        cfg = SegmentationConfig(device="cpu")
        m = BuildingSegmentationModel(cfg)
        x = _rand_batch()
        proba = m.predict_proba(x)
        assert proba.shape == (2, 1, 64, 64)
        assert float(proba.min()) >= 0.0
        assert float(proba.max()) <= 1.0

    def test_get_num_parameters(self):
        m = BuildingSegmentationModel()
        assert m.get_num_parameters() > 0

    def test_save_load_checkpoint(self, tmp_path):
        cfg = SegmentationConfig(device="cpu")
        m = BuildingSegmentationModel(cfg)
        ckpt = str(tmp_path / "seg.pt")
        m.save_checkpoint(ckpt)
        # Load into a fresh model
        cfg2 = SegmentationConfig(checkpoint_path=ckpt, device="cpu")
        m2 = BuildingSegmentationModel(cfg2)
        # Weights should be the same
        x = _rand_batch()
        assert torch.allclose(m.predict(x).float(), m2.predict(x).float())

    def test_cpu_device(self):
        cfg = SegmentationConfig(device="cpu")
        m = BuildingSegmentationModel(cfg)
        x = _rand_batch()
        pred = m.predict(x)
        assert pred.device.type == "cpu"
