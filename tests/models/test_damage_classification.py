"""Tests for damage_classification.py — CPU only, no dataset downloads."""
import pytest
import torch
import numpy as np
from backend.models.damage_classification import (
    SiameseDamageClassifier,
    DamageClassificationModel,
    DamageClassifierConfig,
)


def _rand_batch(b=2, c=3, h=128, w=128):
    return torch.randn(b, c, h, w)


class TestSiameseDamageClassifier:
    def test_forward_shape(self):
        cfg = DamageClassifierConfig(device="cpu", pretrained=False)
        model = SiameseDamageClassifier(cfg)
        before = _rand_batch()
        after = _rand_batch()
        logits = model(before, after)
        assert logits.shape == (2, 4)

    def test_different_batch_sizes(self):
        cfg = DamageClassifierConfig(device="cpu", pretrained=False)
        model = SiameseDamageClassifier(cfg)
        for b in (1, 4):
            out = model(_rand_batch(b=b), _rand_batch(b=b))
            assert out.shape == (b, 4)


class TestDamageClassificationModel:
    @pytest.fixture
    def model(self):
        cfg = DamageClassifierConfig(device="cpu", pretrained=False)
        return DamageClassificationModel(cfg)

    def test_predict_class_range(self, model):
        pred = model.predict(_rand_batch(), _rand_batch())
        assert pred.shape == (2,)
        assert all(0 <= int(v) <= 3 for v in pred)

    def test_predict_proba_sums_to_one(self, model):
        proba = model.predict_proba(_rand_batch(), _rand_batch())
        assert proba.shape == (2, 4)
        sums = proba.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_get_class_names(self, model):
        names = model.get_class_names()
        assert len(names) == 4
        assert "no_damage" in names
        assert "destroyed" in names

    def test_save_load_checkpoint(self, model, tmp_path):
        ckpt = str(tmp_path / "cls.pt")
        model.save_checkpoint(ckpt)
        cfg2 = DamageClassifierConfig(checkpoint_path=ckpt, device="cpu", pretrained=False)
        m2 = DamageClassificationModel(cfg2)
        b, a = _rand_batch(), _rand_batch()
        assert torch.allclose(model.predict(b, a).float(), m2.predict(b, a).float())

    def test_predict_building_numpy(self, model):
        before_np = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        after_np = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        bbox = (10, 10, 110, 110)
        result = model.predict_building(before_np, after_np, bbox)
        assert isinstance(result, int)
        assert 0 <= result <= 3
