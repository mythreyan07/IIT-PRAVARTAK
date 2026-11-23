"""Tests for dataset_preparation.py — no HF downloads required."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.dataset.dataset_preparation import (
    DatasetConfig,
    XBDDataset,
    create_synthetic_dataset,
    create_data_loaders,
    prepare_dataset,
    save_split_manifest,
    validate_dataset_split,
    xbd_collate_fn,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_samples():
    return create_synthetic_dataset(n_samples=40, image_size=(64, 64), seed=7)


@pytest.fixture
def default_config():
    return DatasetConfig(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=4,
        num_workers=0,
        image_size=(64, 64),
        seed=42,
        use_weighted_sampling=False,  # deterministic for tests
    )


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------

class TestDatasetConfig:
    def test_valid_config(self):
        cfg = DatasetConfig()  # default ratios sum to 1.0
        assert cfg.train_ratio + cfg.val_ratio + cfg.test_ratio == pytest.approx(1.0)

    def test_invalid_ratios_raise(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            DatasetConfig(train_ratio=0.5, val_ratio=0.4, test_ratio=0.4)


# ---------------------------------------------------------------------------
# create_synthetic_dataset
# ---------------------------------------------------------------------------

class TestCreateSyntheticDataset:
    def test_correct_length(self):
        samples = create_synthetic_dataset(n_samples=15)
        assert len(samples) == 15

    def test_sample_structure(self):
        samples = create_synthetic_dataset(n_samples=5, image_size=(32, 32))
        s = samples[0]
        assert "before" in s
        assert "after" in s
        assert "pair_id" in s
        assert "annotations" in s

    def test_numpy_images(self):
        samples = create_synthetic_dataset(n_samples=5, image_size=(32, 32))
        assert isinstance(samples[0]["before"], np.ndarray)
        assert samples[0]["before"].shape == (32, 32, 3)

    def test_damage_classes_cycle(self):
        samples = create_synthetic_dataset(n_samples=8, n_damage_classes=4)
        classes = [s["annotations"][0]["damage_class"] for s in samples]
        # Should cycle 0,1,2,3,0,1,2,3
        assert set(classes) == {0, 1, 2, 3}

    def test_different_seeds_differ(self):
        s1 = create_synthetic_dataset(n_samples=5, seed=1)
        s2 = create_synthetic_dataset(n_samples=5, seed=2)
        assert not np.array_equal(s1[0]["before"], s2[0]["before"])

    def test_no_hf_import_needed(self):
        # This should work even if 'datasets' is not installed
        import sys
        datasets_saved = sys.modules.get("datasets")
        # Remove temporarily if it exists
        if datasets_saved is not None:
            del sys.modules["datasets"]
        try:
            result = create_synthetic_dataset(n_samples=3)
            assert len(result) == 3
        finally:
            if datasets_saved is not None:
                sys.modules["datasets"] = datasets_saved


# ---------------------------------------------------------------------------
# prepare_dataset
# ---------------------------------------------------------------------------

class TestPrepareDataset:
    def test_split_sizes_approximate(self, synth_samples, default_config):
        split = prepare_dataset(synth_samples, default_config)
        n = len(synth_samples)
        assert len(split["train"]) == int(n * 0.7)
        assert len(split["val"]) == int(n * 0.15)
        total = sum(len(v) for v in split.values())
        assert total == n

    def test_no_overlap_between_splits(self, synth_samples, default_config):
        split = prepare_dataset(synth_samples, default_config)
        train_ids = {s["pair_id"] for s in split["train"]}
        val_ids = {s["pair_id"] for s in split["val"]}
        test_ids = {s["pair_id"] for s in split["test"]}
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_all_splits_present(self, synth_samples, default_config):
        split = prepare_dataset(synth_samples, default_config)
        assert set(split.keys()) == {"train", "val", "test"}

    def test_seed_reproducibility(self, synth_samples):
        cfg1 = DatasetConfig(seed=99, image_size=(64, 64))
        cfg2 = DatasetConfig(seed=99, image_size=(64, 64))
        s1 = prepare_dataset(synth_samples, cfg1)
        s2 = prepare_dataset(synth_samples, cfg2)
        assert [x["pair_id"] for x in s1["train"]] == [x["pair_id"] for x in s2["train"]]


# ---------------------------------------------------------------------------
# XBDDataset
# ---------------------------------------------------------------------------

class TestXBDDataset:
    def test_len(self, synth_samples):
        ds = XBDDataset(synth_samples, image_size=(64, 64))
        assert len(ds) == len(synth_samples)

    def test_getitem_structure(self, synth_samples):
        ds = XBDDataset(synth_samples, image_size=(64, 64))
        item = ds[0]
        assert "before" in item
        assert "after" in item
        assert "label" in item
        assert "annotations" in item
        assert "pair_id" in item

    def test_getitem_tensor_shapes(self, synth_samples):
        ds = XBDDataset(synth_samples, image_size=(64, 64))
        item = ds[0]
        assert item["before"].shape == (3, 64, 64)
        assert item["after"].shape == (3, 64, 64)

    def test_tensor_dtype_float32(self, synth_samples):
        ds = XBDDataset(synth_samples, image_size=(64, 64))
        item = ds[0]
        assert item["before"].dtype == torch.float32
        assert item["after"].dtype == torch.float32

    def test_pixel_values_in_01(self, synth_samples):
        ds = XBDDataset(synth_samples, image_size=(64, 64))
        item = ds[0]
        assert item["before"].min() >= 0.0
        assert item["before"].max() <= 1.0

    def test_label_is_long_tensor(self, synth_samples):
        ds = XBDDataset(synth_samples, image_size=(64, 64))
        item = ds[0]
        assert item["label"].dtype == torch.long
        assert item["label"].ndim == 0  # scalar

    def test_resize_applied(self):
        samples = create_synthetic_dataset(n_samples=2, image_size=(128, 128))
        ds = XBDDataset(samples, image_size=(32, 32))
        item = ds[0]
        assert item["before"].shape == (3, 32, 32)


# ---------------------------------------------------------------------------
# xbd_collate_fn
# ---------------------------------------------------------------------------

class TestXbdCollateFn:
    def _make_batch(self, n=4, h=32, w=32):
        return [
            {
                "before": torch.zeros(3, h, w),
                "after": torch.zeros(3, h, w),
                "label": torch.tensor(i % 4),
                "annotations": [{"damage_class": i % 4}],
                "pair_id": f"P_{i}",
            }
            for i in range(n)
        ]

    def test_batches_tensors(self):
        batch = self._make_batch(4)
        result = xbd_collate_fn(batch)
        assert result["before"].shape == (4, 3, 32, 32)
        assert result["after"].shape == (4, 3, 32, 32)
        assert result["label"].shape == (4,)

    def test_annotations_as_list_of_lists(self):
        batch = self._make_batch(3)
        result = xbd_collate_fn(batch)
        assert isinstance(result["annotations"], list)
        assert len(result["annotations"]) == 3

    def test_pair_ids_as_list(self):
        batch = self._make_batch(3)
        result = xbd_collate_fn(batch)
        assert isinstance(result["pair_id"], list)
        assert len(result["pair_id"]) == 3


# ---------------------------------------------------------------------------
# create_data_loaders
# ---------------------------------------------------------------------------

class TestCreateDataLoaders:
    def test_all_splits_have_loaders(self, synth_samples, default_config):
        split = prepare_dataset(synth_samples, default_config)
        loaders = create_data_loaders(split, default_config)
        assert set(loaders.keys()) == {"train", "val", "test"}

    def test_loader_iterates(self, synth_samples, default_config):
        split = prepare_dataset(synth_samples, default_config)
        loaders = create_data_loaders(split, default_config)
        batch = next(iter(loaders["train"]))
        assert "before" in batch
        assert batch["before"].shape[0] == default_config.batch_size


# ---------------------------------------------------------------------------
# validate_dataset_split
# ---------------------------------------------------------------------------

class TestValidateDatasetSplit:
    def test_no_leakage_in_clean_split(self, synth_samples, default_config):
        split = prepare_dataset(synth_samples, default_config)
        report = validate_dataset_split(split)
        assert report["leakage_detected"] is False
        assert report["overlapping_pairs"] == []

    def test_detects_leakage(self, synth_samples):
        # Deliberately put same samples in train and test
        split = {
            "train": synth_samples[:5],
            "val": synth_samples[5:8],
            "test": synth_samples[:3],  # overlap with train
        }
        report = validate_dataset_split(split)
        assert report["leakage_detected"] is True
        assert len(report["overlapping_pairs"]) > 0

    def test_report_has_required_keys(self, synth_samples, default_config):
        split = prepare_dataset(synth_samples, default_config)
        report = validate_dataset_split(split)
        assert {"leakage_detected", "overlapping_pairs", "split_sizes", "split_class_distribution"} <= report.keys()


# ---------------------------------------------------------------------------
# save_split_manifest
# ---------------------------------------------------------------------------

class TestSaveSplitManifest:
    def test_saves_json(self, synth_samples, default_config, tmp_path):
        split = prepare_dataset(synth_samples, default_config)
        out = str(tmp_path / "manifest.json")
        save_split_manifest(split, out)
        assert Path(out).exists()
        data = json.loads(Path(out).read_text())
        assert "train" in data
        assert "pair_ids" in data["train"]

    def test_pair_ids_match(self, synth_samples, default_config, tmp_path):
        split = prepare_dataset(synth_samples, default_config)
        out = str(tmp_path / "manifest.json")
        save_split_manifest(split, out)
        data = json.loads(Path(out).read_text())
        assert data["train"]["n_samples"] == len(split["train"])
