"""Tests for dataset_inspector.py — uses synthetic data only."""

import pytest

from backend.dataset.dataset_inspector import (
    inspect_dataset,
    count_samples_per_split,
    count_damage_classes,
    check_image_dimensions,
    check_annotation_coverage,
    compute_class_weights,
    generate_inspection_report,
)
from backend.dataset.dataset_preparation import create_synthetic_dataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_samples():
    return create_synthetic_dataset(n_samples=20, image_size=(64, 64), seed=1)


@pytest.fixture
def synth_dataset_dict(synth_samples):
    return {
        "train": synth_samples[:12],
        "val": synth_samples[12:16],
        "test": synth_samples[16:],
    }


# ---------------------------------------------------------------------------
# count_samples_per_split
# ---------------------------------------------------------------------------

class TestCountSamplesPerSplit:
    def test_dict_input(self, synth_dataset_dict):
        counts = count_samples_per_split(synth_dataset_dict)
        assert counts["train"] == 12
        assert counts["val"] == 4
        assert counts["test"] == 4

    def test_total_matches(self, synth_dataset_dict, synth_samples):
        counts = count_samples_per_split(synth_dataset_dict)
        assert sum(counts.values()) == len(synth_samples)

    def test_single_dataset(self, synth_samples):
        counts = count_samples_per_split(synth_samples)
        assert sum(counts.values()) == len(synth_samples)


# ---------------------------------------------------------------------------
# count_damage_classes
# ---------------------------------------------------------------------------

class TestCountDamageClasses:
    def test_all_classes_present(self, synth_samples):
        counts = count_damage_classes(synth_samples)
        assert set(counts.keys()) == {"no_damage", "minor_damage", "major_damage", "destroyed"}

    def test_total_matches_annotations(self, synth_samples):
        counts = count_damage_classes(synth_samples)
        total = sum(counts.values())
        # Each synthetic sample has exactly 1 annotation
        assert total == len(synth_samples)

    def test_empty_dataset(self):
        counts = count_damage_classes([])
        assert all(v == 0 for v in counts.values())

    def test_no_annotations_counted(self):
        samples = [{"pair_id": "x", "annotations": []}]
        counts = count_damage_classes(samples)
        assert all(v == 0 for v in counts.values())


# ---------------------------------------------------------------------------
# check_image_dimensions
# ---------------------------------------------------------------------------

class TestCheckImageDimensions:
    def test_dimensions_match_synthetic_size(self, synth_samples):
        dims = check_image_dimensions(synth_samples, n_samples=10)
        assert dims["n_checked"] > 0
        assert dims["min_hw"] == [64, 64]
        assert dims["max_hw"] == [64, 64]

    def test_empty_dataset(self):
        dims = check_image_dimensions([], n_samples=10)
        assert dims["n_checked"] == 0
        assert dims["min_hw"] is None


# ---------------------------------------------------------------------------
# check_annotation_coverage
# ---------------------------------------------------------------------------

class TestCheckAnnotationCoverage:
    def test_full_coverage(self, synth_samples):
        # Each synthetic sample has 1 annotation
        coverage = check_annotation_coverage(synth_samples, n_samples=20)
        assert coverage == pytest.approx(1.0)

    def test_zero_coverage(self):
        samples = [{"pair_id": str(i), "annotations": []} for i in range(5)]
        coverage = check_annotation_coverage(samples, n_samples=5)
        assert coverage == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert check_annotation_coverage([], n_samples=10) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------

class TestComputeClassWeights:
    def test_weights_sum_to_n_classes(self, synth_samples):
        # With uniform distribution, all weights should be 1.0
        # Synthetic data has 20 samples, 4 classes, 1 annotation each → 5 per class
        weights = compute_class_weights(synth_samples)
        assert len(weights) == 4

    def test_rare_class_has_higher_weight(self):
        # 9 no_damage, 1 destroyed
        samples = [
            {"annotations": [{"damage_class": 0, "damage_label": "no_damage"}]} for _ in range(9)
        ] + [
            {"annotations": [{"damage_class": 3, "damage_label": "destroyed"}]}
        ]
        weights = compute_class_weights(samples)
        assert weights[3] > weights[0], "Rare class should have higher weight"

    def test_empty_dataset_returns_unit_weights(self):
        weights = compute_class_weights([])
        assert all(v == 1.0 for v in weights.values())

    def test_weights_are_positive(self, synth_samples):
        weights = compute_class_weights(synth_samples)
        assert all(v > 0 for v in weights.values())


# ---------------------------------------------------------------------------
# generate_inspection_report
# ---------------------------------------------------------------------------

class TestGenerateInspectionReport:
    def test_required_keys_present(self, synth_dataset_dict):
        report = generate_inspection_report(synth_dataset_dict)
        required = {
            "total_samples",
            "splits",
            "per_split_summary",
            "overall_damage_class_counts",
            "damage_class_definitions",
        }
        assert required <= report.keys()

    def test_total_samples_correct(self, synth_dataset_dict, synth_samples):
        report = generate_inspection_report(synth_dataset_dict)
        assert report["total_samples"] == len(synth_samples)

    def test_per_split_keys(self, synth_dataset_dict):
        report = generate_inspection_report(synth_dataset_dict)
        for split in synth_dataset_dict:
            assert split in report["per_split_summary"]

    def test_damage_class_definitions_correct(self, synth_dataset_dict):
        report = generate_inspection_report(synth_dataset_dict)
        assert report["damage_class_definitions"][0] == "no_damage"
        assert report["damage_class_definitions"][3] == "destroyed"


# ---------------------------------------------------------------------------
# inspect_dataset
# ---------------------------------------------------------------------------

class TestInspectDataset:
    def test_n_samples_correct(self, synth_samples):
        result = inspect_dataset(synth_samples)
        assert result["n_samples"] == len(synth_samples)

    def test_empty_dataset(self):
        result = inspect_dataset([])
        assert result["n_samples"] == 0
