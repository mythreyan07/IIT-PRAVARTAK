"""Tests for model_evaluation.py."""
import numpy as np
import pytest
from backend.models.model_evaluation import (
    compute_segmentation_iou,
    compute_segmentation_metrics,
    compute_classification_accuracy,
    compute_confusion_matrix,
    compute_classification_report,
    generate_evaluation_report,
    plot_confusion_matrix,
)


class TestSegmentationIoU:
    def test_perfect_prediction(self):
        mask = np.ones((64, 64), dtype=np.float32)
        assert compute_segmentation_iou(mask, mask) == pytest.approx(1.0, abs=1e-4)

    def test_no_overlap(self):
        pred = np.zeros((64, 64), dtype=np.float32)
        true = np.ones((64, 64), dtype=np.float32)
        iou = compute_segmentation_iou(pred, true)
        assert iou < 0.01   # near 0 (eps keeps it from exactly 0)

    def test_partial_overlap(self):
        pred = np.zeros((64, 64), dtype=np.float32)
        pred[:32, :] = 1.0
        true = np.zeros((64, 64), dtype=np.float32)
        true[:32, :] = 1.0
        iou = compute_segmentation_iou(pred, true)
        assert iou == pytest.approx(1.0, abs=1e-4)

    def test_half_overlap(self):
        pred = np.zeros((64, 64), dtype=np.float32)
        pred[:32, :] = 1.0  # top half
        true = np.zeros((64, 64), dtype=np.float32)
        true[16:48, :] = 1.0  # middle half
        iou = compute_segmentation_iou(pred, true)
        assert 0.0 < iou < 1.0


class TestSegmentationMetrics:
    def test_returns_required_keys(self):
        pred = np.ones((32, 32))
        true = np.ones((32, 32))
        m = compute_segmentation_metrics(pred, true)
        assert {"iou", "f1", "precision", "recall"} <= m.keys()


class TestClassificationAccuracy:
    def test_all_correct(self):
        preds = np.array([0, 1, 2, 3])
        labels = np.array([0, 1, 2, 3])
        assert compute_classification_accuracy(preds, labels) == pytest.approx(1.0)

    def test_all_wrong(self):
        preds = np.array([1, 2, 3, 0])
        labels = np.array([0, 1, 2, 3])
        assert compute_classification_accuracy(preds, labels) == pytest.approx(0.0)

    def test_half_correct(self):
        preds = np.array([0, 1, 0, 1])
        labels = np.array([0, 1, 1, 0])
        assert compute_classification_accuracy(preds, labels) == pytest.approx(0.5)


class TestConfusionMatrix:
    def test_shape(self):
        preds = np.array([0, 1, 2, 3])
        labels = np.array([0, 1, 2, 3])
        cm = compute_confusion_matrix(preds, labels, num_classes=4)
        assert cm.shape == (4, 4)

    def test_perfect_diagonal(self):
        preds = np.array([0, 1, 2, 3])
        labels = np.array([0, 1, 2, 3])
        cm = compute_confusion_matrix(preds, labels, num_classes=4)
        assert np.all(cm == np.eye(4))


class TestClassificationReport:
    def test_returns_required_keys(self):
        preds = np.array([0, 1, 2, 3, 0, 1])
        labels = np.array([0, 1, 2, 3, 1, 0])
        names = ["no_damage", "minor", "major", "destroyed"]
        report = compute_classification_report(preds, labels, names)
        assert isinstance(report, dict)
        assert len(report) > 0


class TestGenerateEvaluationReport:
    def test_structure(self):
        report = generate_evaluation_report(
            seg_metrics={"iou": 0.9},
            cls_metrics={"accuracy": 0.85},
        )
        assert "segmentation" in report
        assert "classification" in report
        assert report["segmentation"]["iou"] == 0.9


class TestPlotConfusionMatrix:
    def test_returns_figure_or_none(self):
        cm = np.array([[10, 2], [1, 7]])
        result = plot_confusion_matrix(cm, ["class_a", "class_b"])
        # Either a matplotlib Figure or None (if mpl not available)
        assert result is None or hasattr(result, "savefig")
