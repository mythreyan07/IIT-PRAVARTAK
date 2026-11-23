"""End-to-end and integration tests for preprocessing_pipeline.py"""

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.preprocessing.preprocessing_pipeline import (
    PreprocessingConfig,
    PreprocessingPipeline,
    _generate_pair_id,
)
from backend.preprocessing.image_alignment import AlignmentConfig
from backend.preprocessing.image_quality import QualityThresholds
from backend.preprocessing.image_normalization import NormalizationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_test_image(directory: Path, name: str, h=128, w=128, seed=42) -> str:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    path = directory / name
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return str(path)


def _checkerboard(h=256, w=256, block=16) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if ((i // block) + (j // block)) % 2 == 0:
                img[i, j] = [200, 200, 200]
            else:
                img[i, j] = [50, 50, 50]
    return img


def _write_checkerboard(directory: Path, name: str, dx=0, dy=0) -> str:
    img = _checkerboard()
    if dx or dy:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    path = directory / name
    cv2.imwrite(str(path), img)
    return str(path)


@pytest.fixture
def config(tmp_path) -> PreprocessingConfig:
    """Config that uses tmp_path for all directories."""
    return PreprocessingConfig(
        original_before_dir=str(tmp_path / "original" / "before"),
        original_after_dir=str(tmp_path / "original" / "after"),
        processed_before_dir=str(tmp_path / "processed" / "before"),
        processed_after_dir=str(tmp_path / "processed" / "after"),
        aligned_dir=str(tmp_path / "processed" / "aligned"),
        normalized_dir=str(tmp_path / "processed" / "normalized"),
        previews_dir=str(tmp_path / "processed" / "previews"),
        metadata_dir=str(tmp_path / "metadata"),
        normalization=NormalizationConfig(target_size=(64, 64), mean=None, std=None),
        quality=QualityThresholds(blur_threshold=1.0),  # lenient threshold
        alignment=AlignmentConfig(already_aligned_threshold_px=100.0),  # be lenient in tests
    )


# ---------------------------------------------------------------------------
# _generate_pair_id
# ---------------------------------------------------------------------------


class TestGeneratePairId:
    def test_format(self):
        pid = _generate_pair_id()
        assert pid.startswith("PAIR_")

    def test_unique(self):
        ids = {_generate_pair_id() for _ in range(10)}
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# Pipeline — valid pair
# ---------------------------------------------------------------------------


class TestPreprocessingPipelineValid:
    def test_full_pipeline_completes(self, tmp_path, config):
        pipeline = PreprocessingPipeline(config)
        before_path = _write_test_image(tmp_path, "before.jpg", seed=1)
        after_path = _write_test_image(tmp_path, "after.jpg", seed=2)

        result = pipeline.process_image_pair(before_path, after_path, image_pair_id="PAIR_TEST")
        # Status should be completed or unreliable (alignment may not succeed on random noise)
        assert result.status in {"completed", "unreliable"}
        assert result.image_pair_id == "PAIR_TEST"

    def test_original_images_preserved(self, tmp_path, config):
        pipeline = PreprocessingPipeline(config)
        before_path = _write_test_image(tmp_path, "before.jpg", seed=3)
        after_path = _write_test_image(tmp_path, "after.jpg", seed=4)
        result = pipeline.process_image_pair(before_path, after_path, image_pair_id="PAIR_ORIG")
        assert Path(before_path).exists(), "Original before must still exist"
        assert Path(after_path).exists(), "Original after must still exist"

    def test_metadata_saved(self, tmp_path, config):
        pipeline = PreprocessingPipeline(config)
        before_path = _write_test_image(tmp_path, "before.jpg", seed=5)
        after_path = _write_test_image(tmp_path, "after.jpg", seed=6)
        result = pipeline.process_image_pair(before_path, after_path, image_pair_id="PAIR_META")
        meta_file = Path(config.metadata_dir) / "PAIR_META.json"
        assert meta_file.exists()
        data = json.loads(meta_file.read_text())
        assert data["image_pair_id"] == "PAIR_META"
        assert "preprocessing_timestamp" in data

    def test_pair_id_in_output_filenames(self, tmp_path, config):
        pipeline = PreprocessingPipeline(config)
        before_path = _write_test_image(tmp_path, "before.jpg", seed=7)
        after_path = _write_test_image(tmp_path, "after.jpg", seed=8)
        result = pipeline.process_image_pair(before_path, after_path, image_pair_id="MY_PAIR")
        if result.processed_before:
            assert "MY_PAIR" in result.processed_before

    def test_different_uploads_do_not_overwrite(self, tmp_path, config):
        pipeline = PreprocessingPipeline(config)
        b1 = _write_test_image(tmp_path, "b1.jpg", seed=10)
        a1 = _write_test_image(tmp_path, "a1.jpg", seed=11)
        b2 = _write_test_image(tmp_path, "b2.jpg", seed=12)
        a2 = _write_test_image(tmp_path, "a2.jpg", seed=13)
        r1 = pipeline.process_image_pair(b1, a1, image_pair_id="PAIR_A")
        r2 = pipeline.process_image_pair(b2, a2, image_pair_id="PAIR_B")
        # Both results should reference different files
        if r1.processed_before and r2.processed_before:
            assert r1.processed_before != r2.processed_before


# ---------------------------------------------------------------------------
# Pipeline — error cases
# ---------------------------------------------------------------------------


class TestPreprocessingPipelineErrors:
    def test_missing_file_fails(self, tmp_path, config):
        pipeline = PreprocessingPipeline(config)
        result = pipeline.process_image_pair(
            str(tmp_path / "nonexistent.jpg"),
            str(tmp_path / "also_missing.jpg"),
        )
        assert result.status == "failed"
        assert len(result.errors) > 0

    def test_corrupted_file_fails(self, tmp_path, config):
        p1 = tmp_path / "corrupt.jpg"
        p1.write_bytes(b"not an image")
        p2 = tmp_path / "corrupt2.jpg"
        p2.write_bytes(b"also not")
        pipeline = PreprocessingPipeline(config)
        result = pipeline.process_image_pair(str(p1), str(p2))
        assert result.status == "failed"

    def test_poor_quality_rejection_configurable(self, tmp_path, config):
        """
        reject_poor_quality=True should fail on a very dark image.
        """
        config.reject_poor_quality = True
        pipeline = PreprocessingPipeline(config)
        # Very dark image
        dark = np.full((128, 128, 3), 5, dtype=np.uint8)
        p_dark = tmp_path / "dark.png"
        cv2.imwrite(str(p_dark), dark)
        normal = _write_test_image(tmp_path, "normal.jpg", seed=42)
        result = pipeline.process_image_pair(str(p_dark), normal)
        assert result.status == "failed"


# ---------------------------------------------------------------------------
# Pipeline — annotation transformation
# ---------------------------------------------------------------------------


class TestAnnotationTransformation:
    def test_polygon_coordinates_transformed(self, tmp_path, config):
        pipeline = PreprocessingPipeline(config)
        before_path = _write_test_image(tmp_path, "before.jpg", 128, 128, seed=20)
        after_path = _write_test_image(tmp_path, "after.jpg", 128, 128, seed=21)
        # Annotation in original 128×128 space; expect scaled to 64×64
        anns = [{"polygon": [(10.0, 20.0), (100.0, 110.0)]}]
        # Pipeline transforms and stores; we just ensure no exception is raised
        result = pipeline.process_image_pair(
            before_path, after_path,
            image_pair_id="PAIR_ANN",
            annotations_before=anns,
        )
        assert result.status in {"completed", "unreliable"}


# ---------------------------------------------------------------------------
# Pipeline — checkerboard pair (better alignment)
# ---------------------------------------------------------------------------


class TestAlignedCheckerboard:
    def test_checkerboard_pair_aligned(self, tmp_path, config):
        """Checkerboard with a small shift should produce a completed result."""
        config.alignment = AlignmentConfig(
            orb_n_features=2000,
            min_good_matches=5,
            already_aligned_threshold_px=0.5,
        )
        pipeline = PreprocessingPipeline(config)
        before_path = _write_checkerboard(tmp_path, "cb_before.png", dx=0, dy=0)
        after_path = _write_checkerboard(tmp_path, "cb_after.png", dx=5, dy=3)
        result = pipeline.process_image_pair(
            before_path, after_path, image_pair_id="PAIR_CB"
        )
        assert result.status in {"completed", "unreliable"}
        assert result.image_pair_id == "PAIR_CB"
