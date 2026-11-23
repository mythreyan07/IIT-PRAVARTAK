"""Tests for image_normalization.py"""

import numpy as np
import pytest

from backend.preprocessing.image_normalization import (
    NormalizationConfig,
    apply_inference_normalization,
    apply_training_normalization,
    convert_image_format,
    normalize_image,
    resize_image,
    transform_annotations,
    transform_bounding_box,
    transform_polygon,
    transform_segmentation_mask,
)


def _make_rgb(h=128, w=128, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# resize_image
# ---------------------------------------------------------------------------


class TestResizeImage:
    def test_basic_resize(self):
        img = _make_rgb(256, 256)
        resized, record = resize_image(img, (128, 128))
        assert resized.shape == (128, 128, 3)
        assert record["type"] == "resize"
        assert record["scale_x"] == pytest.approx(0.5)
        assert record["scale_y"] == pytest.approx(0.5)

    def test_upscale(self):
        img = _make_rgb(64, 64)
        resized, record = resize_image(img, (128, 128))
        assert resized.shape == (128, 128, 3)
        assert record["scale_x"] == pytest.approx(2.0)

    def test_no_resize_identity(self):
        img = _make_rgb(128, 128)
        resized, record = resize_image(img, (128, 128))
        assert resized.shape == (128, 128, 3)
        assert record["scale_x"] == pytest.approx(1.0)

    def test_transform_record_has_required_keys(self):
        img = _make_rgb()
        _, record = resize_image(img, (64, 64))
        assert {"type", "original_hw", "target_hw", "scale_x", "scale_y"} <= record.keys()


# ---------------------------------------------------------------------------
# normalize_image
# ---------------------------------------------------------------------------


class TestNormalizeImage:
    def test_uint8_scaled_to_01(self):
        img = _make_rgb()
        config = NormalizationConfig(normalize_range=(0.0, 1.0), mean=None, std=None)
        normed = normalize_image(img, config)
        assert normed.dtype == np.float32
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0

    def test_mean_std_applied(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        config = NormalizationConfig(
            normalize_range=(0.0, 1.0),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
        normed = normalize_image(img, config)
        # 128/255 ≈ 0.502; (0.502 - 0.5) / 0.5 ≈ 0.004
        assert abs(float(normed[0, 0, 0])) < 0.1

    def test_no_mean_std_no_error(self):
        img = _make_rgb()
        config = NormalizationConfig(mean=None, std=None)
        normed = normalize_image(img, config)
        assert normed.dtype == np.float32

    def test_grayscale_no_mean_std(self, gray_image):
        config = NormalizationConfig(mean=None, std=None)
        normed = normalize_image(gray_image, config)
        assert normed.ndim == 2


# ---------------------------------------------------------------------------
# convert_image_format
# ---------------------------------------------------------------------------


class TestConvertImageFormat:
    def test_rgb_to_bgr(self):
        img = _make_rgb()
        bgr = convert_image_format(img, "BGR")
        assert bgr.shape == img.shape
        # Check channel swap
        assert not np.array_equal(img[:, :, 0], bgr[:, :, 0])

    def test_rgb_to_gray(self):
        img = _make_rgb()
        gray = convert_image_format(img, "GRAY")
        assert gray.ndim == 2

    def test_rgb_to_rgba(self):
        img = _make_rgb()
        rgba = convert_image_format(img, "RGBA")
        assert rgba.shape[2] == 4

    def test_rgb_to_rgb_identity(self):
        img = _make_rgb()
        same = convert_image_format(img, "RGB")
        assert np.array_equal(img, same)

    def test_unknown_format_raises(self):
        img = _make_rgb()
        with pytest.raises(ValueError, match="Unknown target"):
            convert_image_format(img, "XYZ")


# ---------------------------------------------------------------------------
# apply_training_normalization
# ---------------------------------------------------------------------------


class TestApplyTrainingNormalization:
    def test_output_shape_matches_target(self):
        img = _make_rgb(256, 256)
        config = NormalizationConfig(target_size=(128, 128))
        normed, record = apply_training_normalization(img, config)
        assert normed.shape[:2] == (128, 128)

    def test_record_has_stage(self):
        img = _make_rgb()
        _, record = apply_training_normalization(img)
        assert record["stage"] == "training"

    def test_float32_output(self):
        img = _make_rgb()
        normed, _ = apply_training_normalization(img)
        assert normed.dtype == np.float32


# ---------------------------------------------------------------------------
# apply_inference_normalization
# ---------------------------------------------------------------------------


class TestApplyInferenceNormalization:
    def test_same_as_training_except_stage(self):
        img = _make_rgb()
        config = NormalizationConfig(target_size=(64, 64))
        t_img, t_rec = apply_training_normalization(img, config)
        i_img, i_rec = apply_inference_normalization(img, config)
        assert np.allclose(t_img, i_img)
        assert i_rec["stage"] == "inference"
        assert t_rec["stage"] == "training"


# ---------------------------------------------------------------------------
# Annotation coordinate preservation
# ---------------------------------------------------------------------------


class TestPolygonTransform:
    def test_scale_up(self):
        poly = [(10.0, 20.0), (30.0, 40.0)]
        result = transform_polygon(poly, scale_x=2.0, scale_y=3.0)
        assert result == [(20.0, 60.0), (60.0, 120.0)]

    def test_scale_identity(self):
        poly = [(5.0, 15.0), (25.0, 35.0)]
        result = transform_polygon(poly, scale_x=1.0, scale_y=1.0)
        assert result == poly


class TestBoundingBoxTransform:
    def test_scale(self):
        bbox = (10.0, 20.0, 50.0, 80.0)
        result = transform_bounding_box(bbox, scale_x=0.5, scale_y=0.5)
        assert result == pytest.approx((5.0, 10.0, 25.0, 40.0))


class TestSegmentationMaskTransform:
    def test_resize_nearest_neighbour(self):
        mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        resized = transform_segmentation_mask(mask, (4, 4))
        assert resized.shape == (4, 4)
        # Verify nearest-neighbour — no new integer values introduced
        assert set(np.unique(resized)).issubset({0, 1, 2, 3})

    def test_categorical_values_preserved(self):
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[32:96, 32:96] = 5  # building class
        resized = transform_segmentation_mask(mask, (64, 64))
        assert 5 in np.unique(resized)


class TestTransformAnnotations:
    def test_polygon_transformed(self):
        record = {
            "type": "resize",
            "original_hw": [256, 256],
            "target_hw": [128, 128],
            "scale_x": 0.5,
            "scale_y": 0.5,
        }
        anns = [{"polygon": [(10.0, 20.0), (30.0, 40.0)]}]
        result = transform_annotations(anns, record)
        assert result[0]["polygon"] == [(5.0, 10.0), (15.0, 20.0)]

    def test_bbox_transformed(self):
        record = {
            "type": "resize",
            "original_hw": [256, 256],
            "target_hw": [128, 128],
            "scale_x": 0.5,
            "scale_y": 0.5,
        }
        anns = [{"bbox": [0, 0, 100, 100]}]
        result = transform_annotations(anns, record)
        assert result[0]["bbox"] == pytest.approx([0, 0, 50, 50])

    def test_mask_transformed(self):
        record = {
            "type": "resize",
            "original_hw": [128, 128],
            "target_hw": [64, 64],
            "scale_x": 0.5,
            "scale_y": 0.5,
        }
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[32:96, 32:96] = 1
        anns = [{"mask": mask}]
        result = transform_annotations(anns, record)
        assert result[0]["mask"].shape == (64, 64)

    def test_unsupported_transform_raises(self):
        anns = [{"polygon": [(0, 0)]}]
        with pytest.raises(ValueError, match="Only 'resize'"):
            transform_annotations(anns, {"type": "rotate"})
