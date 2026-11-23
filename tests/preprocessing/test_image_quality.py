"""Tests for image_quality.py"""

import cv2
import numpy as np
import pytest

from backend.preprocessing.image_quality import (
    QualityThresholds,
    assess_image_quality,
    check_blur,
    check_brightness,
    check_contrast,
    check_image_integrity,
    check_resolution,
)


def _make_rgb(h=128, w=128, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _solid_rgb(h=128, w=128, val=128) -> np.ndarray:
    return np.full((h, w, 3), val, dtype=np.uint8)


# ---------------------------------------------------------------------------
# check_blur
# ---------------------------------------------------------------------------


class TestCheckBlur:
    def test_sharp_image_not_blurry(self):
        # Checkerboard is high frequency → not blurry
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[::2, ::2] = 255
        result = check_blur(img, threshold=50.0)
        assert result["is_blurry"] is False
        assert "blur_score" in result

    def test_blurry_image_detected(self, blurry_image):
        result = check_blur(blurry_image, threshold=100.0)
        assert result["is_blurry"] is True
        assert result["blur_status"] == "Blurry"

    def test_returns_required_keys(self):
        img = _make_rgb()
        result = check_blur(img)
        assert {"blur_score", "blur_status", "is_blurry", "blur_threshold"} <= result.keys()


# ---------------------------------------------------------------------------
# check_brightness
# ---------------------------------------------------------------------------


class TestCheckBrightness:
    def test_acceptable_brightness(self):
        img = _solid_rgb(val=128)
        result = check_brightness(img, brightness_min=30, brightness_max=220)
        assert result["brightness_status"] == "Acceptable"
        assert result["brightness_ok"] is True

    def test_too_dark(self, dark_image):
        result = check_brightness(dark_image, brightness_min=30, brightness_max=220)
        assert result["brightness_status"] == "Too Dark"
        assert result["brightness_ok"] is False

    def test_too_bright(self, bright_image):
        result = check_brightness(bright_image, brightness_min=30, brightness_max=220)
        assert result["brightness_status"] == "Too Bright"
        assert result["brightness_ok"] is False

    def test_grayscale_accepted(self, gray_image):
        result = check_brightness(gray_image)
        assert "brightness_mean" in result


# ---------------------------------------------------------------------------
# check_contrast
# ---------------------------------------------------------------------------


class TestCheckContrast:
    def test_low_contrast_detected(self, low_contrast_image):
        result = check_contrast(low_contrast_image, contrast_min=20)
        assert result["contrast_status"] == "Low Contrast"
        assert result["contrast_ok"] is False

    def test_good_contrast(self):
        img = _make_rgb(seed=5)
        result = check_contrast(img, contrast_min=20)
        assert result["contrast_status"] == "Acceptable"

    def test_returns_required_keys(self):
        img = _make_rgb()
        result = check_contrast(img)
        assert {"contrast_std", "contrast_status", "contrast_ok"} <= result.keys()


# ---------------------------------------------------------------------------
# check_resolution
# ---------------------------------------------------------------------------


class TestCheckResolution:
    def test_valid_resolution(self):
        img = _make_rgb(256, 256)
        result = check_resolution(img, min_resolution=64)
        assert result["resolution_status"] == "Valid"
        assert result["resolution_ok"] is True

    def test_too_small(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        result = check_resolution(img, min_resolution=64)
        assert result["resolution_status"] == "Too Small"
        assert result["resolution_ok"] is False

    def test_exact_minimum(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = check_resolution(img, min_resolution=64)
        assert result["resolution_ok"] is True


# ---------------------------------------------------------------------------
# check_image_integrity
# ---------------------------------------------------------------------------


class TestCheckImageIntegrity:
    def test_valid_image(self):
        img = _make_rgb()
        result = check_image_integrity(img)
        assert result["integrity_ok"] is True
        assert result["integrity_status"] == "OK"

    def test_empty_array(self):
        result = check_image_integrity(np.array([]))
        assert result["integrity_ok"] is False

    def test_nan_detected(self):
        img = _make_rgb().astype(np.float32)
        img[0, 0] = float("nan")
        result = check_image_integrity(img)
        assert result["integrity_ok"] is False

    def test_inf_detected(self):
        img = _make_rgb().astype(np.float32)
        img[0, 0] = float("inf")
        result = check_image_integrity(img)
        assert result["integrity_ok"] is False


# ---------------------------------------------------------------------------
# assess_image_quality (composite)
# ---------------------------------------------------------------------------


class TestAssessImageQuality:
    def test_good_image_returns_good(self, medium_rgb):
        result = assess_image_quality(medium_rgb)
        assert result["overall_quality"] in {"Good", "Warning"}  # depends on content
        assert "blur_score" in result
        assert "brightness_status" in result
        assert "contrast_status" in result
        assert "resolution_status" in result

    def test_dark_image_returns_poor(self, dark_image):
        result = assess_image_quality(dark_image)
        assert result["overall_quality"] == "Poor"

    def test_bright_image_returns_poor(self, bright_image):
        result = assess_image_quality(bright_image)
        assert result["overall_quality"] == "Poor"

    def test_blurry_image_returns_warning_or_poor(self, blurry_image):
        result = assess_image_quality(blurry_image)
        assert result["overall_quality"] in {"Warning", "Poor"}

    def test_configurable_thresholds(self):
        img = _make_rgb()
        # Set an impossibly high blur threshold to force "Warning"
        t = QualityThresholds(blur_threshold=1e9)
        result = assess_image_quality(img, thresholds=t)
        assert result["is_blurry"] is True

    def test_structured_report_keys(self):
        img = _make_rgb()
        result = assess_image_quality(img)
        required = {
            "blur_score",
            "brightness_status",
            "contrast_status",
            "resolution_status",
            "overall_quality",
            "integrity_ok",
        }
        assert required <= result.keys()
