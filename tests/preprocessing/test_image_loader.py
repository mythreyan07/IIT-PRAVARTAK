"""Tests for image_loader.py"""

import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.preprocessing.image_loader import (
    ImageLoadError,
    convert_colour_space,
    load_image,
    preserve_original_image,
    save_processed_image,
    validate_colour_channels,
    validate_file_format,
    validate_image_dimensions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_image(tmp_dir: str, filename: str, image: np.ndarray) -> str:
    path = os.path.join(tmp_dir, filename)
    # Convert RGB → BGR for cv2.imwrite
    if image.ndim == 3 and image.shape[2] == 3:
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(path, image)
    return path


def _make_rgb(h=64, w=64) -> np.ndarray:
    rng = np.random.default_rng(99)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# validate_file_format
# ---------------------------------------------------------------------------


class TestValidateFileFormat:
    def test_valid_jpg(self, tmp_path):
        p = tmp_path / "test.jpg"
        p.write_bytes(b"\xff\xd8\xff")  # fake JPEG header
        result = validate_file_format(str(p))
        assert result == p

    def test_valid_png(self, tmp_path):
        p = tmp_path / "test.png"
        p.write_bytes(b"\x89PNG")
        validate_file_format(str(p))

    def test_unsupported_extension(self, tmp_path):
        p = tmp_path / "test.bmp"
        p.write_bytes(b"BM")
        with pytest.raises(ImageLoadError, match="Unsupported file format"):
            validate_file_format(str(p))

    def test_missing_file(self, tmp_path):
        with pytest.raises(ImageLoadError, match="File not found"):
            validate_file_format(str(tmp_path / "nonexistent.jpg"))

    def test_directory_path(self, tmp_path):
        with pytest.raises(ImageLoadError, match="not a file"):
            validate_file_format(str(tmp_path))


# ---------------------------------------------------------------------------
# validate_image_dimensions
# ---------------------------------------------------------------------------


class TestValidateImageDimensions:
    def test_valid_dimensions(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        validate_image_dimensions(img)  # Should not raise

    def test_too_small(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(ImageLoadError, match="too small"):
            validate_image_dimensions(img)

    def test_minimum_exact(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        validate_image_dimensions(img)  # Should not raise


# ---------------------------------------------------------------------------
# validate_colour_channels
# ---------------------------------------------------------------------------


class TestValidateColourChannels:
    def test_rgb(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        validate_colour_channels(img)

    def test_rgba(self):
        img = np.zeros((64, 64, 4), dtype=np.uint8)
        validate_colour_channels(img)

    def test_grayscale(self):
        img = np.zeros((64, 64), dtype=np.uint8)
        validate_colour_channels(img)

    def test_invalid_channels(self):
        img = np.zeros((64, 64, 2), dtype=np.uint8)
        with pytest.raises(ImageLoadError, match="channel count"):
            validate_colour_channels(img)


# ---------------------------------------------------------------------------
# convert_colour_space
# ---------------------------------------------------------------------------


class TestConvertColourSpace:
    def _bgr_image(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 0] = 100  # B
        img[:, :, 1] = 150  # G
        img[:, :, 2] = 200  # R
        return img

    def test_bgr_to_rgb(self):
        bgr = self._bgr_image()
        rgb = convert_colour_space(bgr, "RGB")
        assert rgb.shape == bgr.shape
        assert int(rgb[0, 0, 0]) == 200  # R was B in BGR

    def test_grayscale_to_rgb(self):
        gray = np.full((64, 64), 128, dtype=np.uint8)
        rgb = convert_colour_space(gray, "RGB")
        assert rgb.shape == (64, 64, 3)
        assert rgb[0, 0, 0] == 128

    def test_rgb_to_gray(self):
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[:, :] = [100, 100, 100]
        gray = convert_colour_space(rgb, "GRAY")
        assert gray.ndim == 2

    def test_unknown_target(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ImageLoadError, match="Unknown target"):
            convert_colour_space(img, "XYZ")


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------


class TestLoadImage:
    def test_load_valid_jpg(self, tmp_path):
        img_rgb = _make_rgb(128, 128)
        path = _write_image(str(tmp_path), "test.jpg", img_rgb)
        loaded, meta = load_image(path)
        assert loaded.shape == (128, 128, 3)
        assert meta["colour_space"] == "RGB"
        assert meta["file_name"] == "test.jpg"

    def test_load_valid_png(self, tmp_path):
        img_rgb = _make_rgb(64, 64)
        path = _write_image(str(tmp_path), "test.png", img_rgb)
        loaded, meta = load_image(path)
        assert loaded.shape == (64, 64, 3)

    def test_load_grayscale(self, tmp_path):
        gray = np.full((64, 64), 100, dtype=np.uint8)
        p = str(tmp_path / "gray.png")
        cv2.imwrite(p, gray)
        loaded, meta = load_image(p, colour_space="GRAY")
        assert loaded.ndim == 2

    def test_corrupted_file_rejected(self, tmp_path):
        p = tmp_path / "corrupt.jpg"
        p.write_bytes(b"not an image at all")
        with pytest.raises(ImageLoadError):
            load_image(str(p))

    def test_unsupported_format_rejected(self, tmp_path):
        p = tmp_path / "image.gif"
        p.write_bytes(b"GIF89a")
        with pytest.raises(ImageLoadError, match="Unsupported file format"):
            load_image(str(p))


# ---------------------------------------------------------------------------
# preserve_original_image
# ---------------------------------------------------------------------------


class TestPreserveOriginalImage:
    def test_copies_to_dest(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        dest_dir = tmp_path / "original" / "before"
        img = _make_rgb()
        src_path = _write_image(str(src_dir), "before.jpg", img)
        result = preserve_original_image(src_path, str(dest_dir), "PAIR_001", "before")
        assert Path(result).exists()
        # Source must still exist (not moved)
        assert Path(src_path).exists()

    def test_does_not_modify_source(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        img = _make_rgb()
        src_path = _write_image(str(src_dir), "before.jpg", img)
        original_bytes = Path(src_path).read_bytes()
        dest_dir = tmp_path / "original"
        preserve_original_image(src_path, str(dest_dir), "PAIR_002", "before")
        assert Path(src_path).read_bytes() == original_bytes


# ---------------------------------------------------------------------------
# save_processed_image
# ---------------------------------------------------------------------------


class TestSaveProcessedImage:
    def test_saves_png(self, tmp_path):
        img = _make_rgb()
        dest = save_processed_image(img, str(tmp_path), "PAIR_001", "before")
        assert Path(dest).exists()
        assert dest.suffix == ".png"

    def test_roundtrip_readable(self, tmp_path):
        img = _make_rgb(64, 64)
        dest = save_processed_image(img, str(tmp_path), "PAIR_001", "after")
        loaded = cv2.imread(str(dest))
        assert loaded is not None
        assert loaded.shape == (64, 64, 3)
