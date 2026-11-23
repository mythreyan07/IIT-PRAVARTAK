"""
Shared pytest fixtures for preprocessing tests.

All test images are generated synthetically — no external dataset is required.
"""

import numpy as np
import pytest


def _solid_rgb(h: int, w: int, r: int, g: int, b: int, dtype=np.uint8) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=dtype)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    return img


def _random_rgb(h: int, w: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _checkerboard(h: int, w: int, block: int = 32) -> np.ndarray:
    """Create a checkerboard pattern — has strong features for alignment tests."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if ((i // block) + (j // block)) % 2 == 0:
                img[i, j] = [200, 200, 200]
            else:
                img[i, j] = [50, 50, 50]
    return img


@pytest.fixture
def tiny_rgb():
    """128×128 random RGB image."""
    return _random_rgb(128, 128, seed=1)


@pytest.fixture
def medium_rgb():
    """512×512 random RGB image."""
    return _random_rgb(512, 512, seed=2)


@pytest.fixture
def gray_image():
    """128×128 grayscale image."""
    rng = np.random.default_rng(10)
    return rng.integers(0, 256, (128, 128), dtype=np.uint8)


@pytest.fixture
def rgba_image():
    """128×128 RGBA image (OpenCV BGRA layout)."""
    rng = np.random.default_rng(20)
    return rng.integers(0, 256, (128, 128, 4), dtype=np.uint8)


@pytest.fixture
def blurry_image():
    """512×512 heavily blurred image with low Laplacian variance."""
    base = _random_rgb(512, 512, seed=3)
    import cv2
    return cv2.GaussianBlur(base, (51, 51), 30)


@pytest.fixture
def dark_image():
    """128×128 very dark image (mean ~ 10)."""
    return _solid_rgb(128, 128, 10, 10, 10)


@pytest.fixture
def bright_image():
    """128×128 very bright image (mean ~ 245)."""
    return _solid_rgb(128, 128, 245, 245, 245)


@pytest.fixture
def low_contrast_image():
    """128×128 near-uniform grey image."""
    return _solid_rgb(128, 128, 120, 120, 120)


@pytest.fixture
def checkerboard_pair():
    """
    Returns (reference, slightly_shifted) checkerboard images for alignment tests.
    The shifted image is translated by 5 px right and 3 px down.
    """
    import cv2
    ref = _checkerboard(512, 512, block=32)
    M = np.float32([[1, 0, 5], [0, 1, 3]])
    shifted = cv2.warpAffine(ref, M, (512, 512))
    return ref, shifted


@pytest.fixture
def identical_pair():
    """Returns the same image twice — should be detected as already aligned."""
    img = _checkerboard(256, 256, block=16)
    return img, img.copy()
