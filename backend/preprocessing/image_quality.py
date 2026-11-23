"""
Image quality assessment: blur, brightness, contrast, resolution, integrity.

Returns structured quality reports; never hard-rejects images based on a
single hard-coded threshold — all thresholds are configurable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QualityThresholds:
    """All quality thresholds in one place for easy override."""

    # Laplacian variance: below this is considered blurry
    blur_threshold: float = 50.0

    # Brightness (mean pixel value 0-255)
    brightness_min: float = 30.0
    brightness_max: float = 220.0

    # Contrast (standard deviation of pixel values 0-255)
    contrast_min: float = 20.0

    # Minimum acceptable dimension (px)
    min_resolution: int = 64


DEFAULT_THRESHOLDS = QualityThresholds()


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def check_blur(
    image: np.ndarray,
    threshold: float = DEFAULT_THRESHOLDS.blur_threshold,
) -> Dict[str, Any]:
    """
    Estimate blur using Laplacian variance.

    Higher variance = sharper image.
    Returns {'blur_score': float, 'blur_status': str, 'is_blurry': bool}.
    """
    gray = _to_gray(image)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_blurry = score < threshold
    return {
        "blur_score": round(score, 4),
        "blur_threshold": threshold,
        "blur_status": "Blurry" if is_blurry else "Acceptable",
        "is_blurry": is_blurry,
    }


def check_brightness(
    image: np.ndarray,
    brightness_min: float = DEFAULT_THRESHOLDS.brightness_min,
    brightness_max: float = DEFAULT_THRESHOLDS.brightness_max,
) -> Dict[str, Any]:
    """
    Assess brightness using mean pixel intensity on the grayscale image.

    Returns brightness metrics and status.
    """
    gray = _to_gray(image)
    mean_val = float(gray.mean())
    if mean_val < brightness_min:
        status = "Too Dark"
    elif mean_val > brightness_max:
        status = "Too Bright"
    else:
        status = "Acceptable"
    return {
        "brightness_mean": round(mean_val, 4),
        "brightness_min_threshold": brightness_min,
        "brightness_max_threshold": brightness_max,
        "brightness_status": status,
        "brightness_ok": status == "Acceptable",
    }


def check_contrast(
    image: np.ndarray,
    contrast_min: float = DEFAULT_THRESHOLDS.contrast_min,
) -> Dict[str, Any]:
    """
    Assess contrast using the standard deviation of pixel intensities.

    Low standard deviation = low contrast.
    """
    gray = _to_gray(image)
    std_val = float(gray.std())
    # Also compute the full histogram spread as a secondary metric
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten()
    nonzero = np.where(hist > 0)[0]
    hist_range = int(nonzero[-1] - nonzero[0]) if len(nonzero) >= 2 else 0

    status = "Low Contrast" if std_val < contrast_min else "Acceptable"
    return {
        "contrast_std": round(std_val, 4),
        "contrast_hist_range": hist_range,
        "contrast_min_threshold": contrast_min,
        "contrast_status": status,
        "contrast_ok": status == "Acceptable",
    }


def check_resolution(
    image: np.ndarray,
    min_resolution: int = DEFAULT_THRESHOLDS.min_resolution,
) -> Dict[str, Any]:
    """
    Confirm that image dimensions meet the minimum resolution requirement.
    """
    h, w = image.shape[:2]
    ok = h >= min_resolution and w >= min_resolution
    return {
        "width": w,
        "height": h,
        "min_resolution_threshold": min_resolution,
        "resolution_status": "Valid" if ok else "Too Small",
        "resolution_ok": ok,
    }


def check_image_integrity(image: np.ndarray) -> Dict[str, Any]:
    """
    Lightweight integrity check: confirm the array is finite and non-empty.

    Does NOT re-read from disk — assumes the caller loaded the image with
    image_loader.load_image() which already attempted OpenCV decoding.
    """
    if image is None or image.size == 0:
        return {"integrity_status": "Empty", "integrity_ok": False}
    has_nan = bool(np.isnan(image.astype(np.float32)).any())
    has_inf = bool(np.isinf(image.astype(np.float32)).any())
    if has_nan or has_inf:
        return {
            "integrity_status": "Contains NaN/Inf",
            "integrity_ok": False,
            "has_nan": has_nan,
            "has_inf": has_inf,
        }
    return {
        "integrity_status": "OK",
        "integrity_ok": True,
        "has_nan": False,
        "has_inf": False,
    }


# ---------------------------------------------------------------------------
# Composite report
# ---------------------------------------------------------------------------


def assess_image_quality(
    image: np.ndarray,
    thresholds: QualityThresholds | None = None,
) -> Dict[str, Any]:
    """
    Run all quality checks and return a structured report.

    Example output::

        {
            "blur_score": 120.5,
            "brightness_status": "Acceptable",
            "contrast_status": "Acceptable",
            "resolution_status": "Valid",
            "overall_quality": "Good",
            ...
        }

    ``overall_quality`` is 'Good', 'Warning', or 'Poor':
    - 'Poor': integrity failure, resolution too small, or extreme brightness
    - 'Warning': blurry or low contrast
    - 'Good': all checks pass
    """
    t = thresholds or DEFAULT_THRESHOLDS

    integrity = check_image_integrity(image)
    resolution = check_resolution(image, min_resolution=t.min_resolution)
    blur = check_blur(image, threshold=t.blur_threshold)
    brightness = check_brightness(
        image,
        brightness_min=t.brightness_min,
        brightness_max=t.brightness_max,
    )
    contrast = check_contrast(image, contrast_min=t.contrast_min)

    # Determine overall quality
    if not integrity["integrity_ok"] or not resolution["resolution_ok"]:
        overall = "Poor"
    elif not brightness["brightness_ok"]:
        overall = "Poor"
    elif blur["is_blurry"] or not contrast["contrast_ok"]:
        overall = "Warning"
    else:
        overall = "Good"

    report: Dict[str, Any] = {
        **integrity,
        **resolution,
        **blur,
        **brightness,
        **contrast,
        "overall_quality": overall,
    }
    logger.debug("Quality assessment: overall=%s, blur=%.1f", overall, blur["blur_score"])
    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Return a uint8 single-channel grayscale array regardless of input format."""
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        raise ValueError(f"Cannot convert {image.shape[2]}-channel image to grayscale.")
    # Normalise to uint8 if float
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray
