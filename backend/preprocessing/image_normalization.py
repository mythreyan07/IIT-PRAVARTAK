"""
Image resizing, normalisation and format conversion.

Training, validation and inference all use the same NormalizationConfig so
preprocessing is consistent across all pipeline stages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NormalizationConfig:
    """Single source of truth for all normalisation parameters."""

    # Target spatial dimensions (H, W)
    target_size: Tuple[int, int] = (1024, 1024)

    # Channel ordering expected by the model
    channel_order: str = "RGB"  # 'RGB' or 'BGR'

    # Pixel value range after normalisation
    normalize_range: Tuple[float, float] = (0.0, 1.0)

    # ImageNet-style per-channel mean and std (applied AFTER scaling to 0-1)
    # Set to None to skip mean/std normalisation
    mean: Optional[Tuple[float, float, float]] = (0.485, 0.456, 0.406)
    std: Optional[Tuple[float, float, float]] = (0.229, 0.224, 0.225)

    # Interpolation method for resizing images (not masks)
    interpolation: int = cv2.INTER_LINEAR

    # Interpolation method for categorical segmentation masks
    mask_interpolation: int = cv2.INTER_NEAREST


DEFAULT_CONFIG = NormalizationConfig()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Resize image to target_size (H, W).

    Returns (resized_image, transform_record).
    The transform_record stores scale factors so annotations can be adapted.
    """
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size
    resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    transform_record = {
        "type": "resize",
        "original_hw": [orig_h, orig_w],
        "target_hw": [target_h, target_w],
        "scale_x": target_w / orig_w,
        "scale_y": target_h / orig_h,
    }
    return resized, transform_record


def normalize_image(
    image: np.ndarray,
    config: NormalizationConfig | None = None,
) -> np.ndarray:
    """
    Normalise a uint8 image to float32 within [normalize_range].

    If config.mean and config.std are set, also apply per-channel
    ImageNet-style normalisation: (x - mean) / std.
    """
    cfg = config or DEFAULT_CONFIG
    low, high = cfg.normalize_range

    if image.dtype == np.uint8:
        img_f = image.astype(np.float32) / 255.0
    else:
        img_f = image.astype(np.float32)
        # Clamp to [0,1] if not already there
        img_f = np.clip(img_f, 0.0, 1.0)

    # Rescale to requested range
    if (low, high) != (0.0, 1.0):
        img_f = img_f * (high - low) + low

    # Per-channel mean/std
    if cfg.mean is not None and cfg.std is not None and image.ndim == 3:
        mean = np.array(cfg.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(cfg.std, dtype=np.float32).reshape(1, 1, 3)
        img_f = (img_f - mean) / (std + 1e-8)

    return img_f


def convert_image_format(
    image: np.ndarray,
    target_format: str,
) -> np.ndarray:
    """
    Convert image to 'RGB', 'BGR', 'GRAY', or 'RGBA'.

    Assumes input is RGB (the pipeline's internal colour space).
    """
    target = target_format.upper()
    channels = 1 if image.ndim == 2 else image.shape[2]

    if target == "RGB":
        if channels == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        return image.copy() if channels == 3 else cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    elif target == "BGR":
        if channels == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif target == "GRAY":
        if channels == 1:
            return image.copy()
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    elif target == "RGBA":
        if channels == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
        return image.copy()
    raise ValueError(f"Unknown target format: '{target_format}'")


def apply_training_normalization(
    image: np.ndarray,
    config: NormalizationConfig | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Full normalisation for training/validation/test images.

    1. Resize to config.target_size.
    2. Ensure correct channel order.
    3. Normalise pixel values.

    Returns (processed_image, transform_record).
    """
    cfg = config or DEFAULT_CONFIG
    img, record = resize_image(image, cfg.target_size, cfg.interpolation)
    img = convert_image_format(img, cfg.channel_order)
    img = normalize_image(img, cfg)
    record["stage"] = "training"
    record["channel_order"] = cfg.channel_order
    record["normalize_range"] = list(cfg.normalize_range)
    return img, record


def apply_inference_normalization(
    image: np.ndarray,
    config: NormalizationConfig | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Full normalisation for uploaded inference images.

    Identical to training normalisation so preprocessing is consistent.
    """
    cfg = config or DEFAULT_CONFIG
    img, record = apply_training_normalization(image, cfg)
    record["stage"] = "inference"
    return img, record


# ---------------------------------------------------------------------------
# Annotation coordinate preservation
# ---------------------------------------------------------------------------


def transform_polygon(
    polygon: List[Tuple[float, float]],
    scale_x: float,
    scale_y: float,
) -> List[Tuple[float, float]]:
    """Scale a list of (x, y) polygon vertices."""
    return [(x * scale_x, y * scale_y) for x, y in polygon]


def transform_bounding_box(
    bbox: Tuple[float, float, float, float],
    scale_x: float,
    scale_y: float,
) -> Tuple[float, float, float, float]:
    """Scale a (x_min, y_min, x_max, y_max) bounding box."""
    x_min, y_min, x_max, y_max = bbox
    return (
        x_min * scale_x,
        y_min * scale_y,
        x_max * scale_x,
        y_max * scale_y,
    )


def transform_segmentation_mask(
    mask: np.ndarray,
    target_size: Tuple[int, int],
) -> np.ndarray:
    """
    Resize a categorical segmentation mask using nearest-neighbour interpolation.

    target_size is (H, W).
    """
    h, w = target_size
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def transform_annotations(
    annotations: List[Dict[str, Any]],
    transform_record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Apply the geometric transform recorded in transform_record to a list
    of annotation dicts.

    Each annotation may contain:
        - 'polygon': list of [x, y] pairs
        - 'bbox': [x_min, y_min, x_max, y_max]
        - 'mask': np.ndarray (H×W categorical mask)

    Returns a new list of transformed annotation dicts.
    """
    if transform_record.get("type") != "resize":
        raise ValueError("Only 'resize' transform records are currently supported.")

    sx = transform_record["scale_x"]
    sy = transform_record["scale_y"]
    target_hw = transform_record["target_hw"]
    transformed = []

    for ann in annotations:
        t = dict(ann)  # shallow copy
        if "polygon" in t:
            t["polygon"] = transform_polygon(t["polygon"], sx, sy)
        if "bbox" in t:
            t["bbox"] = list(transform_bounding_box(tuple(t["bbox"]), sx, sy))
        if "mask" in t and t["mask"] is not None:
            t["mask"] = transform_segmentation_mask(t["mask"], target_hw)
        transformed.append(t)

    return transformed
