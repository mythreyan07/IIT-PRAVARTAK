"""
Image loading, validation and preservation utilities.

Supports JPG, JPEG, PNG in RGB, RGBA and grayscale.
Original images are never modified; copies are written to data/original/.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS: frozenset = frozenset({".jpg", ".jpeg", ".png"})
SUPPORTED_CHANNELS: frozenset = frozenset({1, 3, 4})  # Gray, RGB, RGBA

MIN_DIMENSION = 64
MAX_DIMENSION = 65536


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ImageLoadError(Exception):
    """Raised when an image cannot be loaded or validated."""


def validate_file_format(path: str | Path) -> Path:
    """
    Confirm the file exists and has a supported extension.

    Returns the resolved Path on success.
    Raises ImageLoadError on failure.
    """
    p = Path(path)
    if not p.exists():
        raise ImageLoadError(f"File not found: {p}")
    if not p.is_file():
        raise ImageLoadError(f"Path is not a file: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Unsupported file format '{p.suffix}'. "
            f"Supported formats: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return p


def validate_image_dimensions(image: np.ndarray) -> None:
    """
    Confirm that height and width are within acceptable bounds.

    Raises ImageLoadError when outside [MIN_DIMENSION, MAX_DIMENSION].
    """
    h, w = image.shape[:2]
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        raise ImageLoadError(
            f"Image too small ({w}×{h}). Minimum is {MIN_DIMENSION}×{MIN_DIMENSION}."
        )
    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        raise ImageLoadError(
            f"Image too large ({w}×{h}). Maximum is {MAX_DIMENSION}×{MAX_DIMENSION}."
        )


def validate_colour_channels(image: np.ndarray) -> None:
    """
    Confirm the image has 1, 3 or 4 channels.

    Raises ImageLoadError for unexpected channel counts.
    """
    channels = 1 if image.ndim == 2 else image.shape[2]
    if channels not in SUPPORTED_CHANNELS:
        raise ImageLoadError(
            f"Unsupported channel count {channels}. "
            f"Expected one of {sorted(SUPPORTED_CHANNELS)}."
        )


def convert_colour_space(
    image: np.ndarray,
    target: str = "RGB",
) -> np.ndarray:
    """
    Convert image to the requested colour space.

    Supported targets: 'RGB', 'BGR', 'GRAY', 'RGBA'.
    OpenCV reads images as BGR internally; callers receive RGB by default.
    """
    channels = 1 if image.ndim == 2 else image.shape[2]
    target = target.upper()

    if target == "RGB":
        if channels == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif target == "BGR":
        if channels == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if channels == 3:
            return image.copy()
    elif target == "GRAY":
        if channels == 1:
            return image.copy()
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif target == "RGBA":
        if channels == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        if channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        raise ImageLoadError(f"Unknown target colour space: '{target}'")

    raise ImageLoadError(
        f"Cannot convert {channels}-channel image to '{target}'."
    )


def load_image(
    path: str | Path,
    colour_space: str = "RGB",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load, validate and convert an image file.

    Returns:
        (image_array, metadata_dict)

    Raises ImageLoadError for any validation failure.
    """
    p = validate_file_format(path)

    # Read raw bytes to detect corruption before OpenCV decoding
    try:
        raw = np.frombuffer(p.read_bytes(), dtype=np.uint8)
    except OSError as exc:
        raise ImageLoadError(f"Cannot read file '{p}': {exc}") from exc

    image = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ImageLoadError(
            f"OpenCV could not decode '{p}'. "
            "The file may be corrupted or in an unsupported encoding."
        )

    validate_image_dimensions(image)
    validate_colour_channels(image)

    original_channels = 1 if image.ndim == 2 else image.shape[2]
    image = convert_colour_space(image, target=colour_space)

    metadata: Dict[str, Any] = {
        "source_path": str(p),
        "file_name": p.name,
        "original_dimensions": list(image.shape[:2]),  # [H, W]
        "original_channels": original_channels,
        "colour_space": colour_space,
        "file_size_bytes": p.stat().st_size,
        "dtype": str(image.dtype),
    }

    logger.debug("Loaded %s — %s channels → %s", p.name, original_channels, colour_space)
    return image, metadata


def preserve_original_image(
    source_path: str | Path,
    dest_dir: str | Path,
    image_pair_id: str,
    role: str,  # 'before' or 'after'
) -> Path:
    """
    Copy the original file into dest_dir without modification.

    The destination filename encodes the pair ID and role to avoid collisions.
    Returns the destination Path.
    """
    src = Path(source_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / f"{image_pair_id}_{role}{src.suffix.lower()}"
    if dest.exists():
        logger.warning(
            "Overwriting existing original at %s — pair ID '%s' was reused.",
            dest,
            image_pair_id,
        )
    shutil.copy2(src, dest)
    logger.info("Preserved original: %s → %s", src.name, dest)
    return dest


def save_processed_image(
    image: np.ndarray,
    dest_dir: str | Path,
    image_pair_id: str,
    role: str,
    suffix: str = ".png",
    colour_space_in: str = "RGB",
) -> Path:
    """
    Write a processed image (RGB or GRAY numpy array) to dest_dir as PNG/JPEG.

    OpenCV expects BGR for colour images; this function handles the conversion.
    Returns the saved file Path.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{image_pair_id}_{role}{suffix}"

    channels = 1 if image.ndim == 2 else image.shape[2]
    if channels == 3 and colour_space_in.upper() == "RGB":
        save_array = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif channels == 4 and colour_space_in.upper() == "RGBA":
        save_array = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    else:
        save_array = image

    ok = cv2.imwrite(str(dest), save_array)
    if not ok:
        raise ImageLoadError(f"cv2.imwrite failed for '{dest}'.")
    logger.info("Saved processed image: %s", dest)
    return dest
