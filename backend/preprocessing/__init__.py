"""
EAGLE-DAR OpenCV Preprocessing Module

Provides image loading, quality assessment, alignment, normalisation,
and pipeline orchestration for before/after satellite image pairs.
"""

from .image_loader import (
    load_image,
    validate_file_format,
    validate_image_dimensions,
    validate_colour_channels,
    convert_colour_space,
    preserve_original_image,
    save_processed_image,
)
from .image_quality import (
    check_blur,
    check_brightness,
    check_contrast,
    check_resolution,
    check_image_integrity,
    assess_image_quality,
)
from .image_alignment import (
    detect_features,
    match_features,
    estimate_homography,
    align_images,
    calculate_alignment_quality,
    verify_alignment,
)
from .image_normalization import (
    resize_image,
    normalize_image,
    convert_image_format,
    apply_training_normalization,
    apply_inference_normalization,
)
from .preprocessing_pipeline import PreprocessingPipeline, PreprocessingConfig

__all__ = [
    # Loader
    "load_image",
    "validate_file_format",
    "validate_image_dimensions",
    "validate_colour_channels",
    "convert_colour_space",
    "preserve_original_image",
    "save_processed_image",
    # Quality
    "check_blur",
    "check_brightness",
    "check_contrast",
    "check_resolution",
    "check_image_integrity",
    "assess_image_quality",
    # Alignment
    "detect_features",
    "match_features",
    "estimate_homography",
    "align_images",
    "calculate_alignment_quality",
    "verify_alignment",
    # Normalisation
    "resize_image",
    "normalize_image",
    "convert_image_format",
    "apply_training_normalization",
    "apply_inference_normalization",
    # Pipeline
    "PreprocessingPipeline",
    "PreprocessingConfig",
]
