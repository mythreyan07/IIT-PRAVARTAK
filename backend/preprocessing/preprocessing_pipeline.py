"""
End-to-end preprocessing pipeline for before/after satellite image pairs.

Orchestrates: load → quality check → dimension/channel validation →
alignment → resize → normalise → verify → save → metadata.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .image_loader import (
    load_image,
    preserve_original_image,
    save_processed_image,
    ImageLoadError,
)
from .image_quality import assess_image_quality, QualityThresholds
from .image_alignment import align_images, verify_alignment, AlignmentConfig
from .image_normalization import (
    apply_inference_normalization,
    transform_annotations,
    NormalizationConfig,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    """Single configuration object for the full pipeline."""

    # Root paths
    original_before_dir: str = "data/original/before"
    original_after_dir: str = "data/original/after"
    processed_before_dir: str = "data/processed/before"
    processed_after_dir: str = "data/processed/after"
    aligned_dir: str = "data/processed/aligned"
    normalized_dir: str = "data/processed/normalized"
    previews_dir: str = "data/processed/previews"
    metadata_dir: str = "metadata"

    # Sub-configurations
    quality: QualityThresholds = field(default_factory=QualityThresholds)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)

    # Whether to skip the pipeline on a "Poor" quality image
    reject_poor_quality: bool = False

    # Colour space used internally throughout the pipeline
    internal_colour_space: str = "RGB"


# ---------------------------------------------------------------------------
# Pipeline result dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineResult:
    image_pair_id: str
    preprocessing_status: str  # 'completed', 'failed', 'unreliable'
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # Paths
    original_before: Optional[str] = None
    original_after: Optional[str] = None
    processed_before: Optional[str] = None
    processed_after: Optional[str] = None
    aligned_after: Optional[str] = None
    normalized_before: Optional[str] = None
    normalized_after: Optional[str] = None

    # Dimensions
    original_dimensions: Optional[List[int]] = None
    processed_dimensions: Optional[List[int]] = None

    # Quality
    before_quality: Optional[Dict[str, Any]] = None
    after_quality: Optional[Dict[str, Any]] = None

    # Alignment
    alignment_required: bool = False
    alignment_status: str = "not_attempted"
    alignment_method: Optional[str] = None
    alignment_quality: Optional[float] = None
    homography: Optional[List[List[float]]] = None

    # Normalisation transform
    normalization_record: Optional[Dict[str, Any]] = None

    # Timestamps
    preprocessing_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pipeline class
# ---------------------------------------------------------------------------


class PreprocessingPipeline:
    """
    Runs the full preprocessing sequence for a before/after image pair.

    Usage::

        pipeline = PreprocessingPipeline(config)
        result = pipeline.process_image_pair(before_path, after_path)
    """

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()
        self._ensure_dirs()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process_image_pair(
        self,
        before_path: str,
        after_path: str,
        image_pair_id: str | None = None,
        annotations_before: Optional[List[Dict]] = None,
        annotations_after: Optional[List[Dict]] = None,
    ) -> PipelineResult:
        """
        Process a before/after image pair through the full pipeline.

        Returns a PipelineResult; also saves metadata JSON to disk.
        """
        pair_id = image_pair_id or _generate_pair_id()
        result = PipelineResult(image_pair_id=pair_id, preprocessing_status="in_progress")
        cfg = self.config

        # --- Step 1: Preserve originals ---
        try:
            orig_before = preserve_original_image(
                before_path, cfg.original_before_dir, pair_id, "before"
            )
            orig_after = preserve_original_image(
                after_path, cfg.original_after_dir, pair_id, "after"
            )
            result.original_before = str(orig_before)
            result.original_after = str(orig_after)
        except Exception as exc:  # noqa: BLE001
            return self._fail(result, f"Failed to preserve originals: {exc}")

        # --- Step 2: Load images ---
        try:
            img_before, meta_before = load_image(before_path, cfg.internal_colour_space)
            img_after, meta_after = load_image(after_path, cfg.internal_colour_space)
        except ImageLoadError as exc:
            return self._fail(result, f"Image load error: {exc}")

        result.original_dimensions = meta_before["original_dimensions"]

        # --- Step 3: Quality assessment ---
        result.before_quality = assess_image_quality(img_before, cfg.quality)
        result.after_quality = assess_image_quality(img_after, cfg.quality)

        for label, quality in [("before", result.before_quality), ("after", result.after_quality)]:
            if quality["overall_quality"] == "Poor":
                msg = f"{label.capitalize()} image quality is 'Poor': {quality}"
                result.warnings.append(msg)
                logger.warning(msg)
                if cfg.reject_poor_quality:
                    return self._fail(result, f"Rejected due to poor quality ({label}).")
            elif quality["overall_quality"] == "Warning":
                result.warnings.append(
                    f"{label.capitalize()} image quality warning: {quality}"
                )

        # --- Step 4: Save processed (pre-alignment) copies ---
        try:
            proc_before_path = save_processed_image(
                img_before, cfg.processed_before_dir, pair_id, "before"
            )
            proc_after_path = save_processed_image(
                img_after, cfg.processed_after_dir, pair_id, "after"
            )
            result.processed_before = str(proc_before_path)
            result.processed_after = str(proc_after_path)
        except Exception as exc:  # noqa: BLE001
            return self._fail(result, f"Failed to save processed images: {exc}")

        # --- Step 5: Alignment ---
        align_result = align_images(img_before, img_after, cfg.alignment)
        result.alignment_required = align_result["alignment_needed"]
        result.alignment_method = align_result["method"]
        result.warnings.extend(align_result.get("warnings", []))

        if align_result["method"] == "FAILED":
            result.preprocessing_status = "unreliable"
            result.alignment_status = "failed"
            result.errors.append(
                "Image registration failed — pair is unreliable for building-level comparison."
            )
            self._save_metadata(result)
            return result

        aligned_after = align_result["aligned_image"]
        H = align_result["homography"]
        q = align_result.get("quality", {})
        result.alignment_status = "successful"
        result.alignment_quality = q.get("ncc")
        if H is not None:
            result.homography = H.tolist()

        # Save aligned after image
        try:
            aligned_path = save_processed_image(
                aligned_after, cfg.aligned_dir, pair_id, "after_aligned"
            )
            result.aligned_after = str(aligned_path)
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"Could not save aligned image: {exc}")

        # Verify alignment
        verify = verify_alignment(img_before, aligned_after, cfg.alignment)
        if not verify["verified"]:
            result.warnings.append(
                f"Post-alignment verification failed: {verify['explanation']}"
            )
            if result.preprocessing_status != "unreliable":
                result.preprocessing_status = "unreliable"
                result.errors.append("Alignment verification failed — pair flagged as unreliable.")
                self._save_metadata(result)
                return result

        # --- Step 6: Normalisation ---
        try:
            norm_before, norm_record = apply_inference_normalization(
                img_before, cfg.normalization
            )
            norm_after, _ = apply_inference_normalization(aligned_after, cfg.normalization)
        except Exception as exc:  # noqa: BLE001
            return self._fail(result, f"Normalisation failed: {exc}")

        result.normalization_record = norm_record
        result.processed_dimensions = list(norm_before.shape[:2])

        # Save normalised images (convert float back to uint8 for storage)
        try:
            norm_before_uint8 = _float_to_uint8(norm_before)
            norm_after_uint8 = _float_to_uint8(norm_after)
            norm_before_path = save_processed_image(
                norm_before_uint8, cfg.normalized_dir, pair_id, "before_norm"
            )
            norm_after_path = save_processed_image(
                norm_after_uint8, cfg.normalized_dir, pair_id, "after_norm"
            )
            result.normalized_before = str(norm_before_path)
            result.normalized_after = str(norm_after_path)
        except Exception as exc:  # noqa: BLE001
            result.warnings.append(f"Could not save normalised images: {exc}")

        # --- Step 7: Transform annotations ---
        if annotations_before and norm_record:
            try:
                annotations_before = transform_annotations(annotations_before, norm_record)
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(f"Annotation transform (before) failed: {exc}")
        if annotations_after and norm_record:
            try:
                annotations_after = transform_annotations(annotations_after, norm_record)
            except Exception as exc:  # noqa: BLE001
                result.warnings.append(f"Annotation transform (after) failed: {exc}")

        # --- Done ---
        result.preprocessing_status = "completed"
        self._save_metadata(result)
        logger.info("Pipeline completed for pair %s — status=%s", pair_id, result.preprocessing_status)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        cfg = self.config
        for d in [
            cfg.original_before_dir,
            cfg.original_after_dir,
            cfg.processed_before_dir,
            cfg.processed_after_dir,
            cfg.aligned_dir,
            cfg.normalized_dir,
            cfg.previews_dir,
            cfg.metadata_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def _fail(self, result: PipelineResult, reason: str) -> PipelineResult:
        result.preprocessing_status = "failed"
        result.errors.append(reason)
        logger.error("Pipeline failed for pair %s: %s", result.image_pair_id, reason)
        self._save_metadata(result)
        return result

    def _save_metadata(self, result: PipelineResult) -> None:
        dest = Path(self.config.metadata_dir) / f"{result.image_pair_id}.json"
        try:
            dest.write_text(
                json.dumps(result.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not save metadata for %s: %s", result.image_pair_id, exc)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _generate_pair_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6].upper()
    return f"PAIR_{ts}_{short_uuid}"


def _float_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert a float32 image (any range) back to uint8 for disk storage.

    Uses min-max normalisation per image so relative differences are preserved.
    """
    img = image.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)
