"""
Image registration / alignment for before-and-after satellite image pairs.

Strategy:
1. ORB feature detection + ratio-test matching + RANSAC homography (primary).
2. ECC (Enhanced Correlation Coefficient) registration (fallback).

If neither method produces a reliable result the pair is flagged as
unreliable — the caller must NOT proceed with building-level comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AlignmentConfig:
    # ORB
    orb_n_features: int = 5000
    # Lowe ratio test
    ratio_threshold: float = 0.75
    # Minimum inlier matches for a valid homography
    min_good_matches: int = 10
    # RANSAC reprojection threshold (pixels)
    ransac_reproj_threshold: float = 5.0
    # ECC: max iterations for the optimiser
    ecc_max_iter: int = 1000
    # ECC: convergence epsilon
    ecc_epsilon: float = 1e-6
    # Alignment is considered "good" if mean residual < this many pixels
    residual_threshold_px: float = 3.0
    # If the median displacement between matched keypoints is below this,
    # treat the images as already aligned (skip registration).
    already_aligned_threshold_px: float = 2.0


DEFAULT_CONFIG = AlignmentConfig()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def detect_features(
    image: np.ndarray,
    config: AlignmentConfig | None = None,
) -> Tuple[Tuple, np.ndarray]:
    """
    Detect ORB keypoints and compute descriptors.

    Returns (keypoints, descriptors).
    Raises RuntimeError when no features are found.
    """
    cfg = config or DEFAULT_CONFIG
    gray = _to_gray(image)
    orb = cv2.ORB_create(nfeatures=cfg.orb_n_features)
    kps, descs = orb.detectAndCompute(gray, None)
    if descs is None or len(kps) == 0:
        raise RuntimeError("No ORB features detected in the image.")
    logger.debug("Detected %d ORB keypoints.", len(kps))
    return kps, descs


def match_features(
    desc_ref: np.ndarray,
    desc_query: np.ndarray,
    config: AlignmentConfig | None = None,
) -> list:
    """
    Match descriptors using BFMatcher with Hamming distance and Lowe ratio test.

    Returns a list of good DMatch objects.
    """
    cfg = config or DEFAULT_CONFIG
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn_matches = matcher.knnMatch(desc_ref, desc_query, k=2)
    good = [
        m
        for m, n in knn_matches
        if m.distance < cfg.ratio_threshold * n.distance
    ]
    logger.debug("Good matches after ratio test: %d", len(good))
    return good


def estimate_homography(
    kps_ref: tuple,
    kps_query: tuple,
    good_matches: list,
    config: AlignmentConfig | None = None,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Estimate the homography from matched keypoints using RANSAC.

    Returns (H, inlier_mask) where H is None when estimation fails.
    """
    cfg = config or DEFAULT_CONFIG
    if len(good_matches) < cfg.min_good_matches:
        logger.warning(
            "Insufficient good matches for homography (%d < %d).",
            len(good_matches),
            cfg.min_good_matches,
        )
        return None, np.array([])

    src_pts = np.float32([kps_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kps_query[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        cfg.ransac_reproj_threshold,
    )
    if H is None:
        logger.warning("RANSAC homography estimation returned None.")
        return None, np.array([])

    n_inliers = int(mask.sum()) if mask is not None else 0
    logger.debug("Homography estimated — %d inliers.", n_inliers)
    return H, mask


def align_images(
    image_ref: np.ndarray,
    image_query: np.ndarray,
    config: AlignmentConfig | None = None,
) -> Dict[str, Any]:
    """
    Align image_query to image_ref.

    Returns a result dict:
    {
        "aligned_image":    np.ndarray or None,
        "homography":       np.ndarray or None,
        "method":           "ORB_RANSAC" | "ECC" | "IDENTITY" | "FAILED",
        "alignment_needed": bool,
        "n_good_matches":   int,
        "n_inliers":        int,
        "quality":          dict from calculate_alignment_quality(),
        "warnings":         list[str],
    }
    """
    cfg = config or DEFAULT_CONFIG
    warnings: list[str] = []

    # --- Check whether alignment is already unnecessary ---
    displacement = _estimate_median_displacement(image_ref, image_query, cfg)
    if displacement is not None and displacement < cfg.already_aligned_threshold_px:
        logger.info(
            "Images appear pre-aligned (median displacement=%.2f px). Skipping registration.",
            displacement,
        )
        quality = calculate_alignment_quality(image_ref, image_query)
        return {
            "aligned_image": image_query.copy(),
            "homography": np.eye(3, dtype=np.float64),
            "method": "IDENTITY",
            "alignment_needed": False,
            "n_good_matches": 0,
            "n_inliers": 0,
            "quality": quality,
            "warnings": warnings,
        }

    # --- Primary: ORB + RANSAC ---
    try:
        kps_ref, desc_ref = detect_features(image_ref, cfg)
        kps_qry, desc_qry = detect_features(image_query, cfg)
        good = match_features(desc_ref, desc_qry, cfg)
        H, mask = estimate_homography(kps_ref, kps_qry, good, cfg)

        if H is not None:
            h, w = image_ref.shape[:2]
            aligned = cv2.warpPerspective(image_query, H, (w, h))
            quality = calculate_alignment_quality(image_ref, aligned)
            n_inliers = int(mask.sum()) if mask is not None else 0
            if quality["mean_residual_px"] > cfg.residual_threshold_px:
                warnings.append(
                    f"Alignment residual {quality['mean_residual_px']:.2f} px "
                    f"exceeds threshold {cfg.residual_threshold_px} px."
                )
            return {
                "aligned_image": aligned,
                "homography": H,
                "method": "ORB_RANSAC",
                "alignment_needed": True,
                "n_good_matches": len(good),
                "n_inliers": n_inliers,
                "quality": quality,
                "warnings": warnings,
            }
        else:
            warnings.append("ORB+RANSAC homography failed — trying ECC fallback.")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"ORB feature detection/matching failed: {exc}")

    # --- Fallback: ECC registration ---
    try:
        logger.info("Attempting ECC registration.")
        aligned, warp_matrix = _ecc_align(image_ref, image_query, cfg)
        H_ecc = np.eye(3, dtype=np.float64)
        H_ecc[:2, :] = warp_matrix
        quality = calculate_alignment_quality(image_ref, aligned)
        return {
            "aligned_image": aligned,
            "homography": H_ecc,
            "method": "ECC",
            "alignment_needed": True,
            "n_good_matches": 0,
            "n_inliers": 0,
            "quality": quality,
            "warnings": warnings,
        }
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"ECC registration also failed: {exc}")

    # --- Both methods failed ---
    warnings.append(
        "CRITICAL: Image registration failed. This pair is unreliable for "
        "building-level damage comparison."
    )
    return {
        "aligned_image": None,
        "homography": None,
        "method": "FAILED",
        "alignment_needed": True,
        "n_good_matches": 0,
        "n_inliers": 0,
        "quality": {"alignment_valid": False},
        "warnings": warnings,
    }


def calculate_alignment_quality(
    image_ref: np.ndarray,
    image_aligned: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute quality metrics for a registration result.

    Uses Normalised Cross-Correlation (NCC) and mean absolute difference (MAD).
    """
    gray_ref = _to_gray(image_ref)
    gray_aln = _to_gray(image_aligned)

    # Resize query to reference if shapes differ (should not happen post-warp)
    if gray_ref.shape != gray_aln.shape:
        gray_aln = cv2.resize(gray_aln, (gray_ref.shape[1], gray_ref.shape[0]))

    ref_f = gray_ref.astype(np.float32)
    aln_f = gray_aln.astype(np.float32)

    # NCC
    ref_norm = (ref_f - ref_f.mean()) / (ref_f.std() + 1e-8)
    aln_norm = (aln_f - aln_f.mean()) / (aln_f.std() + 1e-8)
    ncc = float(np.mean(ref_norm * aln_norm))

    # MAD (mean absolute difference, 0-255 scale)
    mad = float(np.mean(np.abs(ref_f - aln_f)))

    # Approximate mean residual in "pixel displacement" units via phase correlation
    try:
        (dx, dy), _resp = cv2.phaseCorrelate(ref_f, aln_f)
        mean_residual = float(np.sqrt(dx**2 + dy**2))
    except Exception:  # noqa: BLE001
        mean_residual = float("nan")

    return {
        "ncc": round(ncc, 6),
        "mad": round(mad, 4),
        "mean_residual_px": round(mean_residual, 4) if not np.isnan(mean_residual) else None,
        "alignment_valid": ncc > 0.5,
    }


def verify_alignment(
    image_ref: np.ndarray,
    image_aligned: np.ndarray,
    config: AlignmentConfig | None = None,
) -> Dict[str, Any]:
    """
    Post-registration verification.

    Returns a dict with 'verified' (bool) and an explanation.
    """
    cfg = config or DEFAULT_CONFIG
    quality = calculate_alignment_quality(image_ref, image_aligned)
    residual = quality.get("mean_residual_px")

    verified = (
        quality["alignment_valid"]
        and (residual is None or residual < cfg.residual_threshold_px)
    )
    return {
        "verified": verified,
        "quality": quality,
        "explanation": (
            "Alignment verified."
            if verified
            else f"Alignment verification failed — NCC={quality['ncc']:.4f}, "
            f"residual={residual} px."
        ),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"Cannot convert {image.shape[2]}-channel image to grayscale.")


def _estimate_median_displacement(
    img1: np.ndarray,
    img2: np.ndarray,
    config: AlignmentConfig,
) -> Optional[float]:
    """Quick displacement estimate using sparse ORB matching."""
    try:
        kps1, d1 = detect_features(img1, config)
        kps2, d2 = detect_features(img2, config)
        good = match_features(d1, d2, config)
        if len(good) < 5:
            return None
        pts1 = np.float32([kps1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in good])
        dists = np.linalg.norm(pts1 - pts2, axis=1)
        return float(np.median(dists))
    except Exception:  # noqa: BLE001
        return None


def _ecc_align(
    image_ref: np.ndarray,
    image_query: np.ndarray,
    config: AlignmentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ECC-based registration with affine warp model.

    Returns (aligned_image, warp_matrix_2x3).
    """
    gray_ref = _to_gray(image_ref)
    gray_qry = _to_gray(image_query)

    if gray_ref.shape != gray_qry.shape:
        gray_qry = cv2.resize(gray_qry, (gray_ref.shape[1], gray_ref.shape[0]))

    warp_mode = cv2.MOTION_AFFINE
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        config.ecc_max_iter,
        config.ecc_epsilon,
    )
    _cc, warp_matrix = cv2.findTransformECC(
        gray_ref.astype(np.float32),
        gray_qry.astype(np.float32),
        warp_matrix,
        warp_mode,
        criteria,
    )
    h, w = image_ref.shape[:2]
    aligned = cv2.warpAffine(
        image_query,
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    return aligned, warp_matrix
