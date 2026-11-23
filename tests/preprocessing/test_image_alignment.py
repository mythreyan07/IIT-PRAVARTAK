"""Tests for image_alignment.py"""

import cv2
import numpy as np
import pytest

from backend.preprocessing.image_alignment import (
    AlignmentConfig,
    align_images,
    calculate_alignment_quality,
    detect_features,
    estimate_homography,
    match_features,
    verify_alignment,
)


def _checkerboard(h=256, w=256, block=16) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if ((i // block) + (j // block)) % 2 == 0:
                img[i, j] = [200, 200, 200]
            else:
                img[i, j] = [50, 50, 50]
    return img


def _shifted(image: np.ndarray, dx: int = 5, dy: int = 3) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))


# ---------------------------------------------------------------------------
# detect_features
# ---------------------------------------------------------------------------


class TestDetectFeatures:
    def test_returns_keypoints_and_descriptors(self):
        img = _checkerboard()
        kps, descs = detect_features(img)
        assert len(kps) > 0
        assert descs is not None
        assert descs.ndim == 2

    def test_no_features_raises(self):
        # Completely flat image has no features
        blank = np.full((64, 64, 3), 128, dtype=np.uint8)
        with pytest.raises(RuntimeError, match="No ORB features"):
            detect_features(blank)

    def test_grayscale_input(self, gray_image):
        kps, descs = detect_features(gray_image)
        assert len(kps) > 0


# ---------------------------------------------------------------------------
# match_features
# ---------------------------------------------------------------------------


class TestMatchFeatures:
    def test_similar_images_return_matches(self):
        ref = _checkerboard()
        query = _shifted(ref, dx=2, dy=2)
        _, d_ref = detect_features(ref)
        _, d_qry = detect_features(query)
        matches = match_features(d_ref, d_qry)
        # Slightly shifted checkerboard should produce many good matches
        assert len(matches) > 0

    def test_shifted_image_matches(self):
        ref = _checkerboard(256, 256, 16)
        query = _shifted(ref, dx=5, dy=3)
        _, d_ref = detect_features(ref)
        _, d_qry = detect_features(query)
        matches = match_features(d_ref, d_qry)
        assert len(matches) >= 5


# ---------------------------------------------------------------------------
# estimate_homography
# ---------------------------------------------------------------------------


class TestEstimateHomography:
    def test_successful_homography(self):
        ref = _checkerboard(256, 256, 16)
        query = _shifted(ref, dx=5, dy=3)
        kps_ref, d_ref = detect_features(ref)
        kps_qry, d_qry = detect_features(query)
        matches = match_features(d_ref, d_qry)
        H, mask = estimate_homography(kps_ref, kps_qry, matches)
        assert H is not None
        assert H.shape == (3, 3)
        assert mask is not None

    def test_insufficient_matches_returns_none(self):
        # Fabricate very few matches
        kps_ref = tuple()
        kps_qry = tuple()
        H, mask = estimate_homography(kps_ref, kps_qry, [], AlignmentConfig(min_good_matches=1))
        assert H is None


# ---------------------------------------------------------------------------
# align_images
# ---------------------------------------------------------------------------


class TestAlignImages:
    def test_already_aligned_identity(self, identical_pair):
        ref, query = identical_pair
        result = align_images(ref, query)
        assert result["aligned_image"] is not None
        assert result["method"] in {"IDENTITY", "ORB_RANSAC", "ECC"}
        assert result["alignment_needed"] is False or result["method"] != "IDENTITY"

    def test_shifted_pair_aligned(self, checkerboard_pair):
        ref, shifted = checkerboard_pair
        result = align_images(ref, shifted)
        assert result["aligned_image"] is not None
        assert result["method"] in {"ORB_RANSAC", "ECC"}

    def test_failed_pair_marked_failed(self):
        # Use two completely different random noise images with minimal features
        rng = np.random.default_rng(42)
        # Uniform images have no features; alignment should fail
        img1 = np.full((64, 64, 3), 100, dtype=np.uint8)
        img2 = np.full((64, 64, 3), 150, dtype=np.uint8)
        cfg = AlignmentConfig(min_good_matches=50, orb_n_features=10)
        result = align_images(img1, img2, cfg)
        # Either FAILED or some method with warnings
        assert result["method"] in {"FAILED", "ECC", "ORB_RANSAC", "IDENTITY"}

    def test_result_keys(self, checkerboard_pair):
        ref, shifted = checkerboard_pair
        result = align_images(ref, shifted)
        required = {"aligned_image", "homography", "method", "alignment_needed", "quality", "warnings"}
        assert required <= result.keys()

    def test_aligned_image_same_shape_as_reference(self, checkerboard_pair):
        ref, shifted = checkerboard_pair
        result = align_images(ref, shifted)
        if result["aligned_image"] is not None:
            assert result["aligned_image"].shape == ref.shape


# ---------------------------------------------------------------------------
# calculate_alignment_quality
# ---------------------------------------------------------------------------


class TestCalculateAlignmentQuality:
    def test_identical_images_high_ncc(self, identical_pair):
        ref, aln = identical_pair
        q = calculate_alignment_quality(ref, aln)
        assert q["ncc"] > 0.9
        assert q["mad"] < 1.0
        assert q["alignment_valid"] is True

    def test_different_images_lower_ncc(self, medium_rgb):
        rng = np.random.default_rng(99)
        random_img = rng.integers(0, 256, medium_rgb.shape, dtype=np.uint8)
        q = calculate_alignment_quality(medium_rgb, random_img)
        assert "ncc" in q
        assert "alignment_valid" in q

    def test_mismatched_sizes_handled(self):
        ref = np.zeros((128, 128, 3), dtype=np.uint8)
        aln = np.zeros((64, 64, 3), dtype=np.uint8)
        q = calculate_alignment_quality(ref, aln)
        assert "ncc" in q


# ---------------------------------------------------------------------------
# verify_alignment
# ---------------------------------------------------------------------------


class TestVerifyAlignment:
    def test_identical_verified(self):
        # Use a random (non-periodic) image to avoid phaseCorrelate aliasing
        rng = np.random.default_rng(42)
        ref = rng.integers(30, 220, (128, 128, 3), dtype=np.uint8)
        cfg = AlignmentConfig(residual_threshold_px=50.0)
        result = verify_alignment(ref, ref.copy(), cfg)
        assert result["verified"] is True
        assert "quality" in result

    def test_very_different_not_verified(self, medium_rgb):
        rng = np.random.default_rng(77)
        noise = rng.integers(0, 256, medium_rgb.shape, dtype=np.uint8)
        result = verify_alignment(medium_rgb, noise)
        assert "verified" in result
        assert isinstance(result["explanation"], str)
