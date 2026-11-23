"""
Dataset inspection and analysis utilities.

Works with both real HuggingFace datasets and synthetic mock datasets
(dicts of lists) so tests never need to download anything.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .hf_dataset_loader import get_damage_class_names, get_building_annotations

logger = logging.getLogger(__name__)

_DAMAGE_CLASSES = get_damage_class_names()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inspect_dataset(dataset: Any) -> Dict[str, Any]:
    """
    Return a summary of a single split dataset.

    Accepts a datasets.Dataset object or a plain list of sample dicts.
    """
    samples = _to_list(dataset)
    n = len(samples)
    if n == 0:
        return {"n_samples": 0, "note": "Empty dataset"}

    damage_counts = count_damage_classes(dataset)
    dim_info = check_image_dimensions(dataset, n_samples=min(50, n))
    ann_coverage = check_annotation_coverage(dataset, n_samples=min(50, n))

    return {
        "n_samples": n,
        "damage_class_counts": damage_counts,
        "image_dimensions": dim_info,
        "annotation_coverage": ann_coverage,
    }


def count_samples_per_split(dataset_dict: Any) -> Dict[str, int]:
    """
    Count the number of samples in each split.

    Accepts a datasets.DatasetDict or a plain dict of {split: list}.
    """
    if hasattr(dataset_dict, "keys"):
        return {split: len(dataset_dict[split]) for split in dataset_dict.keys()}
    if isinstance(dataset_dict, dict):
        return {k: len(v) for k, v in dataset_dict.items() if isinstance(v, (list, Sequence))}
    # Single split
    return {"default": len(_to_list(dataset_dict))}


def count_damage_classes(dataset: Any) -> Dict[str, int]:
    """
    Count how many buildings fall into each damage class across all samples.

    Returns {class_name: count}.
    """
    samples = _to_list(dataset)
    counter: Counter = Counter()
    for sample in samples:
        anns = _get_annotations_safe(sample)
        for ann in anns:
            label = ann.get("damage_label", _DAMAGE_CLASSES.get(ann.get("damage_class", 0), "unknown"))
            counter[label] += 1
    # Ensure all classes appear
    result = {name: 0 for name in _DAMAGE_CLASSES.values()}
    result.update(counter)
    return result


def check_image_dimensions(
    dataset: Any,
    n_samples: int = 50,
) -> Dict[str, Any]:
    """
    Sample up to n_samples images and return dimension statistics.

    Returns {"min_hw": [H, W], "max_hw": [H, W], "mean_hw": [H, W], "n_checked": int}.
    """
    samples = _to_list(dataset)[:n_samples]
    heights, widths = [], []

    for sample in samples:
        # Support PIL Image, numpy array, or dict with an image field
        img = _extract_any_image(sample)
        if img is None:
            continue
        try:
            from PIL import Image as _PILImage
            if isinstance(img, _PILImage.Image):
                w, h = img.size
            elif isinstance(img, np.ndarray):
                h, w = img.shape[:2]
            else:
                continue
            heights.append(h)
            widths.append(w)
        except Exception:  # noqa: BLE001
            continue

    if not heights:
        return {"min_hw": None, "max_hw": None, "mean_hw": None, "n_checked": 0}

    return {
        "min_hw": [int(min(heights)), int(min(widths))],
        "max_hw": [int(max(heights)), int(max(widths))],
        "mean_hw": [int(np.mean(heights)), int(np.mean(widths))],
        "n_checked": len(heights),
    }


def check_annotation_coverage(
    dataset: Any,
    n_samples: int = 50,
) -> float:
    """
    Return the fraction of sampled items that have at least one valid annotation.
    """
    samples = _to_list(dataset)[:n_samples]
    if not samples:
        return 0.0
    covered = sum(1 for s in samples if len(_get_annotations_safe(s)) > 0)
    return covered / len(samples)


def compute_class_weights(dataset: Any) -> Dict[int, float]:
    """
    Compute per-class weights as the inverse of class frequency.

    Useful for weighted loss / weighted sampling to handle class imbalance.
    Returns {class_id: weight}.
    """
    samples = _to_list(dataset)
    counts: Dict[int, int] = {c: 0 for c in _DAMAGE_CLASSES}
    for sample in samples:
        anns = _get_annotations_safe(sample)
        for ann in anns:
            cls = ann.get("damage_class", 0)
            counts[cls] = counts.get(cls, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return {c: 1.0 for c in _DAMAGE_CLASSES}

    n_classes = len(_DAMAGE_CLASSES)
    weights: Dict[int, float] = {}
    for cls_id, cnt in counts.items():
        if cnt == 0:
            weights[cls_id] = float(n_classes)
        else:
            weights[cls_id] = total / (n_classes * cnt)
    return weights


def generate_inspection_report(dataset_dict: Any) -> Dict[str, Any]:
    """
    Generate a full inspection report for a DatasetDict or dict of splits.

    Returns a structured dict with per-split summaries and overall stats.
    """
    split_counts = count_samples_per_split(dataset_dict)
    per_split: Dict[str, Any] = {}

    if hasattr(dataset_dict, "keys"):
        splits_iter = {k: dataset_dict[k] for k in dataset_dict.keys()}
    elif isinstance(dataset_dict, dict):
        splits_iter = dataset_dict
    else:
        splits_iter = {"default": dataset_dict}

    all_damage: Counter = Counter()
    for split_name, split_data in splits_iter.items():
        summary = inspect_dataset(split_data)
        per_split[split_name] = summary
        for cls, cnt in summary.get("damage_class_counts", {}).items():
            all_damage[cls] += cnt

    total_samples = sum(split_counts.values())
    return {
        "total_samples": total_samples,
        "splits": split_counts,
        "per_split_summary": per_split,
        "overall_damage_class_counts": dict(all_damage),
        "damage_class_definitions": _DAMAGE_CLASSES,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _to_list(dataset: Any) -> List[Dict[str, Any]]:
    """Convert any dataset-like object to a plain list of dicts."""
    if isinstance(dataset, list):
        return dataset
    # datasets.Dataset supports iteration
    try:
        return list(dataset)
    except TypeError:
        return []


def _get_annotations_safe(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        return get_building_annotations(sample)
    except Exception:  # noqa: BLE001
        return []


def _extract_any_image(sample: Dict[str, Any]) -> Any:
    """Return the first image-like object found in a sample dict."""
    from PIL import Image as PILImage
    for key in ["before", "after", "pre_image", "post_image", "image", "image_pre", "image_post"]:
        val = sample.get(key)
        if val is None:
            continue
        if isinstance(val, PILImage.Image):
            return val
        if isinstance(val, np.ndarray):
            return val
        if isinstance(val, dict):
            # datasets library image wrapper
            img_bytes = val.get("bytes")
            if img_bytes:
                try:
                    import io
                    return PILImage.open(io.BytesIO(img_bytes))
                except Exception:  # noqa: BLE001
                    pass
    return None
