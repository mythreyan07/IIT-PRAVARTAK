"""
HuggingFace dataset loader for xBD (xView2 Building Damage Assessment).

Primary dataset:  kshitijrajsharma/xview2-xbd
Fallback dataset: hannan022/xview2-xbd
QA dataset:       DakshJ27/xbd-damage-qa

Tokens are read from the HF_TOKEN environment variable or accepted as a
parameter — never hardcoded, never printed or logged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

# Damage class mapping (xBD standard)
_DAMAGE_CLASSES: Dict[int, str] = {
    0: "no_damage",
    1: "minor_damage",
    2: "major_damage",
    3: "destroyed",
}

# xBD datasets in preference order
_XBD_DATASET_IDS = [
    "kshitijrajsharma/xview2-xbd",
    "hannan022/xview2-xbd",
]

_QA_DATASET_ID = "DakshJ27/xbd-damage-qa"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_damage_class_names() -> Dict[int, str]:
    """Return the standard xBD damage class mapping."""
    return dict(_DAMAGE_CLASSES)


def load_xbd_dataset(
    dataset_id: Optional[str] = None,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
):
    """
    Load the xBD dataset from HuggingFace.

    Tries dataset_id first; if None, tries each ID in _XBD_DATASET_IDS.
    Returns a datasets.Dataset (single split) or datasets.DatasetDict (all splits).

    Raises DatasetLoadError if all attempts fail.
    """
    _check_datasets_import()
    from datasets import load_dataset, DatasetDict

    resolved_token = _resolve_token(token)
    ids_to_try = [dataset_id] if dataset_id else list(_XBD_DATASET_IDS)

    last_exc: Optional[Exception] = None
    for ds_id in ids_to_try:
        try:
            logger.info("Attempting to load dataset: %s (split=%s)", ds_id, split or "all")
            kwargs: Dict[str, Any] = {
                "cache_dir": cache_dir,
                "trust_remote_code": True,
            }
            if resolved_token:
                kwargs["token"] = resolved_token
            if split:
                kwargs["split"] = split
            ds = load_dataset(ds_id, **kwargs)
            logger.info("Successfully loaded dataset: %s", ds_id)
            return ds
        except Exception as exc:  # noqa: BLE001
            _safe_log_error(ds_id, exc)
            last_exc = exc

    raise DatasetLoadError(
        f"Could not load any xBD dataset. Last error: {last_exc}"
    )


def load_qa_dataset(
    dataset_id: str = _QA_DATASET_ID,
    split: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    """
    Load the xBD QA dataset (DakshJ27/xbd-damage-qa).

    Returns a datasets.Dataset or DatasetDict.
    """
    _check_datasets_import()
    from datasets import load_dataset

    try:
        logger.info("Loading QA dataset: %s (split=%s)", dataset_id, split or "all")
        kwargs: Dict[str, Any] = {
            "cache_dir": cache_dir,
            "trust_remote_code": True,
        }
        if split:
            kwargs["split"] = split
        ds = load_dataset(dataset_id, **kwargs)
        logger.info("QA dataset loaded successfully.")
        return ds
    except Exception as exc:  # noqa: BLE001
        _safe_log_error(dataset_id, exc)
        raise DatasetLoadError(f"Could not load QA dataset '{dataset_id}': {exc}") from exc


def get_image_pair(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract before/after PIL images from a dataset sample.

    Supports multiple possible field naming conventions used by different
    xBD mirrors on HuggingFace.

    Returns:
        {
            "before": PIL.Image,
            "after":  PIL.Image,
            "pair_id": str,
        }
    Raises SampleParseError if neither image can be extracted.
    """
    before = _extract_image(sample, ["pre_image", "before", "image_pre", "pre"])
    after = _extract_image(sample, ["post_image", "after", "image_post", "post"])

    if before is None:
        raise SampleParseError("Could not find 'before' image in sample. "
                               f"Available keys: {list(sample.keys())}")
    if after is None:
        raise SampleParseError("Could not find 'after' image in sample. "
                               f"Available keys: {list(sample.keys())}")

    pair_id = str(sample.get("pair_id", sample.get("scene_id", sample.get("id", "unknown"))))
    return {"before": before, "after": after, "pair_id": pair_id}


def get_building_annotations(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract building polygon annotations from a dataset sample.

    Returns a list of dicts::

        [
            {
                "uid": str,
                "polygon": [[x, y], ...],
                "damage_class": int,   # 0-3
                "damage_label": str,
            },
            ...
        ]

    Returns an empty list if no annotations are found (never raises).
    """
    # Field naming conventions differ across dataset mirrors
    for key in ["buildings", "annotations", "labels", "features"]:
        raw = sample.get(key)
        if raw is not None:
            return _parse_annotations(raw)

    # Some datasets embed a GeoJSON-like structure
    for key in ["damage_polygons_xy", "building_damage_xy"]:
        raw = sample.get(key)
        if raw is not None:
            return _parse_geojson_annotations(raw)

    # Try to decode from JSON string fields
    for key in ["annotation", "label_json"]:
        raw = sample.get(key)
        if isinstance(raw, str):
            try:
                return _parse_annotations(json.loads(raw))
            except (json.JSONDecodeError, TypeError):
                pass

    logger.debug("No building annotations found in sample with keys: %s", list(sample.keys()))
    return []


def download_and_cache(
    dataset_id: Optional[str] = None,
    cache_dir: str = "data/hf_cache",
    split: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """
    Download and cache the dataset locally.

    Returns the path to the cache directory.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    ds = load_xbd_dataset(dataset_id=dataset_id, split=split,
                          cache_dir=str(cache_path), token=token)
    logger.info("Dataset cached at: %s", cache_path)
    return str(cache_path)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DatasetLoadError(RuntimeError):
    """Raised when the dataset cannot be loaded from HuggingFace."""


class SampleParseError(ValueError):
    """Raised when a required field cannot be extracted from a sample."""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_datasets_import() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required. Install it with: "
            "pip install datasets"
        ) from exc


def _resolve_token(explicit_token: Optional[str]) -> Optional[str]:
    """Return the HF token without logging its value."""
    if explicit_token:
        return explicit_token
    env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not env_token:
        logger.debug("No HF_TOKEN found; proceeding without authentication (public datasets only).")
    return env_token or None


def _safe_log_error(dataset_id: str, exc: Exception) -> None:
    """Log dataset load failure without exposing any token value."""
    msg = str(exc)
    # Strip anything that looks like a bearer token from the message
    if "token" in msg.lower() or "bearer" in msg.lower() or "hf_" in msg.lower():
        msg = "[error message redacted — may contain credential details]"
    logger.warning("Failed to load '%s': %s", dataset_id, msg)


def _extract_image(
    sample: Dict[str, Any],
    keys: List[str],
) -> Optional[Image.Image]:
    """Try a list of key names and return the first PIL Image found."""
    for key in keys:
        val = sample.get(key)
        if val is None:
            continue
        if isinstance(val, Image.Image):
            return val.convert("RGB")
        if isinstance(val, np.ndarray):
            return Image.fromarray(val.astype(np.uint8)).convert("RGB")
        if isinstance(val, (str, Path)) and Path(str(val)).exists():
            try:
                return Image.open(str(val)).convert("RGB")
            except Exception:  # noqa: BLE001
                pass
        # datasets library may wrap images in a dict {"bytes": ..., "path": ...}
        if isinstance(val, dict):
            img_bytes = val.get("bytes")
            img_path = val.get("path")
            if img_bytes:
                import io
                try:
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:  # noqa: BLE001
                    pass
            if img_path and Path(str(img_path)).exists():
                try:
                    return Image.open(str(img_path)).convert("RGB")
                except Exception:  # noqa: BLE001
                    pass
    return None


def _parse_annotations(raw: Any) -> List[Dict[str, Any]]:
    """Parse annotations from a list or dict into the standard format."""
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    result = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        damage_int = _resolve_damage_class(
            item.get("damage_class", item.get("label", item.get("damage_type", 0)))
        )
        polygon = _resolve_polygon(item.get("polygon", item.get("coordinates", item.get("points", []))))
        result.append({
            "uid": str(item.get("uid", item.get("id", f"building_{i}"))),
            "polygon": polygon,
            "damage_class": damage_int,
            "damage_label": _DAMAGE_CLASSES.get(damage_int, "unknown"),
        })
    return result


def _parse_geojson_annotations(raw: Any) -> List[Dict[str, Any]]:
    """Parse GeoJSON-style feature collections."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(raw, dict):
        return []
    features = raw.get("features", [])
    result = []
    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            continue
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = []
        if isinstance(geom, dict) and geom.get("type") == "Polygon":
            rings = geom.get("coordinates", [[]])
            coords = [[float(x), float(y)] for x, y in (rings[0] if rings else [])]
        damage_int = _resolve_damage_class(props.get("subtype", props.get("damage_class", 0)))
        result.append({
            "uid": str(props.get("uid", f"building_{i}")),
            "polygon": coords,
            "damage_class": damage_int,
            "damage_label": _DAMAGE_CLASSES.get(damage_int, "unknown"),
        })
    return result


def _resolve_damage_class(val: Any) -> int:
    """Convert various damage class representations to integer 0-3."""
    if isinstance(val, int):
        return max(0, min(3, val))
    if isinstance(val, float):
        return max(0, min(3, int(val)))
    if isinstance(val, str):
        mapping = {
            "no-damage": 0, "no_damage": 0, "nodamage": 0,
            "minor-damage": 1, "minor_damage": 1, "minordamage": 1,
            "major-damage": 2, "major_damage": 2, "majordamage": 2,
            "destroyed": 3, "un-classified": 0, "unclassified": 0,
        }
        return mapping.get(val.lower().strip(), 0)
    return 0


def _resolve_polygon(raw: Any) -> List[List[float]]:
    """Normalise polygon to list of [x, y] pairs."""
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        result = []
        for pt in raw:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                result.append([float(pt[0]), float(pt[1])])
            elif isinstance(pt, dict):
                result.append([float(pt.get("x", 0)), float(pt.get("y", 0))])
        return result
    return []
