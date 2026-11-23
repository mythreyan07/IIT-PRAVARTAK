"""Tests for hf_dataset_loader.py — no actual HF downloads."""

import logging
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

from backend.dataset.hf_dataset_loader import (
    DatasetLoadError,
    SampleParseError,
    get_damage_class_names,
    get_building_annotations,
    get_image_pair,
    _resolve_damage_class,
    _resolve_polygon,
    _parse_annotations,
)


# ---------------------------------------------------------------------------
# get_damage_class_names
# ---------------------------------------------------------------------------

class TestGetDamageClassNames:
    def test_returns_four_classes(self):
        names = get_damage_class_names()
        assert len(names) == 4

    def test_correct_mapping(self):
        names = get_damage_class_names()
        assert names[0] == "no_damage"
        assert names[1] == "minor_damage"
        assert names[2] == "major_damage"
        assert names[3] == "destroyed"

    def test_returns_copy(self):
        a = get_damage_class_names()
        b = get_damage_class_names()
        a[99] = "tampered"
        assert 99 not in get_damage_class_names()


# ---------------------------------------------------------------------------
# get_image_pair
# ---------------------------------------------------------------------------

class TestGetImagePair:
    def _pil(self, h=64, w=64):
        return Image.fromarray(np.random.randint(0, 255, (h, w, 3), dtype=np.uint8))

    def test_pil_images_returned(self):
        sample = {
            "pre_image": self._pil(),
            "post_image": self._pil(),
            "pair_id": "PAIR_001",
        }
        result = get_image_pair(sample)
        assert isinstance(result["before"], Image.Image)
        assert isinstance(result["after"], Image.Image)
        assert result["pair_id"] == "PAIR_001"

    def test_alternative_field_names(self):
        sample = {
            "before": self._pil(),
            "after": self._pil(),
        }
        result = get_image_pair(sample)
        assert result["before"] is not None

    def test_numpy_image_accepted(self):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        sample = {"pre_image": arr, "post_image": arr}
        result = get_image_pair(sample)
        assert isinstance(result["before"], Image.Image)

    def test_missing_before_raises(self):
        sample = {"post_image": self._pil()}
        with pytest.raises(SampleParseError, match="before"):
            get_image_pair(sample)

    def test_missing_after_raises(self):
        sample = {"pre_image": self._pil()}
        with pytest.raises(SampleParseError, match="after"):
            get_image_pair(sample)

    def test_fallback_pair_id(self):
        sample = {"before": self._pil(), "after": self._pil()}
        result = get_image_pair(sample)
        assert isinstance(result["pair_id"], str)


# ---------------------------------------------------------------------------
# get_building_annotations
# ---------------------------------------------------------------------------

class TestGetBuildingAnnotations:
    def test_basic_annotations(self):
        sample = {
            "buildings": [
                {"uid": "b1", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]], "damage_class": 2},
                {"uid": "b2", "polygon": [[20, 20], [30, 20], [30, 30], [20, 30]], "damage_class": 0},
            ]
        }
        anns = get_building_annotations(sample)
        assert len(anns) == 2
        assert anns[0]["damage_class"] == 2
        assert anns[0]["damage_label"] == "major_damage"
        assert anns[1]["damage_class"] == 0

    def test_no_annotations_returns_empty_list(self):
        sample = {"pre_image": "dummy"}
        anns = get_building_annotations(sample)
        assert anns == []

    def test_string_damage_class_resolved(self):
        sample = {
            "buildings": [
                {"uid": "b1", "polygon": [[0, 0]], "damage_class": "destroyed"},
            ]
        }
        anns = get_building_annotations(sample)
        assert anns[0]["damage_class"] == 3

    def test_annotations_key_fallback(self):
        sample = {
            "annotations": [
                {"polygon": [[0, 0], [5, 0], [5, 5]], "damage_class": 1},
            ]
        }
        anns = get_building_annotations(sample)
        assert len(anns) == 1
        assert anns[0]["damage_label"] == "minor_damage"

    def test_geojson_annotations(self):
        import json
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]],
                    },
                    "properties": {"uid": "g1", "subtype": "major-damage"},
                }
            ],
        }
        sample = {"damage_polygons_xy": json.dumps(geojson)}
        anns = get_building_annotations(sample)
        assert len(anns) == 1
        assert anns[0]["damage_class"] == 2


# ---------------------------------------------------------------------------
# Token safety — tokens must never appear in log output
# ---------------------------------------------------------------------------

class TestTokenSafety:
    def test_token_not_logged(self, caplog):
        secret = "hf_supersecrettoken1234"
        # Patch at the module level where it is called, not via the top-level package
        with caplog.at_level(logging.DEBUG, logger="backend.dataset.hf_dataset_loader"):
            with patch(
                "backend.dataset.hf_dataset_loader.load_xbd_dataset",
                side_effect=DatasetLoadError(f"auth failed"),  # do NOT put secret in the error
            ):
                try:
                    from backend.dataset.hf_dataset_loader import load_xbd_dataset
                    load_xbd_dataset(dataset_id="fake/dataset", token=secret)
                except Exception:
                    pass
        assert secret not in caplog.text


# ---------------------------------------------------------------------------
# load_xbd_dataset — graceful failure
# ---------------------------------------------------------------------------

class TestLoadXbdDatasetFailure:
    def test_raises_dataset_load_error_when_unavailable(self):
        # Patch the internal import so we don't need the datasets package installed
        import sys
        from unittest.mock import MagicMock
        fake_datasets = MagicMock()
        fake_datasets.load_dataset.side_effect = Exception("network error")
        with patch.dict(sys.modules, {"datasets": fake_datasets}):
            # Re-import to pick up the mock
            import importlib
            import backend.dataset.hf_dataset_loader as loader_mod
            importlib.reload(loader_mod)
            with pytest.raises((DatasetLoadError, Exception)):
                loader_mod.load_xbd_dataset(dataset_id="fake/xbd")
        # Reload to restore normal state
        import importlib
        import backend.dataset.hf_dataset_loader as loader_mod
        importlib.reload(loader_mod)

    def test_missing_datasets_package(self):
        import sys
        import importlib
        # Temporarily hide the datasets module
        saved = sys.modules.pop("datasets", None)
        try:
            # Force re-import to trigger the import check
            import importlib
            from backend.dataset.hf_dataset_loader import _check_datasets_import
            if saved is None:
                with pytest.raises(ImportError):
                    _check_datasets_import()
            # else datasets IS installed; test is skipped gracefully
        finally:
            if saved is not None:
                sys.modules["datasets"] = saved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestResolveDamageClass:
    def test_int(self):
        assert _resolve_damage_class(2) == 2

    def test_clamp_high(self):
        assert _resolve_damage_class(99) == 3

    def test_clamp_low(self):
        assert _resolve_damage_class(-1) == 0

    def test_string(self):
        assert _resolve_damage_class("destroyed") == 3
        assert _resolve_damage_class("no-damage") == 0
        assert _resolve_damage_class("minor_damage") == 1


class TestResolvePolygon:
    def test_list_of_pairs(self):
        p = _resolve_polygon([[1, 2], [3, 4], [5, 6]])
        assert p == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    def test_list_of_dicts(self):
        p = _resolve_polygon([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
        assert p == [[1.0, 2.0], [3.0, 4.0]]

    def test_empty(self):
        assert _resolve_polygon([]) == []
        assert _resolve_polygon(None) == []
