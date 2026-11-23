"""Tests for damage_heatmap.py"""

import numpy as np
import pytest

from backend.visualization.damage_heatmap import (
    create_damage_colormap,
    create_damage_heatmap,
    create_building_overlay,
    generate_scene_heatmap,
    plot_damage_distribution,
)
from backend.reasoning.evidence_generator import EvidenceItem


def _make_rgb(h=128, w=128, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_evidence_items(n=4):
    items = []
    for i in range(n):
        items.append(EvidenceItem(
            building_id=f"B{i:03d}",
            damage_class=i % 4,
            confidence=0.8,
            before_quality="Good",
            after_quality="Good",
            alignment_quality=0.9,
            coordinates=[[10, 10], [50, 10], [50, 50], [10, 50]],
            evidence_text="Test evidence",
            reasoning_steps=[],
        ))
    return items


class TestCreateDamageColormap:
    def test_returns_four_keys(self):
        cm = create_damage_colormap()
        assert set(cm.keys()) == {0, 1, 2, 3}

    def test_values_are_rgb_tuples(self):
        cm = create_damage_colormap()
        for v in cm.values():
            assert len(v) == 3
            assert all(0 <= c <= 255 for c in v)


class TestCreateDamageHeatmap:
    def test_output_shape(self):
        img = _make_rgb(128, 128)
        damage_map = np.zeros((128, 128), dtype=np.uint8)
        damage_map[32:96, 32:96] = 2
        result = create_damage_heatmap(img, damage_map)
        assert result.shape == (128, 128, 3)
        assert result.dtype == np.uint8

    def test_no_damage_returns_similar_to_input(self):
        img = _make_rgb(64, 64)
        damage_map = np.zeros((64, 64), dtype=np.uint8)
        result = create_damage_heatmap(img, damage_map, alpha=0.0)
        assert result.shape == (64, 64, 3)


class TestCreateBuildingOverlay:
    def test_output_shape(self):
        img = _make_rgb(256, 256)
        buildings = [{"polygon": [[10, 10], [50, 10], [50, 50], [10, 50]]}]
        damage_classes = [1]
        result = create_building_overlay(img, buildings, damage_classes)
        assert result.shape == (256, 256, 3)

    def test_empty_buildings(self):
        img = _make_rgb(64, 64)
        result = create_building_overlay(img, [], [])
        assert result.shape == (64, 64, 3)


class TestGenerateSceneHeatmap:
    def test_returns_required_keys(self):
        before = _make_rgb(128, 128, seed=1)
        after = _make_rgb(128, 128, seed=2)
        buildings = [
            {"polygon": [[10, 10], [50, 10], [50, 50], [10, 50]],
             "damage_class": 2, "confidence": 0.8}
        ]
        result = generate_scene_heatmap(before, after, buildings)
        assert "heatmap" in result
        assert "legend" in result
        assert "stats" in result

    def test_heatmap_is_array(self):
        before = _make_rgb(128, 128)
        after = _make_rgb(128, 128)
        result = generate_scene_heatmap(before, after, [])
        assert isinstance(result["heatmap"], np.ndarray)


class TestPlotDamageDistribution:
    def test_returns_plotly_figure(self):
        import plotly.graph_objects as go
        items = _make_evidence_items(4)
        fig = plot_damage_distribution(items)
        assert isinstance(fig, go.Figure)

    def test_empty_items(self):
        import plotly.graph_objects as go
        fig = plot_damage_distribution([])
        assert isinstance(fig, go.Figure)
