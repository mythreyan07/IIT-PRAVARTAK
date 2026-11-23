"""Tests for evidence_generator.py"""
import json
import pytest

from backend.reasoning.evidence_generator import (
    EvidenceItem,
    EvidenceGeneratorConfig,
    aggregate_scene_evidence,
    classify_damage_severity,
    compute_confidence,
    evidence_to_dict,
    generate_evidence,
)


def _synthetic_assessment(n=3):
    buildings = [
        {"building_id": f"BLD_{i:03d}", "damage_class": i % 4,
         "proba": [0.1, 0.2, 0.3, 0.4][:(4)], "bbox": [10*i, 10*i, 50*i+50, 50*i+50]}
        for i in range(n)
    ]
    # Fix: each proba must be a 4-element list summing to 1
    for b in buildings:
        cls = b["damage_class"]
        proba = [0.05, 0.05, 0.05, 0.05]
        proba[cls] = 0.85
        b["proba"] = proba
    return {
        "buildings": buildings,
        "alignment_quality": 0.85,
        "before_quality": {"overall_quality": "Good"},
        "after_quality": {"overall_quality": "Good"},
    }


class TestGenerateEvidence:
    def test_returns_one_item_per_building(self):
        result = _synthetic_assessment(3)
        items = generate_evidence(result)
        assert len(items) == 3

    def test_evidence_item_fields(self):
        result = _synthetic_assessment(1)
        items = generate_evidence(result)
        item = items[0]
        assert isinstance(item.building_id, str)
        assert isinstance(item.damage_class, int)
        assert 0.0 <= item.confidence <= 1.0
        assert isinstance(item.evidence_text, str) and len(item.evidence_text) > 0
        assert isinstance(item.reasoning_steps, list) and len(item.reasoning_steps) > 0

    def test_empty_buildings_list(self):
        result = {"buildings": [], "alignment_quality": 0.9,
                  "before_quality": {}, "after_quality": {}}
        items = generate_evidence(result)
        assert items == []


class TestComputeConfidence:
    def test_returns_float_in_range(self):
        c = compute_confidence([0.1, 0.1, 0.1, 0.7], 0.9, 0.8)
        assert 0.0 <= c <= 1.0

    def test_empty_proba_returns_zero(self):
        assert compute_confidence([], 0.9, 0.8) == 0.0

    def test_perfect_inputs_near_one(self):
        c = compute_confidence([0.0, 0.0, 0.0, 1.0], 1.0, 1.0)
        assert c > 0.8


class TestClassifyDamageSeverity:
    def test_no_damage(self):
        s = classify_damage_severity(0, 0.9)
        assert s["severity"] == "none"
        assert s["action_required"] is False

    def test_minor_damage(self):
        s = classify_damage_severity(1, 0.9)
        assert s["severity"] == "low"
        assert s["action_required"] is False

    def test_major_damage_requires_action(self):
        s = classify_damage_severity(2, 0.9)
        assert s["severity"] == "high"
        assert s["action_required"] is True

    def test_destroyed_requires_action(self):
        s = classify_damage_severity(3, 0.9)
        assert s["severity"] == "total"
        assert s["action_required"] is True

    def test_low_confidence_suppresses_action(self):
        s = classify_damage_severity(2, 0.1)
        assert s["action_required"] is False


class TestAggregateSceneEvidence:
    def test_counts_match(self):
        result = _synthetic_assessment(4)
        items = generate_evidence(result)
        summary = aggregate_scene_evidence(items)
        assert summary["total_buildings"] == 4
        assert summary["total_buildings"] == summary["damaged_buildings"] + summary["undamaged_buildings"]

    def test_mean_confidence_in_range(self):
        items = generate_evidence(_synthetic_assessment(3))
        summary = aggregate_scene_evidence(items)
        assert 0.0 <= summary["mean_confidence"] <= 1.0

    def test_empty_items(self):
        summary = aggregate_scene_evidence([])
        assert summary["total_buildings"] == 0


class TestEvidenceToDict:
    def test_json_serialisable(self):
        items = generate_evidence(_synthetic_assessment(2))
        for item in items:
            d = evidence_to_dict(item)
            json.dumps(d)  # must not raise

    def test_has_required_keys(self):
        items = generate_evidence(_synthetic_assessment(1))
        d = evidence_to_dict(items[0])
        for k in ("building_id", "damage_class", "confidence", "evidence_text"):
            assert k in d
