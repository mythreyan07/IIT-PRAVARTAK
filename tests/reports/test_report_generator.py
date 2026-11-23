"""Tests for report_generator.py"""

import json
from pathlib import Path

import pytest

from backend.reports.report_generator import (
    DamageReport,
    create_damage_report,
    format_report_summary,
    generate_json_report,
    load_report,
)
from backend.reasoning.evidence_generator import EvidenceItem


def _make_evidence_items(n=3):
    items = []
    for i in range(n):
        items.append(EvidenceItem(
            building_id=f"B{i:03d}",
            damage_class=i % 4,
            confidence=0.75,
            before_quality="Good",
            after_quality="Good",
            alignment_quality=0.85,
            coordinates=[[0, 0], [10, 0], [10, 10], [0, 10]],
            evidence_text="Building shows damage",
            reasoning_steps=["Step 1", "Step 2"],
        ))
    return items


def _make_report() -> DamageReport:
    items = _make_evidence_items()
    qa = [{"question": "Q1?", "answer": "A1."}, {"question": "Q2?", "answer": "A2."}]
    return create_damage_report(
        scene_id="SCENE_001",
        evidence_items=items,
        qa_pairs=qa,
        image_paths={"before": "data/before.jpg", "after": "data/after.jpg"},
        disaster_type="earthquake",
    )


class TestCreateDamageReport:
    def test_returns_damage_report(self):
        report = _make_report()
        assert isinstance(report, DamageReport)

    def test_total_buildings_matches_evidence(self):
        report = _make_report()
        assert report.total_buildings == 3

    def test_scene_id_preserved(self):
        report = _make_report()
        assert report.scene_id == "SCENE_001"

    def test_disaster_type_preserved(self):
        report = _make_report()
        assert report.disaster_type == "earthquake"

    def test_has_report_id(self):
        report = _make_report()
        assert report.report_id and len(report.report_id) > 0

    def test_has_timestamp(self):
        report = _make_report()
        assert report.timestamp and len(report.timestamp) > 0


class TestGenerateJsonReport:
    def test_saves_valid_json(self, tmp_path):
        report = _make_report()
        dest = tmp_path / "report.json"
        result_path = generate_json_report(report, dest)
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data["scene_id"] == "SCENE_001"

    def test_creates_parent_dirs(self, tmp_path):
        report = _make_report()
        dest = tmp_path / "subdir" / "nested" / "report.json"
        generate_json_report(report, dest)
        assert dest.exists()


class TestLoadReport:
    def test_roundtrip(self, tmp_path):
        report = _make_report()
        dest = tmp_path / "report.json"
        generate_json_report(report, dest)
        loaded = load_report(dest)
        assert loaded.scene_id == report.scene_id
        assert loaded.total_buildings == report.total_buildings

    def test_returns_damage_report_type(self, tmp_path):
        report = _make_report()
        dest = tmp_path / "r.json"
        generate_json_report(report, dest)
        loaded = load_report(dest)
        assert isinstance(loaded, DamageReport)


class TestFormatReportSummary:
    def test_returns_non_empty_string(self):
        report = _make_report()
        summary = format_report_summary(report)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_contains_scene_id(self):
        report = _make_report()
        summary = format_report_summary(report)
        assert "SCENE_001" in summary


class TestPdfGeneration:
    def test_pdf_graceful_when_fpdf_missing(self, tmp_path):
        """PDF generation should return None gracefully if fpdf2 not installed."""
        from backend.reports.report_generator import generate_pdf_report
        import sys
        # Mock fpdf as unavailable
        orig = sys.modules.get("fpdf")
        sys.modules["fpdf"] = None  # type: ignore[assignment]
        try:
            report = _make_report()
            result = generate_pdf_report(report, None, tmp_path / "report.pdf")
            # Should return None or a Path, never raise
            assert result is None or isinstance(result, Path)
        finally:
            if orig is None:
                sys.modules.pop("fpdf", None)
            else:
                sys.modules["fpdf"] = orig
