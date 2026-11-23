"""
PDF and JSON report generation for EAGLE-DAR damage assessments.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    output_dir: str = "reports"
    include_images: bool = True
    pdf_title: str = "EAGLE-DAR Damage Assessment Report"
    organization_name: str = "EAGLE-DAR"
    include_qa: bool = True


@dataclass
class DamageReport:
    report_id: str
    timestamp: str
    scene_id: str
    disaster_type: str
    total_buildings: int
    damaged_buildings: int
    by_damage_class: Dict[str, int]
    overall_severity: str
    evidence_items: List[Dict[str, Any]]
    qa_pairs: List[Dict[str, str]]
    image_paths: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_damage_report(
    scene_id: str,
    evidence_items: List[Any],
    qa_pairs: List[Dict[str, str]],
    image_paths: Dict[str, str],
    disaster_type: str = "unknown",
) -> DamageReport:
    """Build a DamageReport from evidence items and QA pairs."""
    from backend.reasoning.evidence_generator import aggregate_scene_evidence, evidence_to_dict

    summary = aggregate_scene_evidence(evidence_items)
    evidence_dicts = [evidence_to_dict(e) for e in evidence_items]

    return DamageReport(
        report_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        scene_id=scene_id,
        disaster_type=disaster_type,
        total_buildings=summary["total_buildings"],
        damaged_buildings=summary["damaged_buildings"],
        by_damage_class=summary["by_damage_class"],
        overall_severity=summary["overall_severity"],
        evidence_items=evidence_dicts,
        qa_pairs=qa_pairs,
        image_paths=image_paths,
        metadata={
            "mean_confidence": summary.get("mean_confidence"),
            "action_required_count": summary.get("action_required_count"),
        },
    )


def generate_json_report(
    damage_report: DamageReport,
    output_path: str | Path,
) -> Path:
    """Serialise report to a JSON file and return the path."""
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(damage_report)
    dest.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("JSON report saved: %s", dest)
    return dest


def generate_pdf_report(
    damage_report: DamageReport,
    heatmap_image: Optional[Any],  # np.ndarray or None
    output_path: str | Path,
    config: ReportConfig | None = None,
) -> Optional[Path]:
    """
    Generate a PDF report. Returns Path on success, None if fpdf2 unavailable.
    """
    cfg = config or ReportConfig()
    try:
        from fpdf import FPDF
    except ImportError:
        logger.warning("fpdf2 not installed — PDF report cannot be generated. Install with: pip install fpdf2")
        return None

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, cfg.pdf_title, ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {damage_report.timestamp}", ln=True, align="C")
    pdf.cell(0, 6, f"Organisation: {cfg.organization_name}", ln=True, align="C")
    pdf.ln(5)

    # Executive summary
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Executive Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Scene ID: {damage_report.scene_id}", ln=True)
    pdf.cell(0, 6, f"Disaster Type: {damage_report.disaster_type}", ln=True)
    pdf.cell(0, 6, f"Total Buildings Assessed: {damage_report.total_buildings}", ln=True)
    pdf.cell(0, 6, f"Damaged Buildings: {damage_report.damaged_buildings}", ln=True)
    pdf.cell(0, 6, f"Overall Severity: {damage_report.overall_severity.upper()}", ln=True)
    pdf.ln(3)

    # Damage breakdown
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Damage Breakdown", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for cls_name, count in damage_report.by_damage_class.items():
        pdf.cell(0, 6, f"  {cls_name.replace('_', ' ').title()}: {count}", ln=True)
    pdf.ln(3)

    # Heatmap image
    if cfg.include_images and heatmap_image is not None:
        try:
            import tempfile, cv2
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            if hasattr(heatmap_image, "shape"):  # numpy array
                import cv2 as _cv2
                _cv2.imwrite(tmp_path, _cv2.cvtColor(heatmap_image, _cv2.COLOR_RGB2BGR))
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 7, "Damage Heatmap", ln=True)
                available_w = pdf.w - pdf.l_margin - pdf.r_margin
                pdf.image(tmp_path, w=available_w)
                pdf.ln(3)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not embed heatmap in PDF: %s", exc)

    # Building table (top 20)
    if damage_report.evidence_items:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Building Details (up to 20)", ln=True)
        pdf.set_font("Helvetica", "B", 9)
        col_w = [40, 50, 35, 35]
        headers = ["Building ID", "Damage Class", "Confidence", "Action Required"]
        for h, w in zip(headers, col_w):
            pdf.cell(w, 6, h, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        for ev in damage_report.evidence_items[:20]:
            pdf.cell(col_w[0], 6, str(ev.get("building_id", ""))[:18], border=1)
            cls = ev.get("damage_class", 0)
            cls_name = ["No Damage", "Minor", "Major", "Destroyed"][max(0, min(3, int(cls)))]
            pdf.cell(col_w[1], 6, cls_name, border=1)
            conf = ev.get("confidence", 0)
            pdf.cell(col_w[2], 6, f"{float(conf):.2%}", border=1)
            sev = ev.get("reasoning_steps", [])
            action = "Yes" if int(cls) >= 2 else "No"
            pdf.cell(col_w[3], 6, action, border=1)
            pdf.ln()
        pdf.ln(3)

    # Q&A section
    if cfg.include_qa and damage_report.qa_pairs:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Questions & Answers", ln=True)
        pdf.set_font("Helvetica", "", 10)
        for qa in damage_report.qa_pairs:
            pdf.set_font("Helvetica", "B", 10)
            pdf.multi_cell(0, 6, f"Q: {qa.get('question', '')}")
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 6, f"A: {qa.get('answer', '')}")
            pdf.ln(2)

    pdf.output(str(dest))
    logger.info("PDF report saved: %s", dest)
    return dest


def load_report(path: str | Path) -> DamageReport:
    """Load a DamageReport from a saved JSON file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DamageReport(**data)


def format_report_summary(report: DamageReport) -> str:
    pct = (report.damaged_buildings / report.total_buildings * 100) if report.total_buildings else 0
    lines = [
        f"=== EAGLE-DAR Damage Assessment Report ===",
        f"Scene: {report.scene_id}  |  Disaster: {report.disaster_type}",
        f"Timestamp: {report.timestamp}",
        f"Total buildings: {report.total_buildings}",
        f"Damaged: {report.damaged_buildings} ({pct:.1f}%)",
        f"Overall severity: {report.overall_severity.upper()}",
        "Breakdown:",
    ]
    for cls_name, count in report.by_damage_class.items():
        lines.append(f"  {cls_name.replace('_', ' ').title()}: {count}")
    return "\n".join(lines)
