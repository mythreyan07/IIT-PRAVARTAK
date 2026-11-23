"""
Evidence generation from damage assessment results.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DAMAGE_CLASS_NAMES = {
    0: "no_damage",
    1: "minor_damage",
    2: "major_damage",
    3: "destroyed",
}

SEVERITY_MAP = {
    0: {"severity": "none",  "label": "No Damage",    "action_required": False},
    1: {"severity": "low",   "label": "Minor Damage", "action_required": False},
    2: {"severity": "high",  "label": "Major Damage", "action_required": True},
    3: {"severity": "total", "label": "Destroyed",    "action_required": True},
}


@dataclass
class EvidenceItem:
    building_id: str
    damage_class: int
    confidence: float
    before_quality: Dict[str, Any]
    after_quality: Dict[str, Any]
    alignment_quality: float
    coordinates: Optional[List]
    evidence_text: str
    reasoning_steps: List[str] = field(default_factory=list)


@dataclass
class EvidenceGeneratorConfig:
    min_confidence: float = 0.5
    include_coordinates: bool = True
    include_quality_metrics: bool = True


def compute_confidence(
    damage_proba: List[float],
    alignment_quality: float,
    image_quality: float,
) -> float:
    """Combine model probability, alignment and image quality into one confidence score."""
    if not damage_proba:
        return 0.0
    max_prob = max(damage_proba)
    # Clamp inputs
    align = max(0.0, min(1.0, alignment_quality))
    img_q = max(0.0, min(1.0, image_quality))
    # Weighted combination
    confidence = 0.6 * max_prob + 0.25 * align + 0.15 * img_q
    return round(float(confidence), 4)


def classify_damage_severity(damage_class: int, confidence: float) -> Dict[str, Any]:
    """Return severity dict for a given damage class and confidence."""
    base = SEVERITY_MAP.get(damage_class, SEVERITY_MAP[0]).copy()
    # Downgrade action_required when confidence is low
    if confidence < 0.4 and base["action_required"]:
        base = base.copy()
        base["action_required"] = False
    return base


def format_evidence_text(item: EvidenceItem) -> str:
    class_name = DAMAGE_CLASS_NAMES.get(item.damage_class, "unknown")
    severity = classify_damage_severity(item.damage_class, item.confidence)
    lines = [
        f"Building {item.building_id}: {severity['label']} (confidence {item.confidence:.2%})",
        f"  Damage class: {item.damage_class} ({class_name})",
        f"  Alignment quality: {item.alignment_quality:.3f}",
    ]
    if item.before_quality:
        lines.append(f"  Before image quality: {item.before_quality.get('overall_quality', 'N/A')}")
    if item.after_quality:
        lines.append(f"  After image quality: {item.after_quality.get('overall_quality', 'N/A')}")
    if item.coordinates:
        lines.append(f"  Coordinates: {item.coordinates[:2]}...")
    return "\n".join(lines)


def generate_evidence(
    assessment_result: Dict[str, Any],
    config: EvidenceGeneratorConfig | None = None,
) -> List[EvidenceItem]:
    """
    Convert a damage assessment result dict into a list of EvidenceItems.

    assessment_result keys:
        buildings: list of {damage_class, proba, bbox, polygon, building_id (optional)}
        alignment_quality: float
        before_quality: dict
        after_quality: dict
    """
    cfg = config or EvidenceGeneratorConfig()
    buildings = assessment_result.get("buildings", [])
    alignment_quality = float(assessment_result.get("alignment_quality", 0.5))
    before_quality = assessment_result.get("before_quality", {})
    after_quality = assessment_result.get("after_quality", {})

    # Derive a scalar image quality from the quality dicts
    def _quality_score(q: Dict) -> float:
        mapping = {"Good": 1.0, "Warning": 0.6, "Poor": 0.2}
        return mapping.get(q.get("overall_quality", "Good"), 0.6)

    img_q = (_quality_score(before_quality) + _quality_score(after_quality)) / 2.0

    items: List[EvidenceItem] = []
    for idx, bld in enumerate(buildings):
        bld_id = str(bld.get("building_id", f"BLD_{idx:04d}"))
        damage_class = int(bld.get("damage_class", 0))
        proba = bld.get("proba", [1.0, 0.0, 0.0, 0.0])
        confidence = compute_confidence(proba, alignment_quality, img_q)

        if confidence < cfg.min_confidence:
            logger.debug("Building %s skipped — confidence %.2f below threshold.", bld_id, confidence)
            # Still include but mark low confidence
            pass

        coords = bld.get("polygon") or bld.get("bbox") if cfg.include_coordinates else None
        severity = classify_damage_severity(damage_class, confidence)

        steps = [
            f"Model predicted damage class {damage_class} ({DAMAGE_CLASS_NAMES.get(damage_class, '?')})",
            f"Class probability: {proba[damage_class]:.3f}",
            f"Alignment quality: {alignment_quality:.3f}",
            f"Combined confidence: {confidence:.3f}",
            f"Severity classification: {severity['label']}",
        ]

        item = EvidenceItem(
            building_id=bld_id,
            damage_class=damage_class,
            confidence=confidence,
            before_quality=before_quality if cfg.include_quality_metrics else {},
            after_quality=after_quality if cfg.include_quality_metrics else {},
            alignment_quality=alignment_quality,
            coordinates=coords,
            evidence_text="",  # filled below
            reasoning_steps=steps,
        )
        item.evidence_text = format_evidence_text(item)
        items.append(item)

    logger.info("Generated %d evidence items.", len(items))
    return items


def aggregate_scene_evidence(evidence_items: List[EvidenceItem]) -> Dict[str, Any]:
    """Produce a scene-level summary from individual building evidence."""
    total = len(evidence_items)
    by_class: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    action_required = 0
    total_confidence = 0.0

    for item in evidence_items:
        cls = max(0, min(3, item.damage_class))
        by_class[cls] = by_class.get(cls, 0) + 1
        total_confidence += item.confidence
        if classify_damage_severity(item.damage_class, item.confidence)["action_required"]:
            action_required += 1

    damaged = sum(v for k, v in by_class.items() if k > 0)
    mean_conf = total_confidence / total if total else 0.0

    # Overall severity: highest class with >=1 building
    overall_class = max((k for k, v in by_class.items() if v > 0), default=0)
    overall_severity = SEVERITY_MAP.get(overall_class, SEVERITY_MAP[0])["severity"]

    return {
        "total_buildings": total,
        "damaged_buildings": damaged,
        "undamaged_buildings": by_class.get(0, 0),
        "by_damage_class": {DAMAGE_CLASS_NAMES[k]: v for k, v in by_class.items()},
        "action_required_count": action_required,
        "mean_confidence": round(mean_conf, 4),
        "overall_severity": overall_severity,
    }


def evidence_to_dict(item: EvidenceItem) -> Dict[str, Any]:
    """Convert an EvidenceItem to a JSON-serialisable dict."""
    d = asdict(item)
    # Ensure coordinates are serialisable lists (not numpy arrays)
    if d.get("coordinates") is not None:
        try:
            d["coordinates"] = [
                [float(x), float(y)] if hasattr(x, "__float__") else [x, y]
                for x, y in d["coordinates"]
            ] if d["coordinates"] and isinstance(d["coordinates"][0], (list, tuple)) else d["coordinates"]
        except (TypeError, ValueError):
            d["coordinates"] = str(d["coordinates"])
    return d
