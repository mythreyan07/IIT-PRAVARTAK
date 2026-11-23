"""
Evidence-based question answering about damage assessments.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .evidence_generator import (
    EvidenceItem,
    DAMAGE_CLASS_NAMES,
    aggregate_scene_evidence,
    classify_damage_severity,
)

logger = logging.getLogger(__name__)

STANDARD_QUESTIONS = [
    "How many buildings were damaged?",
    "What is the most severely damaged area?",
    "What percentage of buildings sustained major damage or worse?",
    "Are there any buildings that require immediate emergency response?",
    "What is the overall damage severity for this scene?",
]


@dataclass
class ReasoningConfig:
    use_llm: bool = True
    llm_model: str = "claude-haiku-5-5"
    api_key_env_var: str = "ANTHROPIC_API_KEY"
    fallback_to_rules: bool = True
    max_tokens: int = 512


@dataclass
class ReasoningResult:
    question: str
    answer: str
    evidence_used: List[str] = field(default_factory=list)
    reasoning_method: str = "rules"  # "llm" or "rules"
    confidence: float = 1.0


class QuestionReasoner:
    def __init__(self, config: ReasoningConfig | None = None) -> None:
        self.config = config or ReasoningConfig()
        self.llm_available = False
        self._client = None

        if self.config.use_llm:
            try:
                import anthropic  # noqa: F401
                api_key = os.environ.get(self.config.api_key_env_var, "")
                if api_key:
                    import anthropic as _anthropic
                    self._client = _anthropic.Anthropic(api_key=api_key)
                    self.llm_available = True
                    logger.info("LLM reasoning enabled (model=%s).", self.config.llm_model)
                else:
                    logger.info("No API key found — using rule-based reasoning.")
            except ImportError:
                logger.info("anthropic package not installed — using rule-based reasoning.")

    def answer_question(
        self,
        question: str,
        evidence_items: List[EvidenceItem],
        scene_summary: Dict[str, Any] | None = None,
    ) -> ReasoningResult:
        summary = scene_summary or aggregate_scene_evidence(evidence_items)
        evidence_ids = [e.building_id for e in evidence_items]

        if self.llm_available:
            try:
                context = self._build_context(evidence_items, summary)
                answer = self._llm_answer(question, context)
                return ReasoningResult(
                    question=question,
                    answer=answer,
                    evidence_used=evidence_ids,
                    reasoning_method="llm",
                    confidence=0.9,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM call failed (%s) — falling back to rules.", exc)

        answer = self._rule_based_answer(question, evidence_items, summary)
        return ReasoningResult(
            question=question,
            answer=answer,
            evidence_used=evidence_ids,
            reasoning_method="rules",
            confidence=0.8,
        )

    def _llm_answer(self, question: str, context_str: str) -> str:
        prompt = (
            "You are a disaster damage assessment expert. "
            "Given the following building damage assessment data, answer the question concisely.\n\n"
            f"Assessment Data:\n{context_str}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        message = self._client.messages.create(
            model=self.config.llm_model,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()

    def _rule_based_answer(
        self,
        question: str,
        evidence_items: List[EvidenceItem],
        scene_summary: Dict[str, Any],
    ) -> str:
        q = question.lower()
        total = scene_summary.get("total_buildings", 0)
        damaged = scene_summary.get("damaged_buildings", 0)
        by_class = scene_summary.get("by_damage_class", {})
        action_count = scene_summary.get("action_required_count", 0)
        severity = scene_summary.get("overall_severity", "none")

        if "how many" in q and "damaged" in q:
            return (
                f"{damaged} out of {total} buildings were damaged "
                f"({damaged/total*100:.1f}% of assessed buildings)."
                if total else "No buildings assessed."
            )

        if "most severely" in q or "most severe" in q:
            destroyed = [e for e in evidence_items if e.damage_class == 3]
            major = [e for e in evidence_items if e.damage_class == 2]
            if destroyed:
                worst = max(destroyed, key=lambda e: e.confidence)
                return f"Building {worst.building_id} is the most severely damaged (destroyed, confidence {worst.confidence:.2%})."
            if major:
                worst = max(major, key=lambda e: e.confidence)
                return f"Building {worst.building_id} has the highest confirmed damage (major damage, confidence {worst.confidence:.2%})."
            return "No severely damaged buildings were identified."

        if "percentage" in q and ("major" in q or "worse" in q):
            major_plus = by_class.get("major_damage", 0) + by_class.get("destroyed", 0)
            pct = major_plus / total * 100 if total else 0
            return f"{major_plus} buildings ({pct:.1f}%) sustained major damage or worse."

        if "emergency" in q or "immediate" in q:
            if action_count == 0:
                return "No buildings currently require immediate emergency response based on confidence thresholds."
            critical = [e for e in evidence_items if classify_damage_severity(e.damage_class, e.confidence)["action_required"]]
            ids = ", ".join(e.building_id for e in critical[:5])
            suffix = f" (and {action_count - 5} more)" if action_count > 5 else ""
            return f"Yes — {action_count} building(s) require immediate response: {ids}{suffix}."

        if "overall" in q or "severity" in q:
            label_map = {"none": "No significant damage", "low": "Minor damage", "high": "Major damage", "total": "Catastrophic damage"}
            label = label_map.get(severity, "Unknown")
            return f"Overall scene severity: {label}. {damaged}/{total} buildings damaged."

        # Generic fallback
        return self.generate_damage_summary(evidence_items)

    def generate_damage_summary(self, evidence_items: List[EvidenceItem]) -> str:
        summary = aggregate_scene_evidence(evidence_items)
        total = summary["total_buildings"]
        damaged = summary["damaged_buildings"]
        by_class = summary["by_damage_class"]
        severity = summary["overall_severity"]

        parts = [f"Scene summary: {total} buildings assessed, {damaged} damaged."]
        for cls_name, count in by_class.items():
            if count:
                parts.append(f"  {cls_name.replace('_', ' ').title()}: {count}")
        parts.append(f"Overall severity: {severity}.")
        return " ".join(parts)

    def answer_standard_questions(
        self, evidence_items: List[EvidenceItem]
    ) -> List[Dict[str, str]]:
        summary = aggregate_scene_evidence(evidence_items)
        return [
            {
                "question": q,
                "answer": self.answer_question(q, evidence_items, summary).answer,
            }
            for q in STANDARD_QUESTIONS
        ]

    def _build_context(
        self, evidence_items: List[EvidenceItem], summary: Dict[str, Any]
    ) -> str:
        lines = [
            f"Total buildings: {summary.get('total_buildings', 0)}",
            f"Damaged: {summary.get('damaged_buildings', 0)}",
            f"Overall severity: {summary.get('overall_severity', 'unknown')}",
            "",
            "Per-building details:",
        ]
        for item in evidence_items[:20]:  # cap context size
            cls_name = DAMAGE_CLASS_NAMES.get(item.damage_class, "unknown")
            lines.append(
                f"  Building {item.building_id}: {cls_name}, confidence {item.confidence:.2f}"
            )
        return "\n".join(lines)
