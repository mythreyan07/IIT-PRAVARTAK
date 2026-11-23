"""Tests for question_reasoning.py"""
import pytest

from backend.reasoning.evidence_generator import generate_evidence
from backend.reasoning.question_reasoning import (
    QuestionReasoner,
    ReasoningConfig,
    ReasoningResult,
    STANDARD_QUESTIONS,
)


def _make_evidence(n=5):
    result = {
        "buildings": [
            {
                "building_id": f"BLD_{i:03d}",
                "damage_class": i % 4,
                "proba": [0.05, 0.05, 0.05, 0.85] if i % 4 == 3 else
                         [0.85, 0.05, 0.05, 0.05],
                "bbox": [i*10, i*10, i*10+50, i*10+50],
            }
            for i in range(n)
        ],
        "alignment_quality": 0.9,
        "before_quality": {"overall_quality": "Good"},
        "after_quality": {"overall_quality": "Good"},
    }
    return generate_evidence(result)


class TestGenerateDamageSummary:
    def test_returns_non_empty_string(self):
        reasoner = QuestionReasoner(ReasoningConfig(use_llm=False))
        summary = reasoner.generate_damage_summary(_make_evidence(5))
        assert isinstance(summary, str) and len(summary) > 0

    def test_contains_counts(self):
        reasoner = QuestionReasoner(ReasoningConfig(use_llm=False))
        summary = reasoner.generate_damage_summary(_make_evidence(5))
        assert "5" in summary  # total buildings


class TestAnswerStandardQuestions:
    def test_returns_five_questions(self):
        reasoner = QuestionReasoner(ReasoningConfig(use_llm=False))
        answers = reasoner.answer_standard_questions(_make_evidence(4))
        assert len(answers) == 5

    def test_each_has_question_and_answer(self):
        reasoner = QuestionReasoner(ReasoningConfig(use_llm=False))
        for qa in reasoner.answer_standard_questions(_make_evidence(4)):
            assert "question" in qa and "answer" in qa
            assert len(qa["answer"]) > 0

    def test_questions_match_standard(self):
        reasoner = QuestionReasoner(ReasoningConfig(use_llm=False))
        qa_list = reasoner.answer_standard_questions(_make_evidence(3))
        returned_questions = {qa["question"] for qa in qa_list}
        for q in STANDARD_QUESTIONS:
            assert q in returned_questions


class TestRuleBasedAnswers:
    def test_no_llm_still_answers(self):
        reasoner = QuestionReasoner(ReasoningConfig(use_llm=False))
        evidence = _make_evidence(3)
        result = reasoner.answer_question(STANDARD_QUESTIONS[0], evidence)
        assert isinstance(result, ReasoningResult)
        assert result.reasoning_method == "rules"
        assert len(result.answer) > 0

    def test_all_five_questions_answered(self):
        reasoner = QuestionReasoner(ReasoningConfig(use_llm=False))
        evidence = _make_evidence(6)
        for q in STANDARD_QUESTIONS:
            r = reasoner.answer_question(q, evidence)
            assert isinstance(r.answer, str) and len(r.answer) > 0


class TestReasoningResult:
    def test_has_required_fields(self):
        reasoner = QuestionReasoner(ReasoningConfig(use_llm=False))
        result = reasoner.answer_question("How many buildings were damaged?", _make_evidence(2))
        assert hasattr(result, "question")
        assert hasattr(result, "answer")
        assert hasattr(result, "evidence_used")
        assert hasattr(result, "reasoning_method")
        assert hasattr(result, "confidence")

    def test_evidence_used_is_list(self):
        reasoner = QuestionReasoner(ReasoningConfig(use_llm=False))
        result = reasoner.answer_question("test", _make_evidence(3))
        assert isinstance(result.evidence_used, list)


class TestApiKeyNotLogged:
    def test_api_key_not_in_logs(self, caplog):
        import logging
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with caplog.at_level(logging.DEBUG, logger="backend.reasoning.question_reasoning"):
            QuestionReasoner(ReasoningConfig(use_llm=True))
        # Ensure no key-looking strings in logs
        for record in caplog.records:
            assert "sk-ant" not in record.message
            assert "ANTHROPIC_API_KEY" not in record.message or "No API key" in record.message
