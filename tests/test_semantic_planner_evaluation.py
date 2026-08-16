from __future__ import annotations

from scripts.evaluate_v22_semantic_planner import (
    expected_planner_label,
    parse_planner_label,
)
from scripts.evaluate_case_scoped_hard_v21 import _system_metrics


def test_planner_label_parser_is_constrained() -> None:
    assert parse_planner_label("IMPRESSION") == "IMPRESSION"
    assert parse_planner_label("REPORT_FACT\n") == "REPORT_FACT"
    assert parse_planner_label("The answer is FINDINGS.") == "FINDINGS"
    assert parse_planner_label("something else") == "PARSE_FAILURE"


def test_expected_label_uses_question_family_not_gold_section() -> None:
    assert expected_planner_label("CXR1_v21_fact_probe") == "REPORT_FACT"
    assert expected_planner_label("CXR1_v21_near_domain_negative") == "REPORT_FACT"
    assert expected_planner_label("CXR1_v21_unanswerable_a") == "OUTSIDE_REPORT"


def test_agent_metrics_are_capability_based_not_system_name() -> None:
    row = {
        "system": "future_agent_name",
        "is_answerable": True,
        "answer_probability": 0.9,
        "relevant_chunk_ids": ["c1"],
        "retrieved_chunk_ids": ["c1"],
        "retrieved_texts": ["supported answer"],
        "reference_answer": "supported answer",
        "final_intent": "findings",
        "expected_intent": "findings",
        "retrieval_calls": 1,
        "retrieved_chunk_count": 1,
        "retried": False,
    }
    assert _system_metrics([row], threshold=0.5)["route_accuracy_answerable"] == 1.0
