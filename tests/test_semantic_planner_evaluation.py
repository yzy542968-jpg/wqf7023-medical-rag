from __future__ import annotations

from scripts.evaluate_v22_semantic_planner import (
    expected_planner_label,
    parse_planner_label,
)


def test_planner_label_parser_is_constrained() -> None:
    assert parse_planner_label("IMPRESSION") == "IMPRESSION"
    assert parse_planner_label("REPORT_FACT\n") == "REPORT_FACT"
    assert parse_planner_label("The answer is FINDINGS.") == "FINDINGS"
    assert parse_planner_label("something else") == "PARSE_FAILURE"


def test_expected_label_uses_question_family_not_gold_section() -> None:
    assert expected_planner_label("CXR1_v21_fact_probe") == "REPORT_FACT"
    assert expected_planner_label("CXR1_v21_near_domain_negative") == "REPORT_FACT"
    assert expected_planner_label("CXR1_v21_unanswerable_a") == "OUTSIDE_REPORT"
