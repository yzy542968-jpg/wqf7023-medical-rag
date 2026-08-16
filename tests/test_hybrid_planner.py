from __future__ import annotations

import pytest

from medical_rag.agentic.hybrid_planner import select_hybrid_plan


def test_hybrid_keeps_known_lexical_route() -> None:
    plan = select_hybrid_plan(
        "Give the final diagnostic assessment.", semantic_intent="findings"
    )
    assert plan.selected_intent == "impression"
    assert plan.planner_source == "lexical_intent"


def test_hybrid_keeps_known_report_fact_frame() -> None:
    plan = select_hybrid_plan(
        "What does this report state about edema?", semantic_intent="impression"
    )
    assert plan.selected_intent == "unknown"
    assert plan.planner_source == "lexical_report_fact_frame"


def test_hybrid_uses_semantic_fallback_for_reserved_wording() -> None:
    plan = select_hybrid_plan(
        "How did the reader synthesize the study?", semantic_intent="impression"
    )
    assert plan.selected_intent == "impression"
    assert plan.planner_source == "semantic_fallback"


def test_hybrid_rejects_invalid_semantic_intent() -> None:
    with pytest.raises(ValueError, match="Unsupported semantic intent"):
        select_hybrid_plan("Question", semantic_intent="laboratory")
