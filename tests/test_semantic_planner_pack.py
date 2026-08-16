from __future__ import annotations

from scripts.build_v22_semantic_planner_pack import planner_prompt


def test_planner_prompt_has_closed_label_set_and_question() -> None:
    prompt = planner_prompt("State the overall interpretation.")
    assert "FINDINGS" in prompt
    assert "IMPRESSION" in prompt
    assert "REPORT_FACT" in prompt
    assert "OUTSIDE_REPORT" in prompt
    assert "State the overall interpretation." in prompt
