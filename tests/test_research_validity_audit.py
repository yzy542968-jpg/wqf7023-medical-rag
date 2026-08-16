from __future__ import annotations

from scripts.run_research_validity_audit import ambiguity_summary


def test_ambiguity_summary_detects_same_query_with_different_targets() -> None:
    questions = [
        {"qid": "q1", "case_id": "c1", "question": "Same?", "question_type": "x"},
        {"qid": "q2", "case_id": "c2", "question": "Same?", "question_type": "x"},
        {"qid": "q3", "case_id": "c3", "question": "Unique?", "question_type": "x"},
    ]
    summary = ambiguity_summary(questions, {"q1", "q2", "q3"})

    assert summary["unique_query_count"] == 2
    assert summary["ambiguous_question_rows"] == 2
    assert summary["ambiguous_question_rate"] == 2 / 3
