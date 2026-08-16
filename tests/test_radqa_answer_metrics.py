from __future__ import annotations

import pytest

from medical_rag.evaluation.radqa_answer_metrics import (
    best_reference_score,
    evaluate_generation_records,
    is_not_answerable,
    summarize_generation_rows,
)


def prompt(qid: str, answerable: bool) -> dict:
    return {
        "qid": qid,
        "patient_id": "p1",
        "report_id": "r1",
        "question": "Question?",
        "is_answerable": answerable,
        "reference_answers": ["mild edema", "edema"] if answerable else [],
        "agent_action": "ANSWER_FROM_EVIDENCE" if answerable else "ABSTAIN_LOW_EVIDENCE",
        "retrieved_chunk_ids": ["c1"],
        "relevant_chunk_ids": ["c1"] if answerable else [],
    }


def test_multi_reference_and_unanswerable_scoring() -> None:
    assert best_reference_score("edema", ["mild edema", "edema"]) == (1.0, 1.0)
    assert best_reference_score("NOT ANSWERABLE", []) == (1.0, 1.0)
    assert is_not_answerable("Final answer: NOT ANSWERABLE")
    prompts = {"q1": prompt("q1", True), "q2": prompt("q2", False)}
    generations = [
        {"qid": "q1", "answer": "edema"},
        {"qid": "q2", "answer": "NOT ANSWERABLE"},
    ]
    rows = evaluate_generation_records(generations, prompts)
    summary = summarize_generation_rows(rows)
    assert summary["exact_match"] == 1.0
    assert summary["token_f1"] == 1.0
    assert summary["unanswerable_accuracy"] == 1.0


def test_generation_evaluator_rejects_incomplete_outputs() -> None:
    with pytest.raises(ValueError, match="missing generations"):
        evaluate_generation_records([], {"q1": prompt("q1", True)})
