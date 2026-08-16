from __future__ import annotations

from medical_rag.evaluation.radqa_agent import (
    answerability_metrics,
    build_agent_prompt,
    select_answerability_threshold,
)


def test_answerability_threshold_is_selected_only_from_given_rows() -> None:
    rows = [
        {"is_answerable": True, "top1_score": 2.0},
        {"is_answerable": True, "top1_score": 1.5},
        {"is_answerable": False, "top1_score": 0.2},
        {"is_answerable": False, "top1_score": 0.1},
    ]
    selection = select_answerability_threshold(rows)
    selected = selection["selected"]
    assert selected["macro_f1"] == 1.0
    assert answerability_metrics(rows, selected["threshold"])["false_answer_rate"] == 0.0


def test_agent_prompt_exposes_scope_evidence_and_abstention_action() -> None:
    question = {
        "qid": "q1",
        "patient_id": "p1",
        "report_id": "r1",
        "question": "Is edema present?",
        "is_answerable": False,
        "reference_answers": [],
        "relevant_chunk_ids": [],
    }
    retrieval = {
        "system": "report_scoped_bm25",
        "top1_score": 0.0,
        "retrieved_chunk_ids": ["c1"],
        "retrieved_report_ids": ["r1"],
        "retrieved_sections": ["findings"],
        "retrieved_texts": ["The lungs are clear."],
    }
    prompt = build_agent_prompt(question, retrieval, threshold=0.5)
    assert prompt["agent_action"] == "ABSTAIN_LOW_EVIDENCE"
    assert prompt["reference_answer"] == "NOT ANSWERABLE"
    assert "Report scope: r1" in prompt["prompt"]
    assert "return NOT ANSWERABLE" in prompt["prompt"]
