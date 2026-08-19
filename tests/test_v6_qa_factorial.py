from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v6_development_qa_factorial.py"
SPEC = importlib.util.spec_from_file_location("v6_qa_factorial", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_factorial_tasks_use_same_prompt_for_retrieval_condition() -> None:
    question = {
        "qid": "q1",
        "case_id": "source",
        "question": "What is present?",
        "reference_answer": "Opacity",
    }
    cases = {
        "source": {"case_id": "source", "indication": "Cough"},
        "a": {"case_id": "a", "findings": "Opacity", "impression": "Pneumonia"},
        "b": {"case_id": "b", "findings": "Clear", "impression": "Normal"},
    }
    retrieval = [
        {"system": "indication_question_bm25", "qid": "q1", "selected_case_id": "a"},
        {"system": "medsiglip_max_chunk_reranker", "qid": "q1", "selected_case_id": "b"},
    ]

    tasks = MODULE.build_tasks([question], cases, retrieval)

    assert [task["retrieval"] for task in tasks] == ["bm25", "medsiglip"]
    assert tasks[0]["selected_case_id"] == "a"
    assert tasks[1]["selected_case_id"] == "b"
    assert "Findings: Opacity" in tasks[0]["prompt"]
    assert "Findings: Clear" in tasks[1]["prompt"]


def test_exact_match_normalizes_case_and_whitespace() -> None:
    assert MODULE.exact_match("  No acute finding ", "no ACUTE finding") == 1.0
