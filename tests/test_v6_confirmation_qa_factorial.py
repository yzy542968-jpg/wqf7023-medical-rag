from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v6_confirmation_qa_factorial.py"
SPEC = importlib.util.spec_from_file_location("v6_confirmation_qa_factorial", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_confirmation_tasks_use_only_frozen_primary_retrieval_conditions() -> None:
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
        {"system": "bm25", "qid": "q1", "selected_case_id": "a"},
        {
            "system": "medsiglip_max_chunk_reranker",
            "qid": "q1",
            "selected_case_id": "b",
        },
        {"system": "qwen3_embedding", "qid": "q1", "selected_case_id": "source"},
    ]

    tasks = MODULE.build_tasks([question], cases, retrieval)

    assert [task["retrieval"] for task in tasks] == ["bm25", "medsiglip"]
    assert tasks[0]["selected_case_id"] == "a"
    assert tasks[1]["selected_case_id"] == "b"
    assert "Findings: Opacity" in tasks[0]["prompt"]
    assert "Findings: Clear" in tasks[1]["prompt"]


def test_completed_keys_rejects_duplicate_formal_rows(tmp_path: Path) -> None:
    output = tmp_path / "rows.jsonl"
    output.write_text(
        '{"system":"bm25_qwen2_5","qid":"q1"}\n'
        '{"system":"bm25_qwen2_5","qid":"q1"}\n',
        encoding="utf-8",
    )

    try:
        MODULE.completed_keys(output)
    except RuntimeError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("Duplicate formal rows were accepted.")
