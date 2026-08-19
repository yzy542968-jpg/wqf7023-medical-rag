from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "experiments"
    / "post_submission_v6"
    / "confirmation_qa_factorial_summary.json"
)


def test_frozen_confirmation_qa_result_is_complete() -> None:
    result = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert result["status"] == "formal_confirmation_raw_qa_outcomes_frozen"
    assert result["protocol_commit"] == "eee7405"
    assert result["cohort_commit"] == "43fe1a0"
    assert result["retrieval_outcome_commit"] == "c6442c9"

    factorial = result["factorial"]
    assert factorial["question_count"] == 360
    assert factorial["row_count"] == 1440
    assert factorial["retrieval_conditions"] == ["bm25", "medsiglip"]
    assert factorial["generator_conditions"] == ["qwen2_5", "medgemma_1_5"]

    metrics = result["metrics"]
    assert set(metrics) == {
        "bm25_qwen2_5",
        "medsiglip_qwen2_5",
        "bm25_medgemma_1_5",
        "medsiglip_medgemma_1_5",
    }
    assert all(cell["row_count"] == 360 for cell in metrics.values())
    assert all(cell["case_count"] == 120 for cell in metrics.values())
    assert metrics["medsiglip_qwen2_5"]["raw_token_f1"] > metrics["bm25_qwen2_5"]["raw_token_f1"]
    assert metrics["medsiglip_medgemma_1_5"]["raw_token_f1"] > metrics["bm25_medgemma_1_5"]["raw_token_f1"]

    outputs = result["outputs"]
    assert outputs["row_count"] == 1440
    assert len(outputs["rows_sha256"]) == 64
