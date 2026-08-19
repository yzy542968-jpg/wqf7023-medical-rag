from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "experiments"
    / "post_submission_v6"
    / "confirmation_qa_factorial_verified_summary.json"
)


def test_frozen_confirmation_verified_result_is_complete() -> None:
    result = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert result["status"] == "formal_confirmation_verified_qa_outcomes_frozen"
    assert result["protocol_commit"] == "eee7405"
    assert result["cohort_commit"] == "43fe1a0"
    assert result["raw_qa_outcome_commit"] == "5aa8a6b"
    assert result["input_integrity"] == result["output_integrity"]
    assert result["output_integrity"]["row_count"] == 1440
    assert result["verifier"]["changed_from_v5"] is False
    assert result["verifier"]["human_validated_clinical_correctness"] is False

    metrics = result["metrics"]
    assert all(cell["row_count"] == 360 for cell in metrics.values())
    assert metrics["medsiglip_qwen2_5"]["verified_token_f1"] > metrics["bm25_qwen2_5"]["verified_token_f1"]
    assert metrics["medsiglip_medgemma_1_5"]["verified_token_f1"] > metrics["bm25_medgemma_1_5"]["verified_token_f1"]

    outputs = result["outputs"]
    assert outputs["verified_row_count"] == 1440
    assert len(outputs["verified_rows_sha256"]) == 64
