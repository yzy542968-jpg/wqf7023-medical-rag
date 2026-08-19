from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = (
    ROOT
    / "experiments"
    / "post_submission_v6"
    / "confirmation_retrieval_summary.json"
)


def test_frozen_confirmation_retrieval_result_is_complete() -> None:
    result = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert result["status"] == "formal_confirmation_outcomes_frozen"
    assert result["protocol_commit"] == "eee7405"
    assert result["cohort_commit"] == "43fe1a0"
    assert result["candidate_case_count"] == 240
    assert result["target_case_count"] == 120
    assert result["question_count"] == 360

    metrics = result["metrics"]
    assert set(metrics) == {
        "bm25",
        "qwen3_embedding",
        "biovilt_max_chunk_reranker",
        "medsiglip_max_chunk_reranker",
    }
    assert metrics["medsiglip_max_chunk_reranker"]["mrr"] > metrics["bm25"]["mrr"]

    control = result["random_image_control"]
    assert control["count"] == 100
    assert control["fixed_point_count"] == 0
    assert control["unique_assignment_count"] == 100
    assert control["mrr_exceedance_count"] == 0
    assert control["plus_one_monte_carlo_p_mrr"] == 1 / 101

    outputs = result["outputs"]
    assert outputs["row_count"] == 4 * result["question_count"]
    assert len(outputs["rows_sha256"]) == 64
