from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "post_submission_v6"
    / "confirmation_statistical_analysis.json"
)


def test_frozen_confirmation_statistics_match_protocol_gates() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))

    assert result["status"] == "formal_confirmation_statistics_frozen"
    assert result["protocol_commit"] == "eee7405"
    assert result["retrieval_outcome_commit"] == "c6442c9"
    assert result["verified_qa_outcome_commit"] == "3ae127f"
    assert result["method"]["unit"] == "case_id"
    assert result["method"]["bootstrap_resamples"] == 5000
    assert result["method"]["bootstrap_seed"] == 7026

    retrieval = result["primary_retrieval"]
    assert retrieval["criterion_passed"] is True
    assert retrieval["ci_lower"] > 0
    assert result["alignment_specificity"]["criterion_passed"] is True
    assert result["alignment_specificity"]["plus_one_monte_carlo_p_mrr"] == 1 / 101

    assert set(result["primary_qa"]) == {"qwen2_5", "medgemma_1_5"}
    for generator in result["primary_qa"].values():
        assert generator["criterion_passed"] is True
        assert generator["point_difference"] > 0
        assert generator["ci_lower"] > 0

    integrity = result["integrity"]
    assert integrity["target_case_count"] == 120
    assert integrity["normal_target_count"] == 86
    assert integrity["abnormal_target_count"] == 34
    assert integrity["reconstructed_bm25_mrr_matches_frozen"] is True
    assert integrity["reconstructed_medsiglip_mrr_matches_frozen"] is True
