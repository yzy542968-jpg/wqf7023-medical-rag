from __future__ import annotations

from scripts.calibrate_case_scoped_verifier_v2 import evaluate_config


def test_calibration_evaluation_uses_reference_and_raw_sentence_scores() -> None:
    rows = [
        {
            "draft_answer": "No edema.",
            "reference_answer": "No edema.",
            "sentence_checks": [
                {
                    "sentence": "No edema.",
                    "lexical_score": 1.0,
                    "entailment_probability": 0.99,
                    "contradiction_probability": 0.0,
                    "negation_consistent": True,
                }
            ],
        }
    ]
    result = evaluate_config(rows, 0.2, 0.6, 0.75, 0.5)
    assert result["draft_token_f1"] == 1.0
    assert result["final_token_f1"] == 1.0
    assert result["abstention_rate"] == 0.0
