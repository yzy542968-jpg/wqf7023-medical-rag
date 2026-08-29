from __future__ import annotations

from scripts.evaluate_v16_qlora_metrics import validate_matrix


def test_routed_matrix_accepts_mixed_source_model_arms() -> None:
    rows = []
    for question_type in ("findings", "impression"):
        for condition in ("no_history", "retrieved_history", "random_history"):
            rows.append({
                "case_id": "CXR1",
                "question_type": question_type,
                "condition": condition,
                "model_arm": "qlora" if question_type == "impression" else "base",
            })
    validate_matrix(rows, "impression_gate")
