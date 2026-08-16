from __future__ import annotations

from scripts.evaluate_case_scoped_generation_v2 import grouped_bootstrap_ci, summarize_rows


def test_grouped_summary_counts_cases_and_question_types() -> None:
    rows = [
        {
            "case_id": "c1",
            "question_type": "a",
            "draft_token_f1": 0.2,
            "final_token_f1": 0.3,
            "support_rate": 1.0,
            "agent_abstained": False,
            "revised": True,
            "retrieval_recall": 1.0,
        },
        {
            "case_id": "c1",
            "question_type": "b",
            "draft_token_f1": 0.4,
            "final_token_f1": 0.5,
            "support_rate": 0.5,
            "agent_abstained": False,
            "revised": False,
            "retrieval_recall": 0.8,
        },
        {
            "case_id": "c2",
            "question_type": "a",
            "draft_token_f1": 0.6,
            "final_token_f1": 0.7,
            "support_rate": 1.0,
            "agent_abstained": False,
            "revised": False,
            "retrieval_recall": 1.0,
        },
    ]
    summary = summarize_rows(rows, bootstrap_samples=100)
    assert summary["n"] == 3
    assert summary["case_count"] == 2
    assert summary["by_question_type"]["a"]["n"] == 2


def test_grouped_bootstrap_is_reproducible() -> None:
    rows = [
        {"case_id": "c1", "score": 0.0},
        {"case_id": "c2", "score": 1.0},
    ]
    assert grouped_bootstrap_ci(rows, "score", samples=100, seed=3) == grouped_bootstrap_ci(
        rows, "score", samples=100, seed=3
    )
