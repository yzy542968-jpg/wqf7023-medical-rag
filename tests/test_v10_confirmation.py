from __future__ import annotations

import numpy as np

from medical_rag.evaluation.v10_confirmation import (
    case_grouped_bootstrap_difference,
    deterministic_derangement,
    graded_ndcg,
    hit_at_k,
    plus_one_monte_carlo_p,
    reciprocal_rank_at_threshold,
)


def test_retrieval_metrics_reward_relevant_first() -> None:
    gains = np.asarray([1.0, 0.5, 0.0], dtype=np.float32)
    good = np.asarray([0, 1, 2])
    bad = np.asarray([2, 1, 0])
    assert graded_ndcg(gains, good) > graded_ndcg(gains, bad)
    assert reciprocal_rank_at_threshold(gains, good) == 1.0
    assert hit_at_k(gains, bad, k=1) == 0.0


def test_derangements_are_fixed_point_free_and_reproducible() -> None:
    case_ids = [f"C{index}" for index in range(5)]
    first = deterministic_derangement(case_ids, assignment_index=0, seed=7040)
    second = deterministic_derangement(case_ids, assignment_index=0, seed=7040)
    assert first == second
    assert all(source != target for source, target in first.items())


def test_bootstrap_and_monte_carlo_are_deterministic() -> None:
    rows = [
        {"case_id": "a", "system": "left", "score": 1.0},
        {"case_id": "a", "system": "right", "score": 0.0},
        {"case_id": "b", "system": "left", "score": 0.5},
        {"case_id": "b", "system": "right", "score": 0.0},
    ]
    result = case_grouped_bootstrap_difference(
        rows,
        left="left",
        right="right",
        metric="score",
        iterations=100,
        seed=1,
    )
    assert result["mean_difference"] == 0.75
    assert plus_one_monte_carlo_p(0.8, [0.1, 0.9, 0.2]) == 0.5
