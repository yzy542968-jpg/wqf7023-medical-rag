from __future__ import annotations

import numpy as np

from scripts.evaluate_final_qa_confirmation import (
    B3,
    B6,
    apply_frozen_policy,
    case_grouped_exact_bootstrap,
)


def _row(
    case_id: str,
    predicted: list[int],
    gold: list[int],
    question_index: int = 0,
) -> dict:
    return {
        "case_id": case_id,
        "question_index": question_index,
        "predicted_indices": predicted,
        "gold_indices": gold,
    }


def test_apply_frozen_policy_defaults_and_selects_history() -> None:
    keys = [("CXR1", 0), ("CXR2", 0)]
    b3 = {key: _row(key[0], [], [0]) for key in keys}
    b6 = {key: _row(key[0], [0], [0]) for key in keys}
    policy = {
        "question_policy": [
            {"question_id": 3, "source": B6},
            {"question_id": 4, "source": B3},
        ]
    }
    selected, counts = apply_frozen_policy(
        {B3: b3, B6: b6},
        {keys[0]: 3, keys[1]: 4},
        policy,
    )
    assert selected[keys[0]] is b6[keys[0]]
    assert selected[keys[1]] is b3[keys[1]]
    assert counts == {B3: 1, B6: 1}


def test_case_grouped_exact_bootstrap_is_paired_by_case() -> None:
    keys = [("CXR1", 0), ("CXR2", 0)]
    left = {key: _row(key[0], [0], [0]) for key in keys}
    right = {key: _row(key[0], [], [0]) for key in keys}
    result = case_grouped_exact_bootstrap(left, right, samples=100, seed=7)
    assert result["case_count"] == 2
    assert np.isclose(result["observed_difference"], 1.0)
    assert np.isclose(result["ci95_low"], 1.0)
    assert np.isclose(result["ci95_high"], 1.0)


def test_exact_bootstrap_preserves_question_level_estimand() -> None:
    keys = [("CXR1", 0), ("CXR1", 1), ("CXR2", 0)]
    left = {
        key: _row(key[0], [0] if key[0] == "CXR1" else [], [0], key[1])
        for key in keys
    }
    right = {
        key: _row(key[0], [], [0], key[1])
        for key in keys
    }
    result = case_grouped_exact_bootstrap(left, right, samples=100, seed=7)
    assert result["question_count"] == 3
    assert np.isclose(result["observed_difference"], 2 / 3)
