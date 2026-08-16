from __future__ import annotations

import pytest

from scripts.build_final_results_registry import assert_same, find_pair, find_system
from scripts.build_submission_manifest import resolve_human_evaluation


def test_registry_selectors_and_consistency_guard() -> None:
    systems = [{"system": "a", "value": 1}, {"system": "b", "value": 2}]
    pairs = [{"system_a": "a", "system_b": "b", "difference": 1}]
    assert find_system(systems, "b")["value"] == 2
    assert find_pair(pairs, "a", "b")["difference"] == 1
    assert_same(0.2, 0.2, "same")
    with pytest.raises(ValueError):
        assert_same(0.2, 0.3, "different")


def test_human_evaluation_can_be_transparently_not_conducted() -> None:
    human = {
        "v1": {"rows": 36, "completed_rows": 0},
        "v2": {"rows": 36, "completed_rows": 0},
    }
    policy = {
        "status": "not_conducted",
        "limitations_declared": True,
        "scores_claimed": False,
    }
    resolved, complete = resolve_human_evaluation(human, policy)
    assert resolved is True
    assert complete is False


def test_partial_human_ratings_cannot_be_declared_not_conducted() -> None:
    human = {
        "v1": {"rows": 36, "completed_rows": 1},
        "v2": {"rows": 36, "completed_rows": 0},
    }
    policy = {
        "status": "not_conducted",
        "limitations_declared": True,
        "scores_claimed": False,
    }
    resolved, complete = resolve_human_evaluation(human, policy)
    assert resolved is False
    assert complete is False
