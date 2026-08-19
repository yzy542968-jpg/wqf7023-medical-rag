from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_v6_confirmation.py"
SPEC = importlib.util.spec_from_file_location("v6_confirmation_statistics", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_reciprocal_rank_uses_complete_ranking() -> None:
    ranking = ["c1", "c2", "c3", "c4"]

    assert MODULE.reciprocal_rank(ranking, "c1") == 1.0
    assert MODULE.reciprocal_rank(ranking, "c4") == 0.25
    assert MODULE.reciprocal_rank(ranking, "missing") == 0.0


def test_paired_case_bootstrap_is_deterministic_and_grouped() -> None:
    differences = {"case-a": 0.1, "case-b": 0.2, "case-c": 0.3}

    first = MODULE.paired_case_bootstrap(
        differences, resamples=500, seed=7026, confidence_level=0.95
    )
    second = MODULE.paired_case_bootstrap(
        differences, resamples=500, seed=7026, confidence_level=0.95
    )

    assert first == second
    assert first["case_count"] == 3
    assert first["point_difference"] == pytest.approx(0.2)
    assert first["ci_lower"] <= first["point_difference"] <= first["ci_upper"]


def test_differences_for_rankings_averages_three_questions_per_case() -> None:
    questions = [
        {"qid": f"q{index}", "case_id": "target"} for index in range(3)
    ]
    baseline = {f"q{index}": ["other", "target"] for index in range(3)}
    treatment = {f"q{index}": ["target", "other"] for index in range(3)}

    baseline_case, treatment_case, differences = MODULE.differences_for_rankings(
        questions, baseline, treatment
    )

    assert baseline_case["target"] == 0.5
    assert treatment_case["target"] == 1.0
    assert differences["target"] == 0.5
