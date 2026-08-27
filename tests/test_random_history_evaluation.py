from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_v10_random_history_control import (  # noqa: E402
    grouped_linear_values,
    invert_bootstrap_differences,
    paired_linear_bootstrap,
)


def test_grouped_linear_values_average_assignments_then_questions() -> None:
    rows = []
    for assignment in range(5):
        rows.extend(
            [
                {
                    "case_id": "A",
                    "question_type": "findings",
                    "assignment": assignment,
                    "score": float(assignment),
                },
                {
                    "case_id": "A",
                    "question_type": "impression",
                    "assignment": assignment,
                    "score": float(assignment + 2),
                },
            ]
        )
    values = grouped_linear_values(rows, "score", assignment_mean=True)
    assert values == {"A": pytest.approx(3.0)}


def test_paired_linear_bootstrap_preserves_case_pairing() -> None:
    result = paired_linear_bootstrap(
        {"A": 1.0, "B": 2.0},
        {"A": 0.0, "B": 1.0},
        iterations=100,
        seed=3,
    )
    assert result["mean_difference"] == pytest.approx(1.0)
    assert result["ci_95_low"] == pytest.approx(1.0)
    assert result["ci_95_high"] == pytest.approx(1.0)


def test_invert_bootstrap_differences_swaps_interval_signs() -> None:
    result = invert_bootstrap_differences(
        {"metric": {"mean_difference": 0.2, "ci_95_low": 0.1, "ci_95_high": 0.3}}
    )
    assert result == {
        "metric": {
            "mean_difference": pytest.approx(-0.2),
            "ci_95_low": pytest.approx(-0.3),
            "ci_95_high": pytest.approx(-0.1),
        }
    }
