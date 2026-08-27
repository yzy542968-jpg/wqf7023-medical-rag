from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_v13_concept_qa_pilot import (  # noqa: E402
    grouped_values,
    paired_bootstrap,
)


def test_grouped_values_average_questions_within_case() -> None:
    rows = [
        {"case_id": "A", "metric": 1.0},
        {"case_id": "A", "metric": 3.0},
        {"case_id": "B", "metric": 2.0},
        {"case_id": "B", "metric": 4.0},
    ]
    assert grouped_values(rows, "metric") == {
        "A": pytest.approx(2.0),
        "B": pytest.approx(3.0),
    }


def test_paired_bootstrap_uses_case_differences() -> None:
    result = paired_bootstrap(
        {"A": 2.0, "B": 3.0},
        {"A": 1.0, "B": 2.0},
        iterations=100,
        seed=4,
    )
    assert result["mean_difference"] == pytest.approx(1.0)
    assert result["ci_95_low"] == pytest.approx(1.0)
    assert result["ci_95_high"] == pytest.approx(1.0)
