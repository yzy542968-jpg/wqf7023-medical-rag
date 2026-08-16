from __future__ import annotations

import pytest

from medical_rag.evaluation.selective_prediction import (
    apply_platt_scaler,
    calibration_metrics,
    fit_platt_scaler,
    risk_coverage_curve,
)


def test_perfect_ranked_predictions_have_zero_aurc() -> None:
    result = risk_coverage_curve([0.9, 0.8, 0.7], [True, True, True])
    assert result["aurc"] == 0.0
    assert result["points"][-1]["coverage"] == 1.0


def test_calibration_reports_brier_and_reliability_bins() -> None:
    result = calibration_metrics([0.0, 1.0], [False, True], bins=2)
    assert result["brier_score"] == 0.0
    assert result["ece"] == 0.0
    assert sum(row["count"] for row in result["reliability"]) == 2


def test_probability_range_is_validated() -> None:
    with pytest.raises(ValueError):
        calibration_metrics([1.2], [True])


def test_platt_scaler_learns_monotonic_probability_mapping() -> None:
    probabilities = [0.1, 0.2, 0.8, 0.9]
    labels = [False, False, True, True]
    model = fit_platt_scaler(probabilities, labels)
    calibrated = apply_platt_scaler(probabilities, model)
    assert calibrated == sorted(calibrated)
    assert calibrated[0] < 0.5 < calibrated[-1]
