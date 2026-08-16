from __future__ import annotations

import math
from typing import Iterable


def _validate(probabilities: list[float], labels: list[bool]) -> None:
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    if not probabilities:
        raise ValueError("at least one prediction is required")
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError("probabilities must be in [0, 1]")


def calibration_metrics(
    probabilities: Iterable[float], labels: Iterable[bool], bins: int = 10
) -> dict:
    probs = [float(value) for value in probabilities]
    truth = [bool(value) for value in labels]
    _validate(probs, truth)
    if bins < 1:
        raise ValueError("bins must be positive")

    reliability = []
    expected_calibration_error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, probability in enumerate(probs)
            if probability >= lower
            and (probability < upper or (index == bins - 1 and probability <= upper))
        ]
        if not members:
            reliability.append(
                {"lower": lower, "upper": upper, "count": 0, "confidence": None, "accuracy": None}
            )
            continue
        confidence = sum(probs[position] for position in members) / len(members)
        accuracy = sum(truth[position] for position in members) / len(members)
        expected_calibration_error += (
            len(members) / len(probs) * abs(confidence - accuracy)
        )
        reliability.append(
            {
                "lower": lower,
                "upper": upper,
                "count": len(members),
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    brier = sum((probability - float(label)) ** 2 for probability, label in zip(probs, truth)) / len(probs)
    return {
        "n": len(probs),
        "bins": bins,
        "ece": expected_calibration_error,
        "brier_score": brier,
        "reliability": reliability,
    }


def risk_coverage_curve(
    confidences: Iterable[float], correct: Iterable[bool]
) -> dict:
    scores = [float(value) for value in confidences]
    outcomes = [bool(value) for value in correct]
    _validate(scores, outcomes)
    ranked = sorted(
        zip(scores, outcomes, range(len(scores))),
        key=lambda row: (-row[0], row[2]),
    )
    cumulative_errors = 0
    points = []
    for rank, (confidence, is_correct, _) in enumerate(ranked, start=1):
        cumulative_errors += int(not is_correct)
        points.append(
            {
                "coverage": rank / len(ranked),
                "risk": cumulative_errors / rank,
                "threshold": confidence,
                "accepted": rank,
            }
        )
    aurc = sum(point["risk"] for point in points) / len(points)
    return {"n": len(scores), "aurc": aurc, "points": points}


def _logit(probability: float, epsilon: float = 1e-6) -> float:
    clipped = min(1.0 - epsilon, max(epsilon, probability))
    return math.log(clipped / (1.0 - clipped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        negative = math.exp(-value)
        return 1.0 / (1.0 + negative)
    positive = math.exp(value)
    return positive / (1.0 + positive)


def fit_platt_scaler(
    probabilities: Iterable[float],
    labels: Iterable[bool],
    *,
    l2: float = 1e-2,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> dict[str, float | int]:
    probs = [float(value) for value in probabilities]
    truth = [bool(value) for value in labels]
    _validate(probs, truth)
    raw_features = [_logit(value) for value in probs]
    feature_mean = sum(raw_features) / len(raw_features)
    feature_variance = sum(
        (value - feature_mean) ** 2 for value in raw_features
    ) / len(raw_features)
    feature_scale = max(1e-6, math.sqrt(feature_variance))
    features = [(value - feature_mean) / feature_scale for value in raw_features]
    positive_rate = min(
        1.0 - 1e-6, max(1e-6, sum(truth) / len(truth))
    )
    slope = 0.0
    intercept = _logit(positive_rate)
    completed_iterations = 0

    def objective(candidate_slope: float, candidate_intercept: float) -> float:
        loss = 0.5 * l2 * candidate_slope**2
        for feature, label in zip(features, truth):
            logit_value = candidate_slope * feature + candidate_intercept
            loss += (
                max(logit_value, 0.0)
                - float(label) * logit_value
                + math.log1p(math.exp(-abs(logit_value)))
            )
        return loss

    for iteration in range(1, max_iterations + 1):
        predictions = [_sigmoid(slope * value + intercept) for value in features]
        weights = [max(1e-9, value * (1.0 - value)) for value in predictions]
        gradient_slope = sum(
            (prediction - float(label)) * feature
            for prediction, label, feature in zip(predictions, truth, features)
        ) + l2 * slope
        gradient_intercept = sum(
            prediction - float(label) for prediction, label in zip(predictions, truth)
        )
        hessian_slope = sum(
            weight * feature * feature for weight, feature in zip(weights, features)
        ) + l2
        hessian_cross = sum(
            weight * feature for weight, feature in zip(weights, features)
        )
        hessian_intercept = sum(weights)
        determinant = hessian_slope * hessian_intercept - hessian_cross**2
        if abs(determinant) < 1e-12:
            break
        slope_step = (
            hessian_intercept * gradient_slope
            - hessian_cross * gradient_intercept
        ) / determinant
        intercept_step = (
            hessian_slope * gradient_intercept
            - hessian_cross * gradient_slope
        ) / determinant
        current_objective = objective(slope, intercept)
        step_scale = 1.0
        while step_scale >= 1e-6:
            candidate_slope = slope - step_scale * slope_step
            candidate_intercept = intercept - step_scale * intercept_step
            if objective(candidate_slope, candidate_intercept) <= current_objective:
                slope = candidate_slope
                intercept = candidate_intercept
                break
            step_scale *= 0.5
        if step_scale < 1e-6:
            break
        completed_iterations = iteration
        if step_scale * max(abs(slope_step), abs(intercept_step)) < tolerance:
            break
    return {
        "method": "platt_logit",
        "slope": slope,
        "intercept": intercept,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "l2": l2,
        "iterations": completed_iterations,
    }


def apply_platt_scaler(
    probabilities: Iterable[float], model: dict[str, float | int]
) -> list[float]:
    slope = float(model["slope"])
    intercept = float(model["intercept"])
    feature_mean = float(model.get("feature_mean", 0.0))
    feature_scale = float(model.get("feature_scale", 1.0))
    values = [float(value) for value in probabilities]
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("probabilities must be in [0, 1]")
    return [
        _sigmoid(
            slope * ((_logit(value) - feature_mean) / feature_scale) + intercept
        )
        for value in values
    ]
