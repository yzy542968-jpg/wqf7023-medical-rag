from __future__ import annotations

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
