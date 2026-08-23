from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


FEATURE_ORDER = (
    "top1_score",
    "top1_top2_margin",
    "component_agreement",
    "ensemble_variance",
    "evidence_score",
    "evidence_redundancy",
    "view_count",
    "question_findings",
    "question_impression",
    "question_acute",
)


def feature_matrix(rows: Sequence[Mapping[str, float]]) -> np.ndarray:
    return np.asarray(
        [[float(row.get(name, 0.0)) for name in FEATURE_ORDER] for row in rows],
        dtype=np.float64,
    )


@dataclass
class RetrievalCalibrator:
    seed: int = 7046
    model: LogisticRegression | None = None

    def fit(self, rows: Sequence[Mapping[str, float]], labels: Sequence[int]) -> "RetrievalCalibrator":
        values = np.asarray(labels, dtype=np.int64)
        if len(rows) != len(values) or len(rows) < 2:
            raise ValueError("calibration rows and labels must align and contain at least two rows")
        if len(np.unique(values)) != 2:
            raise ValueError("calibration labels must contain both classes")
        self.model = LogisticRegression(
            solver="liblinear",
            random_state=self.seed,
            max_iter=1000,
        ).fit(feature_matrix(rows), values)
        return self

    def predict_proba(self, rows: Sequence[Mapping[str, float]]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("calibrator is not fitted")
        return self.model.predict_proba(feature_matrix(rows))[:, 1]


def expected_calibration_error(labels: Sequence[int], probabilities: Sequence[float], bins: int = 10) -> float:
    labels_array = np.asarray(labels, dtype=np.float64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    if labels_array.shape != probability_array.shape:
        raise ValueError("labels and probabilities must have equal shape")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probability_array >= lower) & (
            probability_array <= upper if index == bins - 1 else probability_array < upper
        )
        if not np.any(mask):
            continue
        result += float(mask.mean()) * abs(float(labels_array[mask].mean()) - float(probability_array[mask].mean()))
    return result


def threshold_for_coverage(probabilities: Sequence[float], target_coverage: float) -> float:
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    values = np.sort(np.asarray(probabilities, dtype=np.float64))[::-1]
    if values.size == 0:
        raise ValueError("probabilities cannot be empty")
    count = max(1, int(np.ceil(target_coverage * len(values))))
    return float(values[count - 1])


def risk_coverage_curve(labels: Sequence[int], probabilities: Sequence[float]) -> list[dict[str, float]]:
    labels_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    order = np.lexsort((np.arange(len(probability_array)), -probability_array))
    rows = []
    correct = 0
    for rank, index in enumerate(order, start=1):
        correct += int(labels_array[index])
        coverage = rank / len(order)
        accuracy = correct / rank
        rows.append({"coverage": coverage, "selective_accuracy": accuracy, "risk": 1.0 - accuracy, "threshold": float(probability_array[index])})
    return rows


def calibration_metrics(labels: Sequence[int], probabilities: Sequence[float]) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    return {
        "brier": float(brier_score_loss(labels_array, probability_array)),
        "ece_10": expected_calibration_error(labels_array, probability_array, bins=10),
        "auroc": float(roc_auc_score(labels_array, probability_array)),
    }


def calibrator_payload(calibrator: RetrievalCalibrator) -> dict[str, object]:
    if calibrator.model is None:
        raise RuntimeError("calibrator is not fitted")
    return {
        "feature_order": list(FEATURE_ORDER),
        "classes": [int(value) for value in calibrator.model.classes_],
        "coef": calibrator.model.coef_.tolist(),
        "intercept": calibrator.model.intercept_.tolist(),
        "seed": calibrator.seed,
    }


def predict_from_payload(
    rows: Sequence[Mapping[str, float]],
    payload: Mapping[str, object],
) -> np.ndarray:
    if tuple(payload["feature_order"]) != FEATURE_ORDER:
        raise ValueError("calibration feature order does not match frozen runtime")
    coefficients = np.asarray(payload["coef"], dtype=np.float64)
    intercept = np.asarray(payload["intercept"], dtype=np.float64)
    logits = feature_matrix(rows) @ coefficients[0] + intercept[0]
    return 1.0 / (1.0 + np.exp(-logits))


__all__ = [
    "FEATURE_ORDER",
    "RetrievalCalibrator",
    "calibration_metrics",
    "calibrator_payload",
    "expected_calibration_error",
    "feature_matrix",
    "predict_from_payload",
    "risk_coverage_curve",
    "threshold_for_coverage",
]
