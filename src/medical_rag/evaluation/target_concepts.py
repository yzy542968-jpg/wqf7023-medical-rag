from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from medical_rag.evaluation.chexbert_pathology import CHEXBERT_FIVE_INDICES, CHEXBERT_LABELS


def case_id_fingerprint(case_ids: Sequence[object]) -> str:
    canonical = sorted({str(case_id).strip() for case_id in case_ids})
    if any(not value for value in canonical):
        raise ValueError("case IDs must be non-empty")
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


def supported_label_indices(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int8)
    if values.ndim != 2 or values.shape[1] != len(CHEXBERT_LABELS):
        raise ValueError("labels must have shape (n, 14)")
    support = values.sum(axis=0)
    return np.flatnonzero((support > 0) & (support < values.shape[0]))


def macro_auprc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have equal shape")
    indices = supported_label_indices(labels)
    if not len(indices):
        return 0.0
    return float(
        np.mean(
            [average_precision_score(labels[:, index], probabilities[:, index]) for index in indices]
        )
    )


def select_f1_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    grid: Sequence[float] = tuple(np.arange(0.05, 1.0, 0.05)),
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have equal shape")
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float64)
    records = []
    for index, label_name in enumerate(CHEXBERT_LABELS):
        target = labels[:, index]
        supported = 0 < int(target.sum()) < len(target)
        best_f1 = -1.0
        if supported:
            for threshold in grid:
                score = float(
                    f1_score(target, probabilities[:, index] >= threshold, zero_division=0)
                )
                if score > best_f1 or (np.isclose(score, best_f1) and threshold > thresholds[index]):
                    best_f1 = score
                    thresholds[index] = float(threshold)
        records.append(
            {
                "condition": label_name,
                "supported": supported,
                "positive_count": int(target.sum()),
                "negative_count": int(len(target) - target.sum()),
                "threshold": float(thresholds[index]),
                "calibration_f1": float(max(best_f1, 0.0)),
            }
        )
    return thresholds, records


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    truth = np.asarray(labels, dtype=np.float64).ravel()
    scores = np.asarray(probabilities, dtype=np.float64).ravel()
    if truth.shape != scores.shape:
        raise ValueError("labels and probabilities must have equal shape")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(scores)
    result = 0.0
    for index in range(bins):
        selected = (scores >= edges[index]) & (
            scores <= edges[index + 1] if index == bins - 1 else scores < edges[index + 1]
        )
        if selected.any():
            result += selected.mean() * abs(scores[selected].mean() - truth[selected].mean())
    return float(result if total else 0.0)


def risk_coverage_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> list[dict[str, float]]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= np.asarray(thresholds, dtype=np.float64)
    confidence = np.mean(np.abs(probabilities - 0.5) * 2.0, axis=1)
    case_error = np.mean(predictions != labels, axis=1)
    order = np.argsort(-confidence, kind="stable")
    output = []
    for coverage in np.arange(0.1, 1.01, 0.1):
        count = max(1, int(np.ceil(len(order) * coverage)))
        selected = order[:count]
        output.append(
            {
                "coverage": float(min(coverage, 1.0)),
                "case_count": int(count),
                "risk_hamming_loss": float(case_error[selected].mean()),
                "minimum_confidence": float(confidence[selected].min()),
            }
        )
    return output


def multilabel_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if labels.shape != probabilities.shape or thresholds.shape != (labels.shape[1],):
        raise ValueError("Invalid multilabel metric shapes")
    predictions = (probabilities >= thresholds).astype(np.int8)
    indices = supported_label_indices(labels)
    supported_set = set(indices.tolist())
    supported_labels = labels[:, indices] if len(indices) else np.empty((len(labels), 0))
    supported_probabilities = (
        probabilities[:, indices] if len(indices) else np.empty((len(labels), 0))
    )
    per_label = {}
    for index, name in enumerate(CHEXBERT_LABELS):
        supported = index in supported_set
        per_label[name] = {
            "supported": supported,
            "positive_count": int(labels[:, index].sum()),
            "auroc": (
                float(roc_auc_score(labels[:, index], probabilities[:, index]))
                if supported
                else None
            ),
            "auprc": (
                float(average_precision_score(labels[:, index], probabilities[:, index]))
                if supported
                else None
            ),
            "f1": float(f1_score(labels[:, index], predictions[:, index], zero_division=0)),
        }
    return {
        "case_count": int(labels.shape[0]),
        "supported_label_count": int(len(indices)),
        "macro_auroc": float(
            np.mean(
                [roc_auc_score(labels[:, index], probabilities[:, index]) for index in indices]
            )
        )
        if len(indices)
        else 0.0,
        "macro_auprc": macro_auprc(labels, probabilities),
        "micro_auroc": (
            float(roc_auc_score(supported_labels.ravel(), supported_probabilities.ravel()))
            if len(indices) and len(np.unique(supported_labels)) == 2
            else None
        ),
        "micro_auprc": (
            float(average_precision_score(supported_labels.ravel(), supported_probabilities.ravel()))
            if len(indices) and supported_labels.sum() > 0
            else None
        ),
        "micro_f1_14": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "macro_f1_14": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1_5": float(
            f1_score(
                labels[:, CHEXBERT_FIVE_INDICES],
                predictions[:, CHEXBERT_FIVE_INDICES],
                average="micro",
                zero_division=0,
            )
        ),
        "macro_f1_5": float(
            f1_score(
                labels[:, CHEXBERT_FIVE_INDICES],
                predictions[:, CHEXBERT_FIVE_INDICES],
                average="macro",
                zero_division=0,
            )
        ),
        "exact_set_accuracy_14": float(np.all(labels == predictions, axis=1).mean()),
        "brier_score": float(np.mean((probabilities - labels) ** 2)),
        "expected_calibration_error": expected_calibration_error(labels, probabilities),
        "risk_coverage": risk_coverage_curve(labels, probabilities, thresholds),
        "per_label": per_label,
    }


def logistic_probabilities(
    embeddings: np.ndarray,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    embeddings = np.asarray(embeddings, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    intercepts = np.asarray(intercepts, dtype=np.float64)
    logits = embeddings @ coefficients.T + intercepts
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def spectrum_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    spectra: Sequence[str],
) -> Mapping[str, dict[str, Any]]:
    if len(spectra) != len(labels):
        raise ValueError("spectra and labels differ in length")
    output = {}
    values = np.asarray(list(map(str, spectra)))
    for spectrum in ("normal", "abnormal", "indeterminate"):
        selected = values == spectrum
        if selected.any():
            output[spectrum] = multilabel_metrics(
                labels[selected], probabilities[selected], thresholds
            )
    return output


__all__ = [
    "case_id_fingerprint",
    "expected_calibration_error",
    "logistic_probabilities",
    "macro_auprc",
    "multilabel_metrics",
    "risk_coverage_curve",
    "select_f1_thresholds",
    "spectrum_metrics",
    "supported_label_indices",
]
