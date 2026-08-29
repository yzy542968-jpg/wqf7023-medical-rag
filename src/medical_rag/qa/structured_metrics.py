from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def load_report_keys(dataset_root: str | Path) -> tuple[str, ...]:
    path = Path(dataset_root) / "report_keys.json"
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("Rad-ReStruct report_keys.json must be a list of strings")
    if len(values) != len(set(values)):
        raise ValueError("Rad-ReStruct report keys must be unique")
    return tuple(values)


def load_answer_vector(
    path: str | Path, report_keys: Sequence[str]
) -> np.ndarray:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Structured answer vector must be an object: {path}")
    missing = [key for key in report_keys if key not in payload]
    extra = sorted(set(payload) - set(report_keys))
    if missing or extra:
        raise ValueError(
            f"Answer-vector keys do not match report space; missing={len(missing)}, "
            f"extra={len(extra)}"
        )
    return np.asarray([bool(payload[key]) for key in report_keys], dtype=np.uint8)


def fit_label_majority(targets: np.ndarray) -> np.ndarray:
    matrix = _binary_matrix(targets, "targets")
    return (matrix.mean(axis=0) >= 0.5).astype(np.uint8)


def repeat_prediction(prediction: np.ndarray, rows: int) -> np.ndarray:
    vector = np.asarray(prediction, dtype=np.uint8)
    if vector.ndim != 1:
        raise ValueError("Majority prediction must be one-dimensional")
    return np.repeat(vector[None, :], rows, axis=0)


def _binary_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError(f"{name} must contain only binary values")
    return matrix.astype(np.uint8, copy=False)


@dataclass(frozen=True)
class StructuredQAMetrics:
    case_count: int
    label_count: int
    supported_label_count: int
    supported_label_macro_f1: float
    supported_label_macro_precision: float
    supported_label_macro_recall: float
    micro_f1: float
    element_accuracy: float
    exact_report_vector_accuracy: float
    reference_positive_rate: float
    predicted_positive_rate: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "case_count": self.case_count,
            "label_count": self.label_count,
            "supported_label_count": self.supported_label_count,
            "supported_label_macro_f1": self.supported_label_macro_f1,
            "supported_label_macro_precision": self.supported_label_macro_precision,
            "supported_label_macro_recall": self.supported_label_macro_recall,
            "micro_f1": self.micro_f1,
            "element_accuracy": self.element_accuracy,
            "exact_report_vector_accuracy": self.exact_report_vector_accuracy,
            "reference_positive_rate": self.reference_positive_rate,
            "predicted_positive_rate": self.predicted_positive_rate,
        }


def structured_qa_metrics(
    targets: np.ndarray, predictions: np.ndarray
) -> StructuredQAMetrics:
    target_matrix = _binary_matrix(targets, "targets")
    prediction_matrix = _binary_matrix(predictions, "predictions")
    if target_matrix.shape != prediction_matrix.shape:
        raise ValueError("targets and predictions must have identical shapes")

    target_positive = target_matrix.sum(axis=0, dtype=np.int64)
    predicted_positive = prediction_matrix.sum(axis=0, dtype=np.int64)
    true_positive = np.logical_and(target_matrix, prediction_matrix).sum(
        axis=0, dtype=np.int64
    )
    false_positive = predicted_positive - true_positive
    false_negative = target_positive - true_positive
    supported = target_positive > 0
    if not supported.any():
        raise ValueError("No answer label has positive support in the reference matrix")

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    f1_denominator = 2 * true_positive + false_positive + false_negative
    precision = np.divide(
        true_positive,
        precision_denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=precision_denominator > 0,
    )
    recall = np.divide(
        true_positive,
        recall_denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=recall_denominator > 0,
    )
    f1 = np.divide(
        2 * true_positive,
        f1_denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=f1_denominator > 0,
    )

    total_tp = int(true_positive.sum())
    total_fp = int(false_positive.sum())
    total_fn = int(false_negative.sum())
    micro_denominator = 2 * total_tp + total_fp + total_fn
    micro_f1 = 2 * total_tp / micro_denominator if micro_denominator else 0.0
    return StructuredQAMetrics(
        case_count=int(target_matrix.shape[0]),
        label_count=int(target_matrix.shape[1]),
        supported_label_count=int(supported.sum()),
        supported_label_macro_f1=float(f1[supported].mean()),
        supported_label_macro_precision=float(precision[supported].mean()),
        supported_label_macro_recall=float(recall[supported].mean()),
        micro_f1=float(micro_f1),
        element_accuracy=float((target_matrix == prediction_matrix).mean()),
        exact_report_vector_accuracy=float(
            np.all(target_matrix == prediction_matrix, axis=1).mean()
        ),
        reference_positive_rate=float(target_matrix.mean()),
        predicted_positive_rate=float(prediction_matrix.mean()),
    )


def stack_answer_vectors(vectors: Iterable[np.ndarray]) -> np.ndarray:
    rows = [np.asarray(vector, dtype=np.uint8) for vector in vectors]
    if not rows:
        raise ValueError("At least one answer vector is required")
    lengths = {row.shape for row in rows}
    if len(lengths) != 1 or rows[0].ndim != 1:
        raise ValueError("All answer vectors must be one-dimensional and equal length")
    return np.stack(rows)
