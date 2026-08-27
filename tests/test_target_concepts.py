from __future__ import annotations

import numpy as np
import pytest

from medical_rag.evaluation.target_concepts import (
    expected_calibration_error,
    logistic_probabilities,
    macro_auprc,
    multilabel_metrics,
    select_f1_thresholds,
)


def test_threshold_selection_prefers_higher_threshold_on_exact_tie() -> None:
    labels = np.zeros((4, 14), dtype=np.int8)
    labels[:2, 0] = 1
    probabilities = np.zeros((4, 14), dtype=np.float64)
    probabilities[:, 0] = [0.9, 0.8, 0.2, 0.1]
    thresholds, records = select_f1_thresholds(labels, probabilities)
    assert thresholds[0] == pytest.approx(0.8)
    assert records[0]["calibration_f1"] == pytest.approx(1.0)
    assert all(value == pytest.approx(0.5) for value in thresholds[1:])


def test_multilabel_metrics_and_logistic_probabilities_are_bounded() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    coefficients = np.zeros((14, 2))
    coefficients[0] = [5.0, -5.0]
    probabilities = logistic_probabilities(embeddings, coefficients, np.zeros(14))
    labels = np.zeros((2, 14), dtype=np.int8)
    labels[0, 0] = 1
    metrics = multilabel_metrics(labels, probabilities, np.full(14, 0.5))
    assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0
    assert metrics["micro_f1_14"] > 0.0
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0
    assert len(metrics["risk_coverage"]) == 10


def test_macro_auprc_excludes_unsupported_labels() -> None:
    labels = np.zeros((4, 14), dtype=np.int8)
    labels[:2, 0] = 1
    probabilities = np.zeros((4, 14), dtype=np.float64)
    probabilities[:, 0] = [0.9, 0.8, 0.2, 0.1]
    assert macro_auprc(labels, probabilities) == pytest.approx(1.0)
    assert expected_calibration_error(labels, probabilities) >= 0.0


def test_multilabel_metrics_handle_single_class_subgroup() -> None:
    labels = np.zeros((3, 14), dtype=np.int8)
    probabilities = np.full((3, 14), 0.1, dtype=np.float64)
    metrics = multilabel_metrics(labels, probabilities, np.full(14, 0.5))
    assert metrics["supported_label_count"] == 0
    assert metrics["micro_auroc"] is None
    assert metrics["micro_auprc"] is None
