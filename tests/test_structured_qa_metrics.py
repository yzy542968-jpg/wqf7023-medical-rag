from __future__ import annotations

import json

import numpy as np
import pytest

from medical_rag.qa.structured_metrics import (
    fit_label_majority,
    load_answer_vector,
    repeat_prediction,
    structured_qa_metrics,
)


def test_structured_metrics_are_label_balanced() -> None:
    targets = np.asarray([[1, 0], [1, 0], [1, 1], [1, 1]], dtype=np.uint8)
    predictions = np.asarray([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.uint8)
    metrics = structured_qa_metrics(targets, predictions)
    assert metrics.element_accuracy == pytest.approx(0.75)
    assert metrics.supported_label_macro_f1 == pytest.approx(0.5)
    assert metrics.micro_f1 == pytest.approx(0.8)
    assert metrics.exact_report_vector_accuracy == pytest.approx(0.5)


def test_fit_and_repeat_majority() -> None:
    targets = np.asarray([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)
    majority = fit_label_majority(targets)
    assert majority.tolist() == [1, 1]
    assert repeat_prediction(majority, 2).tolist() == [[1, 1], [1, 1]]


def test_load_answer_vector_validates_keys(tmp_path) -> None:
    path = tmp_path / "vector.json"
    path.write_text(json.dumps({"a": True, "b": False}), encoding="utf-8")
    assert load_answer_vector(path, ("a", "b")).tolist() == [1, 0]
    with pytest.raises(ValueError):
        load_answer_vector(path, ("a", "c"))
