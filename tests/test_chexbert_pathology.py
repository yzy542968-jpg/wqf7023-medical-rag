from __future__ import annotations

import numpy as np
import pytest

from medical_rag.evaluation.chexbert_pathology import (
    CHEXBERT_FIVE_INDICES,
    build_case_statistics,
    logits_to_rrg_binary,
    metrics_from_case_statistics,
    paired_case_bootstrap,
    random_control_case_bootstrap,
)


def test_rrg_binary_conversion_matches_official_policy() -> None:
    class_ids = np.asarray([[0, 1, 2, 3] * 3 + [0, 1]], dtype=np.int64)
    labels = logits_to_rrg_binary(class_ids)
    assert labels.tolist() == [[0, 1, 0, 1] * 3 + [0, 1]]


def test_case_metrics_count_omissions_and_additions() -> None:
    references = np.zeros((2, 14), dtype=np.int8)
    predictions = np.zeros((2, 14), dtype=np.int8)
    references[0, [1, 4]] = 1
    predictions[0, [1, 5]] = 1
    references[1, CHEXBERT_FIVE_INDICES[0]] = 1
    predictions[1, CHEXBERT_FIVE_INDICES[0]] = 1
    stats = build_case_statistics(["A", "A"], references, predictions)
    metrics = metrics_from_case_statistics(stats)
    assert metrics["case_count"] == 1
    assert metrics["row_count"] == 2
    assert metrics["reference_positive_omission_count"] == 1
    assert metrics["predicted_positive_addition_count"] == 1
    assert metrics["micro_f1_14"] == pytest.approx(2.0 * 2 / (2.0 * 2 + 1 + 1))
    assert metrics["mean_reference_positive_recall"] == pytest.approx(0.75)
    assert metrics["mean_predicted_positive_precision"] == pytest.approx(0.75)


def test_paired_case_bootstrap_is_deterministic_and_paired() -> None:
    references = np.zeros((4, 14), dtype=np.int8)
    references[:, 1] = 1
    weak = np.zeros((4, 14), dtype=np.int8)
    strong = references.copy()
    case_ids = ["A", "A", "B", "B"]
    weak_stats = build_case_statistics(case_ids, references, weak)
    strong_stats = build_case_statistics(case_ids, references, strong)
    first = paired_case_bootstrap(strong_stats, weak_stats, iterations=100, seed=17)
    second = paired_case_bootstrap(strong_stats, weak_stats, iterations=100, seed=17)
    assert first == second
    assert first["micro_f1_14"]["mean_difference"] == pytest.approx(1.0)
    assert first["micro_f1_14"]["ci_95_low"] == pytest.approx(1.0)


def test_random_control_bootstrap_averages_assignments_before_inference() -> None:
    references = np.zeros((4, 14), dtype=np.int8)
    references[:, 1] = 1
    selected = references.copy()
    weak = np.zeros((4, 14), dtype=np.int8)
    medium = references.copy()
    medium[::2, 1] = 0
    case_ids = ["A", "A", "B", "B"]
    result = random_control_case_bootstrap(
        build_case_statistics(case_ids, references, selected),
        [
            build_case_statistics(case_ids, references, weak),
            build_case_statistics(case_ids, references, medium),
        ],
        iterations=100,
        seed=21,
    )
    assert result["micro_f1_14"]["mean_difference"] == pytest.approx(2.0 / 3.0)

