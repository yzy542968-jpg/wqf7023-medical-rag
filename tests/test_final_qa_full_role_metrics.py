from __future__ import annotations

import numpy as np

from scripts.evaluate_final_qa_full_role import official_compatible_metrics


def test_official_compatible_metrics_separates_root_questions() -> None:
    targets = np.asarray(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    predictions = targets.copy()
    report_keys = (
        "thorax_yes",
        "thorax_no",
        "thorax_sign_yes_location_left",
        "thorax_sign_no_location_right",
    )

    result = official_compatible_metrics(targets, predictions, report_keys)

    assert result["upstream_aggregate"]["f1"] == 1.0
    assert result["upstream_aggregate"]["exact_vector_accuracy"] == 1.0
    assert result["root_questions"]["label_count"] == 2
    assert result["root_questions"]["f1"] == 1.0


def test_official_compatible_metrics_rejects_missing_roots() -> None:
    targets = np.asarray([[1, 0]], dtype=np.uint8)

    try:
        official_compatible_metrics(
            targets,
            targets.copy(),
            ("thorax_location_left", "thorax_location_right"),
        )
    except ValueError as error:
        assert "root-question" in str(error)
    else:
        raise AssertionError("Expected a missing-root ValueError")
