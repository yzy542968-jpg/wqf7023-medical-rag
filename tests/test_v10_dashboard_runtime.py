from __future__ import annotations

import numpy as np

from medical_rag.dashboard.v10_runtime import calibration_features, infer_question_type


def test_v10_dashboard_question_types_are_deterministic() -> None:
    assert infer_question_type("What is the most likely diagnosis?") == "impression"
    assert infer_question_type("Is there an acute cardiopulmonary abnormality?") == "acute"
    assert infer_question_type("What findings are visible?") == "findings"


def test_v10_dashboard_calibration_features_preserve_frozen_order() -> None:
    result = {
        "ranking": np.asarray([1, 0]),
        "ensemble_scores": np.asarray([0.1, 0.7]),
        "seed_scores": np.asarray([[0.2, 0.6], [0.1, 0.8]]),
        "bm25": np.asarray([0.2, 0.8]),
        "image_image": np.asarray([0.9, 0.1]),
        "image_report": np.asarray([0.3, 0.7]),
    }
    features = calibration_features(
        result,
        [],
        question_type="impression",
        view_count=1,
    )
    assert features["top1_score"] == 0.7
    assert features["top1_top2_margin"] == 0.6
    assert features["component_agreement"] == 2 / 3
    assert features["question_impression"] == 1.0
    assert features["question_findings"] == 0.0
    assert features["view_count"] == 1.0
