from __future__ import annotations

import numpy as np

from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.similar_case.v10_runtime import (
    component_agreement,
    normalized_scores_and_reciprocal_ranks,
    r4_feature_matrix,
)


def test_bm25_score_all_matches_search_order() -> None:
    retriever = BM25Retriever().fit(
        [
            {"case_id": "a", "report_text": "left pleural effusion"},
            {"case_id": "b", "report_text": "normal chest"},
        ]
    )
    scores = retriever.score_all("pleural effusion")
    assert len(scores) == 2
    assert retriever.search("pleural effusion", top_k=1)[0]["case_id"] == "a"
    assert scores[0] > scores[1]


def test_runtime_feature_matrix_has_frozen_width() -> None:
    values = np.asarray([0.0, 1.0, 0.5], dtype=np.float32)
    features = r4_feature_matrix(values, values[::-1], values, question_type="findings")
    assert features.shape == (3, 9)
    assert np.allclose(features[:, -3:], [[1.0, 0.0, 0.0]] * 3)


def test_score_normalization_uses_stable_reciprocal_ranks() -> None:
    normalized, reciprocal = normalized_scores_and_reciprocal_ranks(
        np.asarray([2.0, 2.0, 0.0], dtype=np.float32)
    )
    assert np.allclose(normalized, [1.0, 1.0, 0.0])
    assert np.allclose(reciprocal, [1.0, 0.5, 1.0 / 3.0])


def test_component_agreement_counts_selected_top1_matches() -> None:
    result = {
        "bm25": np.asarray([2.0, 1.0]),
        "image_image": np.asarray([0.0, 1.0]),
        "image_report": np.asarray([3.0, 2.0]),
    }
    assert component_agreement(result, 0) == 2.0 / 3.0
