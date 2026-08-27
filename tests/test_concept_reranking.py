from __future__ import annotations

import numpy as np
import pytest

from medical_rag.evaluation.chexbert_pathology import CHEXBERT_LABELS
from medical_rag.retrieval.concept_reranking import (
    append_concept_features,
    cluster_fold_assignments,
    concept_agreement_features,
)


def test_cluster_folds_keep_duplicate_cluster_together() -> None:
    mapping = {"A": "cluster-1", "B": "cluster-1", "C": "cluster-2"}
    first = cluster_fold_assignments(["A", "B", "C"], mapping)
    second = cluster_fold_assignments(["C", "B", "A"], mapping)
    assert first == second
    assert first["A"] == first["B"]
    assert all(0 <= value < 5 for value in first.values())


def test_concept_features_reward_matching_candidate() -> None:
    probabilities = np.full(len(CHEXBERT_LABELS), 0.05)
    probabilities[CHEXBERT_LABELS.index("Cardiomegaly")] = 0.95
    labels = np.zeros((2, len(CHEXBERT_LABELS)))
    labels[0, CHEXBERT_LABELS.index("Cardiomegaly")] = 1
    labels[1, CHEXBERT_LABELS.index("No Finding")] = 1
    features = concept_agreement_features(probabilities, labels)
    assert features.shape == (2, 6)
    assert np.all((0.0 <= features) & (features <= 1.0))
    assert features[0, 0] > features[1, 0]
    assert features[0, 3] > features[1, 3]
    assert features[0, 4] > features[1, 4]


def test_append_concept_features_validates_shapes() -> None:
    base = np.ones((2, 17), dtype=np.float32)
    labels = np.zeros((2, len(CHEXBERT_LABELS)), dtype=np.int8)
    output = append_concept_features(base, np.full(len(CHEXBERT_LABELS), 0.5), labels)
    assert output.shape == (2, 23)
    with pytest.raises(ValueError):
        append_concept_features(base[:1], np.full(len(CHEXBERT_LABELS), 0.5), labels)

