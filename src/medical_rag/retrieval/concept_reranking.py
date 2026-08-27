from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from medical_rag.evaluation.chexbert_pathology import CHEXBERT_LABELS


FEATURE_NAMES = (
    "soft_agreement",
    "soft_positive_recall",
    "soft_candidate_precision",
    "soft_cosine",
    "no_finding_agreement",
    "query_concept_confidence",
)


def cluster_fold_assignments(
    case_ids: Sequence[object],
    case_to_cluster: Mapping[str, object],
    *,
    folds: int = 5,
    seed: int = 7145,
) -> dict[str, int]:
    if folds < 2:
        raise ValueError("folds must be at least two")
    assignments: dict[str, int] = {}
    for raw_case_id in case_ids:
        case_id = str(raw_case_id).strip()
        if not case_id or case_id not in case_to_cluster:
            raise ValueError(f"Missing canonical cluster for case {case_id!r}")
        cluster_id = str(case_to_cluster[case_id]).strip()
        if not cluster_id:
            raise ValueError(f"Empty cluster ID for case {case_id!r}")
        payload = f"v14-oof|{seed}|{cluster_id}".encode("utf-8")
        assignments[case_id] = int.from_bytes(hashlib.sha256(payload).digest(), "big") % folds
    return assignments


def concept_agreement_features(
    query_probabilities: np.ndarray,
    candidate_labels: np.ndarray,
    *,
    epsilon: float = 1e-8,
) -> np.ndarray:
    probabilities = np.asarray(query_probabilities, dtype=np.float64)
    labels = np.asarray(candidate_labels, dtype=np.float64)
    if probabilities.shape != (len(CHEXBERT_LABELS),):
        raise ValueError("query probabilities must have shape (14,)")
    if labels.ndim != 2 or labels.shape[1] != len(CHEXBERT_LABELS):
        raise ValueError("candidate labels must have shape (n, 14)")
    if not np.isfinite(probabilities).all() or not np.isfinite(labels).all():
        raise ValueError("concept inputs must be finite")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("query probabilities must be in [0, 1]")
    if np.any((labels < 0.0) | (labels > 1.0)):
        raise ValueError("candidate labels must be binary")

    overlap = labels * probabilities[None, :]
    probability_mass = max(float(probabilities.sum()), epsilon)
    label_mass = np.maximum(labels.sum(axis=1), epsilon)
    probability_norm = max(float(np.linalg.norm(probabilities)), epsilon)
    label_norm = np.maximum(np.linalg.norm(labels, axis=1), epsilon)
    no_finding_index = CHEXBERT_LABELS.index("No Finding")
    no_finding_probability = probabilities[no_finding_index]
    no_finding_label = labels[:, no_finding_index]
    confidence = float(np.mean(np.abs(probabilities - 0.5) * 2.0))

    features = np.column_stack(
        (
            1.0 - np.mean(np.abs(labels - probabilities[None, :]), axis=1),
            overlap.sum(axis=1) / probability_mass,
            overlap.sum(axis=1) / label_mass,
            overlap.sum(axis=1) / (probability_norm * label_norm),
            np.where(no_finding_label > 0.5, no_finding_probability, 1.0 - no_finding_probability),
            np.full(len(labels), confidence, dtype=np.float64),
        )
    )
    if not np.isfinite(features).all():
        raise RuntimeError("concept features contain non-finite values")
    return np.clip(features, 0.0, 1.0).astype(np.float32)


def append_concept_features(
    base_features: np.ndarray,
    query_probabilities: np.ndarray,
    candidate_labels: np.ndarray,
) -> np.ndarray:
    base = np.asarray(base_features, dtype=np.float32)
    if base.ndim != 2 or len(base) != len(candidate_labels):
        raise ValueError("base features and candidate labels differ")
    concept = concept_agreement_features(query_probabilities, candidate_labels)
    return np.concatenate((base, concept), axis=1)


__all__ = [
    "FEATURE_NAMES",
    "append_concept_features",
    "cluster_fold_assignments",
    "concept_agreement_features",
]
