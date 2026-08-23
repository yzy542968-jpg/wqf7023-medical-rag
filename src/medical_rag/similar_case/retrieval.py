from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from medical_rag.multimodal.fusion import minmax_normalize


@dataclass(frozen=True)
class ScoredCase:
    study_id: str
    score: float
    component_scores: Mapping[str, float]


def rank_component_scores(scores: Mapping[str, float]) -> list[str]:
    return [
        study_id
        for study_id, _ in sorted(
            scores.items(), key=lambda item: (-float(item[1]), str(item[0]))
        )
    ]


def cosine_score_map(
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
    candidate_ids: list[str] | tuple[str, ...],
) -> dict[str, float]:
    """Return deterministic cosine scores for one query and aligned candidates."""

    query = np.asarray(query_embedding, dtype=np.float64).reshape(-1)
    candidates = np.asarray(candidate_embeddings, dtype=np.float64)
    if candidates.ndim != 2:
        raise ValueError("Candidate embeddings must be a two-dimensional matrix.")
    if candidates.shape[0] != len(candidate_ids):
        raise ValueError("Candidate IDs and embedding rows must have equal length.")
    if candidates.shape[1] != query.shape[0]:
        raise ValueError("Query and candidate embedding dimensions must match.")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate IDs must be unique.")
    query_norm = np.linalg.norm(query)
    candidate_norms = np.linalg.norm(candidates, axis=1)
    if query_norm <= 0.0 or np.any(candidate_norms <= 0.0):
        raise ValueError("Cosine scoring requires non-zero embeddings.")
    scores = (candidates @ query) / (candidate_norms * query_norm)
    return {
        str(study_id): float(scores[index])
        for index, study_id in enumerate(candidate_ids)
    }


def fuse_component_scores(
    component_scores: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, float],
) -> list[ScoredCase]:
    """Independently min-max normalize and fuse aligned component scores."""

    if not component_scores:
        raise ValueError("At least one score component is required.")
    if set(component_scores) != set(weights):
        raise ValueError("Component names and weight names must match exactly.")
    if any(float(weight) < 0.0 for weight in weights.values()):
        raise ValueError("Fusion weights cannot be negative.")
    weight_total = sum(float(weight) for weight in weights.values())
    if weight_total <= 0.0:
        raise ValueError("At least one fusion weight must be positive.")

    candidate_sets = [set(scores) for scores in component_scores.values()]
    if not candidate_sets[0]:
        return []
    if any(candidates != candidate_sets[0] for candidates in candidate_sets[1:]):
        raise ValueError("All score components must cover the same study IDs.")

    study_ids = sorted(candidate_sets[0])
    normalized: dict[str, np.ndarray] = {}
    for component, scores in component_scores.items():
        normalized[component] = minmax_normalize([scores[study_id] for study_id in study_ids])

    normalized_weights = {
        component: float(weight) / weight_total for component, weight in weights.items()
    }
    fused = np.zeros(len(study_ids), dtype=np.float64)
    for component, values in normalized.items():
        fused += normalized_weights[component] * values

    rows = [
        ScoredCase(
            study_id=study_id,
            score=float(fused[index]),
            component_scores={
                component: float(scores[study_id])
                for component, scores in component_scores.items()
            },
        )
        for index, study_id in enumerate(study_ids)
    ]
    return sorted(rows, key=lambda row: (-row.score, row.study_id))
