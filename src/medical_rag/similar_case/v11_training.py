"""Safe training helpers for a future V11 learned reranker.

The helpers make the shortlist boundary explicit: a query with no positive
inside the candidate shortlist is evaluated as a retrieval failure but cannot
produce a fabricated pairwise training example.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def target_inside_shortlist(qrels: Sequence[float], *, positive_threshold: float = 0.5) -> bool:
    return any(np.isfinite(float(value)) and float(value) >= positive_threshold for value in qrels)


def pairwise_training_mask(qrels: Sequence[float], *, positive_threshold: float = 0.5) -> np.ndarray:
    """Return a mask for valid positive/negative construction.

    A mask with no positives is all false; the caller must retain that query in
    evaluation metrics and exclude it only from pairwise gradient updates.
    """

    values = np.asarray(qrels, dtype=np.float32)
    valid = np.isfinite(values)
    has_positive = bool(np.any(valid & (values >= positive_threshold)))
    return valid & has_positive


def hard_negative_indices(
    retrieval_scores: Sequence[float], qrels: Sequence[float], *, top_k: int = 20,
    maximum_qrel: float = 0.5,
) -> list[int]:
    scores = np.asarray(retrieval_scores, dtype=np.float64)
    relevance = np.asarray(qrels, dtype=np.float64)
    if scores.shape != relevance.shape:
        raise ValueError("retrieval_scores and qrels must have equal shape")
    valid = np.isfinite(scores) & np.isfinite(relevance) & (relevance <= maximum_qrel)
    return sorted(np.flatnonzero(valid).tolist(), key=lambda index: (-float(scores[index]), index))[:top_k]


def build_pairwise_examples(
    features: np.ndarray, qrels: Sequence[float], *, positive_threshold: float = 0.5,
    negative_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic positive/negative rows, or empty arrays on failure.

    This function does not remove the corresponding query from evaluation; it
    only returns no gradient examples when the shortlist has no positive.
    """

    values = np.asarray(qrels, dtype=np.float32)
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or len(values) != matrix.shape[0]:
        raise ValueError("features and qrels have incompatible shapes")
    positive = np.flatnonzero(np.isfinite(values) & (values >= positive_threshold))
    negative = np.flatnonzero(np.isfinite(values) & (values < negative_threshold))
    if not len(positive) or not len(negative):
        empty = np.empty((0, matrix.shape[1]), dtype=np.float32)
        return empty, empty.copy()
    positive_index = int(sorted(positive.tolist(), key=lambda index: (-float(values[index]), index))[0])
    negative_indices = sorted(negative.tolist(), key=lambda index: (float(values[index]), index))
    return np.repeat(matrix[positive_index : positive_index + 1], len(negative_indices), axis=0), matrix[negative_indices]


__all__ = ["build_pairwise_examples", "hard_negative_indices", "pairwise_training_mask", "target_inside_shortlist"]
