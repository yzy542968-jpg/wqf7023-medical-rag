"""Deterministic candidate-pool construction for development audits.

The functions here operate before reranking. They deliberately keep the
candidate budget explicit so a hybrid union cannot hide first-stage recall
failures behind a later scorer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def rank_scores(scores: Mapping[str, float]) -> list[str]:
    """Return a stable descending score ranking."""

    return [
        case_id
        for case_id, _ in sorted(
            ((str(case_id), float(score)) for case_id, score in scores.items()),
            key=lambda item: (-item[1], item[0]),
        )
    ]


def top_k_union(rankings: Sequence[Sequence[str]], k: int) -> list[str]:
    """Return a deterministic union of the first ``k`` items per ranking."""

    if k <= 0:
        raise ValueError("k must be positive")
    result: list[str] = []
    seen: set[str] = set()
    for ranking in rankings:
        for raw_case_id in ranking[:k]:
            case_id = str(raw_case_id)
            if case_id not in seen:
                seen.add(case_id)
                result.append(case_id)
    return result


def reciprocal_rank_fusion_union(
    rankings: Sequence[Sequence[str]],
    *,
    source_top_k: int,
    output_k: int,
    constant: int = 60,
) -> list[str]:
    """Fuse top-k source rankings with RRF and return a fixed candidate budget."""

    if source_top_k <= 0 or output_k <= 0 or constant <= 0:
        raise ValueError("source_top_k, output_k and constant must be positive")
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, raw_case_id in enumerate(ranking[:source_top_k], start=1):
            case_id = str(raw_case_id)
            scores[case_id] = scores.get(case_id, 0.0) + 1.0 / (constant + rank)
    return [case_id for case_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:output_k]]


__all__ = ["rank_scores", "reciprocal_rank_fusion_union", "top_k_union"]
