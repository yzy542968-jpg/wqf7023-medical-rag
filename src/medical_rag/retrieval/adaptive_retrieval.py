from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveRetrievalDecision:
    selected_case_id: str | None
    source: str
    abstained: bool
    base_margin: float
    reranker_margin: float
    reason: str


def score_margin(scores: list[float]) -> float:
    if len(scores) < 2:
        return float("inf") if scores else 0.0
    ordered = sorted(scores, reverse=True)
    return ordered[0] - ordered[1]


def select_adaptive_top1(
    *,
    base_case_ids: list[str],
    base_scores: list[float],
    reranked_case_ids: list[str],
    reranker_scores: list[float],
    reranker_margin_threshold: float,
    base_margin_threshold: float,
    minimum_base_score: float = 0.0,
    minimum_selected_margin: float = 0.0,
) -> AdaptiveRetrievalDecision:
    if not base_case_ids or not reranked_case_ids:
        return AdaptiveRetrievalDecision(
            selected_case_id=None,
            source="none",
            abstained=True,
            base_margin=0.0,
            reranker_margin=0.0,
            reason="no_candidates",
        )
    if len(base_case_ids) != len(base_scores):
        raise ValueError("base case IDs and scores must have the same length")
    if len(reranked_case_ids) != len(reranker_scores):
        raise ValueError("reranked case IDs and scores must have the same length")

    base_margin = score_margin(base_scores)
    reranker_margin = score_margin(reranker_scores)
    base_top1 = base_case_ids[0]
    reranker_top1 = reranked_case_ids[0]

    if base_top1 == reranker_top1:
        selected = base_top1
        source = "agreement"
        selected_margin = max(base_margin, reranker_margin)
        reason = "retrievers_agree"
    elif (
        reranker_margin >= reranker_margin_threshold
        and base_margin <= base_margin_threshold
    ):
        selected = reranker_top1
        source = "reranker"
        selected_margin = reranker_margin
        reason = "reranker_confident_base_uncertain"
    else:
        selected = base_top1
        source = "hybrid"
        selected_margin = base_margin
        reason = "retain_hybrid_ranking"

    low_base_score = max(base_scores) < minimum_base_score
    low_margin = selected_margin < minimum_selected_margin
    if low_base_score and low_margin:
        return AdaptiveRetrievalDecision(
            selected_case_id=None,
            source=source,
            abstained=True,
            base_margin=base_margin,
            reranker_margin=reranker_margin,
            reason="low_retrieval_confidence",
        )
    return AdaptiveRetrievalDecision(
        selected_case_id=selected,
        source=source,
        abstained=False,
        base_margin=base_margin,
        reranker_margin=reranker_margin,
        reason=reason,
    )
