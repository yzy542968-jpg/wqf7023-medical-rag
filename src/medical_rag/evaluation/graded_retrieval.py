from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence


def discounted_cumulative_gain(gains: Sequence[float], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    total = 0.0
    for rank, gain in enumerate(gains[:k], start=1):
        value = float(gain)
        if not 0.0 <= value <= 1.0:
            raise ValueError("Relevance gains must be within [0, 1].")
        total += (2.0**value - 1.0) / math.log2(rank + 1.0)
    return total


def ndcg_at_k(
    qrels: Mapping[str, float],
    ranking: Sequence[str],
    k: int,
) -> float:
    if len(ranking) != len(set(ranking)):
        raise ValueError("Ranking contains duplicate study IDs.")
    observed = [float(qrels.get(study_id, 0.0)) for study_id in ranking]
    ideal = sorted((float(value) for value in qrels.values()), reverse=True)
    ideal_dcg = discounted_cumulative_gain(ideal, k)
    if ideal_dcg == 0.0:
        return 0.0
    return discounted_cumulative_gain(observed, k) / ideal_dcg


def binary_recall_at_k(
    qrels: Mapping[str, float],
    ranking: Sequence[str],
    k: int,
    *,
    threshold: float = 0.50,
) -> float:
    relevant = {study_id for study_id, gain in qrels.items() if float(gain) >= threshold}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranking[:k])) / len(relevant)


def reciprocal_rank_at_threshold(
    qrels: Mapping[str, float],
    ranking: Sequence[str],
    *,
    threshold: float = 0.50,
) -> float:
    for rank, study_id in enumerate(ranking, start=1):
        if float(qrels.get(study_id, 0.0)) >= threshold:
            return 1.0 / rank
    return 0.0


def evaluate_graded_retrieval(
    qrels: Mapping[str, Mapping[str, float]],
    rankings: Mapping[str, Sequence[str]],
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
    binary_threshold: float = 0.50,
) -> dict[str, float]:
    if not qrels:
        return {}
    if not 0.0 <= binary_threshold <= 1.0:
        raise ValueError("binary_threshold must be within [0, 1].")

    output: dict[str, float] = {}
    query_count = len(qrels)
    for k in k_values:
        output[f"ndcg@{k}"] = sum(
            ndcg_at_k(query_qrels, rankings.get(qid, []), k)
            for qid, query_qrels in qrels.items()
        ) / query_count
        output[f"recall@{k}"] = sum(
            binary_recall_at_k(
                query_qrels,
                rankings.get(qid, []),
                k,
                threshold=binary_threshold,
            )
            for qid, query_qrels in qrels.items()
        ) / query_count
    output["mrr"] = sum(
        reciprocal_rank_at_threshold(
            query_qrels,
            rankings.get(qid, []),
            threshold=binary_threshold,
        )
        for qid, query_qrels in qrels.items()
    ) / query_count
    return output


def evaluate_grouped_graded_retrieval(
    qrels: Mapping[str, Mapping[str, float]],
    rankings: Mapping[str, Sequence[str]],
    query_group_ids: Mapping[str, str],
    *,
    k_values: tuple[int, ...] = (1, 5, 10),
    binary_threshold: float = 0.50,
) -> dict[str, float]:
    """Average within patient first so patients with more queries get no extra weight."""

    if set(qrels) != set(query_group_ids):
        raise ValueError("Every qrel query must have exactly one group ID.")
    grouped_queries: dict[str, list[str]] = defaultdict(list)
    for qid, group_id in query_group_ids.items():
        normalized = str(group_id).strip()
        if not normalized:
            raise ValueError(f"Query {qid} has an empty group ID.")
        grouped_queries[normalized].append(qid)
    if not grouped_queries:
        return {}

    per_group: list[dict[str, float]] = []
    for group_qids in grouped_queries.values():
        group_qrels = {qid: qrels[qid] for qid in group_qids}
        group_rankings = {qid: rankings.get(qid, []) for qid in group_qids}
        per_group.append(
            evaluate_graded_retrieval(
                group_qrels,
                group_rankings,
                k_values=k_values,
                binary_threshold=binary_threshold,
            )
        )
    return {
        metric: sum(group[metric] for group in per_group) / len(per_group)
        for metric in per_group[0]
    }
