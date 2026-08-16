from __future__ import annotations


def hit_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return float(bool(relevant_ids.intersection(retrieved_ids[:k])))


def recall_at_k(relevant_ids: set[str], retrieved_ids: list[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(relevant_ids.intersection(retrieved_ids[:k])) / len(relevant_ids)


def reciprocal_rank(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    for index, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in relevant_ids:
            return 1.0 / index
    return 0.0


def evaluate_retrieval(
    qrels: dict[str, set[str]],
    rankings: dict[str, list[str]],
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]:
    if not qrels:
        return {}

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"hit@{k}"] = sum(
            hit_at_k(qrels[qid], rankings.get(qid, []), k) for qid in qrels
        ) / len(qrels)
        metrics[f"recall@{k}"] = sum(
            recall_at_k(qrels[qid], rankings.get(qid, []), k) for qid in qrels
        ) / len(qrels)

    metrics["mrr"] = sum(
        reciprocal_rank(qrels[qid], rankings.get(qid, [])) for qid in qrels
    ) / len(qrels)
    return metrics

