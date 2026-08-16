from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np


def l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero vector.")
    return array / norm


def aggregate_view_embeddings(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("At least one image embedding is required.")
    normalized = np.stack([l2_normalize(item) for item in embeddings])
    return l2_normalize(normalized.mean(axis=0))


def rank_scores(case_ids: Sequence[str], scores: Sequence[float]) -> list[str]:
    if len(case_ids) != len(scores):
        raise ValueError("case_ids and scores must have equal length.")
    pairs = zip(case_ids, scores, strict=True)
    return [case_id for case_id, _ in sorted(pairs, key=lambda item: (-float(item[1]), item[0]))]


def reciprocal_rank_fusion(
    text_ranking: Sequence[str],
    image_ranking: Sequence[str],
    text_weight: float,
    constant: int = 60,
) -> list[str]:
    if not 0.0 <= text_weight <= 1.0:
        raise ValueError("text_weight must be between 0 and 1.")
    if constant <= 0:
        raise ValueError("constant must be positive.")
    if set(text_ranking) != set(image_ranking):
        raise ValueError("Both rankings must contain the same case IDs.")

    text_positions = {case_id: rank for rank, case_id in enumerate(text_ranking, start=1)}
    image_positions = {case_id: rank for rank, case_id in enumerate(image_ranking, start=1)}
    fused = {
        case_id: (
            text_weight / (constant + text_positions[case_id])
            + (1.0 - text_weight) / (constant + image_positions[case_id])
        )
        for case_id in text_positions
    }
    return rank_scores(list(fused), list(fused.values()))


def mean_reciprocal_rank(
    rankings: Mapping[str, Sequence[str]],
    relevant_case_ids: Mapping[str, str],
) -> float:
    if not relevant_case_ids:
        raise ValueError("At least one relevance judgment is required.")
    values = []
    for qid, target in relevant_case_ids.items():
        ranking = rankings[qid]
        try:
            values.append(1.0 / (ranking.index(target) + 1))
        except ValueError:
            values.append(0.0)
    return float(np.mean(values))


def select_text_weight(
    text_rankings: Mapping[str, Sequence[str]],
    image_rankings: Mapping[str, Sequence[str]],
    relevant_case_ids: Mapping[str, str],
    weight_grid: Iterable[float],
    constant: int = 60,
) -> dict[str, object]:
    weights = [float(weight) for weight in weight_grid]
    if not weights:
        raise ValueError("weight_grid cannot be empty.")
    if set(text_rankings) != set(image_rankings) or set(text_rankings) != set(relevant_case_ids):
        raise ValueError("Ranking and relevance query IDs must match.")

    sweep = []
    for weight in weights:
        fused = {
            qid: reciprocal_rank_fusion(
                text_rankings[qid], image_rankings[qid], weight, constant=constant
            )
            for qid in relevant_case_ids
        }
        sweep.append({"text_weight": weight, "mrr": mean_reciprocal_rank(fused, relevant_case_ids)})

    selected = sorted(
        sweep,
        key=lambda row: (-float(row["mrr"]), abs(float(row["text_weight"]) - 0.5), float(row["text_weight"])),
    )[0]
    return {"selected_text_weight": selected["text_weight"], "selected_mrr": selected["mrr"], "sweep": sweep}
