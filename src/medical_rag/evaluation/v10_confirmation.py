from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np


def graded_ndcg(gains: np.ndarray, ranking: np.ndarray, *, k: int = 10) -> float:
    valid = np.asarray(gains, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    ideal = np.sort(valid)[::-1][:k]
    if ideal.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(ideal) + 2, dtype=np.float64))
    ideal_dcg = float(np.sum((np.power(2.0, ideal) - 1.0) / discounts))
    if ideal_dcg <= 0.0:
        return 0.0
    observed = np.asarray(gains, dtype=np.float64)[np.asarray(ranking[:k], dtype=np.int64)]
    observed = np.where(np.isfinite(observed), observed, 0.0)
    observed_dcg = float(
        np.sum((np.power(2.0, observed) - 1.0) / discounts[: len(observed)])
    )
    return observed_dcg / ideal_dcg


def reciprocal_rank_at_threshold(
    gains: np.ndarray,
    ranking: np.ndarray,
    *,
    threshold: float = 0.5,
) -> float:
    for rank, index in enumerate(ranking, start=1):
        value = float(gains[int(index)])
        if np.isfinite(value) and value >= threshold:
            return 1.0 / rank
    return 0.0


def hit_at_k(
    gains: np.ndarray,
    ranking: np.ndarray,
    *,
    k: int,
    threshold: float = 0.5,
) -> float:
    return float(
        any(
            np.isfinite(float(gains[int(index)]))
            and float(gains[int(index)]) >= threshold
            for index in ranking[:k]
        )
    )


def deterministic_derangement(
    case_ids: Sequence[str],
    *,
    assignment_index: int,
    seed: int,
) -> dict[str, str]:
    identifiers = sorted(set(map(str, case_ids)))
    if len(identifiers) < 2:
        raise ValueError("at least two unique case IDs are required")
    if not 0 <= assignment_index < len(identifiers) - 1:
        raise ValueError("assignment_index must be smaller than case_count - 1")
    ordered = sorted(
        identifiers,
        key=lambda case_id: (
            hashlib.sha256(f"v10-shuffle|{seed}|{case_id}".encode("utf-8")).hexdigest(),
            case_id,
        ),
    )
    shift = assignment_index + 1
    mapping = {
        case_id: ordered[(index + shift) % len(ordered)]
        for index, case_id in enumerate(ordered)
    }
    if any(source == target for source, target in mapping.items()):
        raise RuntimeError("derangement contains a fixed point")
    return mapping


def case_grouped_bootstrap_difference(
    rows: Sequence[Mapping[str, object]],
    *,
    left: str,
    right: str,
    metric: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["case_id"])][str(row["system"])].append(float(row[metric]))
    differences = np.asarray(
        [
            np.mean(values[left]) - np.mean(values[right])
            for values in grouped.values()
            if left in values and right in values
        ],
        dtype=np.float64,
    )
    if differences.size == 0:
        raise ValueError("no paired cases found")
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        samples[index] = float(np.mean(rng.choice(differences, size=len(differences), replace=True)))
    return {
        "mean_difference": float(np.mean(differences)),
        "ci_95_low": float(np.quantile(samples, 0.025)),
        "ci_95_high": float(np.quantile(samples, 0.975)),
        "case_count": float(len(differences)),
    }


def plus_one_monte_carlo_p(aligned: float, shuffled_values: Sequence[float]) -> float:
    values = np.asarray(shuffled_values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("shuffled_values cannot be empty")
    return float((1 + np.sum(values >= aligned)) / (len(values) + 1))


__all__ = [
    "case_grouped_bootstrap_difference",
    "deterministic_derangement",
    "graded_ndcg",
    "hit_at_k",
    "plus_one_monte_carlo_p",
    "reciprocal_rank_at_threshold",
]
