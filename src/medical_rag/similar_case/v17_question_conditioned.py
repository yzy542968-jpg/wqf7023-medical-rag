"""Small deterministic utilities for the V17 exploratory retrieval study."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def minmax(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Feature values must be a finite one-dimensional array")
    span = float(array.max() - array.min()) if len(array) else 0.0
    if span <= 1e-12:
        return np.zeros_like(array)
    return (array - float(array.min())) / span


def weighted_ranking(
    candidate_ids: Sequence[str],
    features: np.ndarray,
    weights: Sequence[float],
) -> tuple[list[str], np.ndarray]:
    matrix = np.asarray(features, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape != (len(candidate_ids), len(weight_array)):
        raise ValueError("Candidate, feature, and weight dimensions do not align")
    normalized = np.column_stack([minmax(matrix[:, index]) for index in range(matrix.shape[1])])
    scores = normalized @ weight_array
    order = sorted(range(len(candidate_ids)), key=lambda index: (-float(scores[index]), str(candidate_ids[index])))
    return [str(candidate_ids[index]) for index in order], scores[np.asarray(order)]


def answer_stratum(answers: Sequence[str]) -> str:
    normalized = {" ".join(str(value).lower().split()) for value in answers}
    if normalized == {"yes"}:
        return "positive"
    if normalized == {"no"}:
        return "negative"
    return "non_binary"


def set_f1(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = {" ".join(str(value).lower().split()) for value in left}
    right_set = {" ".join(str(value).lower().split()) for value in right}
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    overlap = len(left_set & right_set)
    if not overlap:
        return 0.0
    return 2.0 * overlap / (len(left_set) + len(right_set))


def summarize_proxy_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one proxy row is required")
    strata: dict[str, dict[str, float | int]] = {}
    for name in ("positive", "negative", "non_binary"):
        selected = [row for row in rows if row["stratum"] == name]
        if not selected:
            continue
        strata[name] = {
            "question_count": len(selected),
            "top1_exact": float(np.mean([row["top1_exact"] for row in selected])),
            "top3_any_exact": float(np.mean([row["top3_any_exact"] for row in selected])),
            "top1_option_f1": float(np.mean([row["top1_option_f1"] for row in selected])),
        }
    balanced = float(np.mean([record["top1_exact"] for record in strata.values()]))
    return {
        "question_count": len(rows),
        "top1_exact": float(np.mean([row["top1_exact"] for row in rows])),
        "top3_any_exact": float(np.mean([row["top3_any_exact"] for row in rows])),
        "top1_option_f1": float(np.mean([row["top1_option_f1"] for row in rows])),
        "top1_same_question_coverage": float(np.mean([row["top1_covered"] for row in rows])),
        "top3_any_same_question_coverage": float(np.mean([row["top3_any_covered"] for row in rows])),
        "mean_top1_qrel_v2": float(np.mean([row.get("top1_qrel_v2", 0.0) for row in rows])),
        "balanced_top1_qid_answer_agreement": balanced,
        "strata": strata,
    }


def deterministic_top_ids(
    candidate_ids: Sequence[str], *, domain: str, seed: int, key: str, count: int
) -> list[str]:
    if count < 1 or count > len(candidate_ids):
        raise ValueError("Invalid deterministic selection count")
    return sorted(
        (str(value) for value in candidate_ids),
        key=lambda value: (
            hashlib.sha256(f"{domain}|{seed}|{key}|{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )[:count]


def fixed_point_free_permutation(keys: Sequence[str], *, domain: str, seed: int) -> dict[str, str]:
    unique = sorted(set(str(value) for value in keys))
    if len(unique) < 2:
        raise ValueError("At least two unique keys are required")
    ordered = sorted(
        unique,
        key=lambda value: (
            hashlib.sha256(f"{domain}|{seed}|{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )
    return {value: ordered[(index + 1) % len(ordered)] for index, value in enumerate(ordered)}


def select_complete_case_pilot(
    question_counts: Mapping[str, int], *, domain: str, seed: int, target_questions: int,
    maximum_questions: int,
) -> list[str]:
    if target_questions < 1 or maximum_questions < target_questions:
        raise ValueError("Invalid pilot question bounds")
    ordered = sorted(
        (str(case_id) for case_id in question_counts),
        key=lambda case_id: (
            hashlib.sha256(f"{domain}|{seed}|{case_id}".encode("utf-8")).hexdigest(),
            case_id,
        ),
    )
    selected: list[str] = []
    total = 0
    for case_id in ordered:
        count = int(question_counts[case_id])
        if count < 1:
            raise ValueError(f"Case {case_id} has no questions")
        if total >= target_questions:
            break
        if total + count > maximum_questions:
            continue
        selected.append(case_id)
        total += count
    if total < target_questions:
        raise ValueError("Could not reach the pilot target without exceeding its maximum")
    return selected


__all__ = [
    "answer_stratum",
    "deterministic_top_ids",
    "fixed_point_free_permutation",
    "minmax",
    "select_complete_case_pilot",
    "set_f1",
    "summarize_proxy_rows",
    "weighted_ranking",
]
