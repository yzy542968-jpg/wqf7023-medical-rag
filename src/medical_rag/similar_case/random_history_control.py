from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence


def canonical_case_id(case_id: object) -> str:
    value = str(case_id).strip()
    if not value:
        raise ValueError("case_id must be non-empty")
    return value


def random_history_order_key(
    *,
    seed: int,
    assignment: int,
    target_case_id: object,
    question_type: str,
    candidate_case_id: object,
) -> str:
    if assignment < 0:
        raise ValueError("assignment must be non-negative")
    target = canonical_case_id(target_case_id)
    candidate = canonical_case_id(candidate_case_id)
    question = str(question_type).strip().lower()
    if not question:
        raise ValueError("question_type must be non-empty")
    payload = (
        f"v10-random-history|{int(seed)}|{int(assignment)}|"
        f"{target}|{question}|{candidate}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_random_history(
    candidate_case_ids: Sequence[object],
    *,
    seed: int,
    assignment: int,
    target_case_id: object,
    question_type: str,
    excluded_case_ids: Iterable[object] = (),
    count: int = 3,
) -> tuple[str, ...]:
    if count < 1:
        raise ValueError("count must be positive")
    target = canonical_case_id(target_case_id)
    excluded = {canonical_case_id(case_id) for case_id in excluded_case_ids}
    excluded.add(target)
    candidates = sorted(
        {
            canonical_case_id(case_id)
            for case_id in candidate_case_ids
            if canonical_case_id(case_id) not in excluded
        },
        key=lambda candidate: (
            random_history_order_key(
                seed=seed,
                assignment=assignment,
                target_case_id=target,
                question_type=question_type,
                candidate_case_id=candidate,
            ),
            candidate,
        ),
    )
    if len(candidates) < count:
        raise ValueError("Not enough eligible random-history candidates")
    return tuple(candidates[:count])


__all__ = [
    "canonical_case_id",
    "random_history_order_key",
    "select_random_history",
]

