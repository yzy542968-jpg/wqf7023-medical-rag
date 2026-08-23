from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from medical_rag.similar_case.schema import PairedCase


POSITIVE_STRINGS = {"1", "1.0", "positive", "present", "yes", "true"}
UNCERTAIN_STRINGS = {"-1", "-1.0", "uncertain", "possible", "u"}
INACTIVE_STRINGS = {
    "",
    "0",
    "0.0",
    "negative",
    "absent",
    "no",
    "false",
    "missing",
    "nan",
    "none",
}


def label_status_weight(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric):
            return 0.0
        if numeric == 1.0:
            return 1.0
        if numeric == -1.0 or numeric == 0.5:
            return 0.5
        if numeric == 0.0:
            return 0.0
        raise ValueError(f"Unsupported numeric report label: {value!r}")
    normalized = " ".join(str(value).lower().split())
    if normalized in POSITIVE_STRINGS:
        return 1.0
    if normalized in UNCERTAIN_STRINGS:
        return 0.5
    if normalized in INACTIVE_STRINGS:
        return 0.0
    raise ValueError(f"Unsupported report label: {value!r}")


def active_label_weights(labels: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for label, status in labels.items():
        key = " ".join(str(label).lower().split())
        if not key:
            raise ValueError("Report label names cannot be empty.")
        weight = label_status_weight(status)
        if weight > 0.0:
            normalized[key] = weight
    return normalized


def active_label_similarity(
    query_labels: Mapping[str, Any],
    candidate_labels: Mapping[str, Any],
) -> float:
    """Weighted Jaccard that gives no credit for shared absent labels."""

    query = active_label_weights(query_labels)
    candidate = active_label_weights(candidate_labels)
    keys = set(query) | set(candidate)
    if not keys:
        return 1.0
    numerator = sum(min(query.get(key, 0.0), candidate.get(key, 0.0)) for key in keys)
    denominator = sum(max(query.get(key, 0.0), candidate.get(key, 0.0)) for key in keys)
    return numerator / denominator if denominator else 0.0


def radgraph_fact_similarity(
    query_facts: Sequence[str] | frozenset[str],
    candidate_facts: Sequence[str] | frozenset[str],
) -> float:
    query = {" ".join(str(fact).lower().split()) for fact in query_facts if str(fact).strip()}
    candidate = {
        " ".join(str(fact).lower().split())
        for fact in candidate_facts
        if str(fact).strip()
    }
    if not query and not candidate:
        return 1.0
    if not query or not candidate:
        return 0.0
    overlap = len(query & candidate)
    precision = overlap / len(candidate)
    recall = overlap / len(query)
    return 2.0 * precision * recall / (precision + recall) if overlap else 0.0


def report_relevance_gain(
    query: PairedCase,
    candidate: PairedCase,
    *,
    active_label_weight: float = 0.60,
    radgraph_fact_weight: float = 0.40,
) -> float:
    if query.study_id == candidate.study_id:
        raise ValueError("A target study cannot be its own relevance candidate.")
    if (
        query.patient_id is not None
        and candidate.patient_id is not None
        and query.patient_id == candidate.patient_id
    ):
        raise ValueError("A same-patient study cannot be a V9 relevance candidate.")
    if active_label_weight < 0 or radgraph_fact_weight < 0:
        raise ValueError("Relevance weights cannot be negative.")
    unavailable = [
        case.study_id
        for case in (query, candidate)
        if case.metadata.get("label_annotation_available", True) is not True
    ]
    if unavailable:
        raise ValueError(
            "Report relevance found unavailable label annotations for "
            + ", ".join(unavailable)
        )
    total_weight = active_label_weight + radgraph_fact_weight
    if not math.isclose(total_weight, 1.0, abs_tol=1e-9):
        raise ValueError("Relevance weights must sum to 1.0.")

    label_score = active_label_similarity(query.labels, candidate.labels)
    fact_score = radgraph_fact_similarity(query.radgraph_facts, candidate.radgraph_facts)
    gain = active_label_weight * label_score + radgraph_fact_weight * fact_score
    return min(1.0, max(0.0, float(gain)))


def build_query_qrels(
    query: PairedCase,
    candidates: Sequence[PairedCase],
    *,
    active_label_weight: float = 0.60,
    radgraph_fact_weight: float = 0.40,
) -> dict[str, float]:
    qrels: dict[str, float] = {}
    for candidate in candidates:
        if candidate.study_id in qrels:
            raise ValueError(f"Duplicate candidate study ID: {candidate.study_id}")
        qrels[candidate.study_id] = report_relevance_gain(
            query,
            candidate,
            active_label_weight=active_label_weight,
            radgraph_fact_weight=radgraph_fact_weight,
        )
    return qrels
