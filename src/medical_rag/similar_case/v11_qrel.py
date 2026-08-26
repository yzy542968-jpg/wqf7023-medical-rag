"""Development-only, structured proxy relevance for the V11 extension.

This module deliberately does not replace the frozen V10 relevance function.
The score is a transparent report-derived proxy, not a clinical gold label.
Empty feature sets receive zero similarity rather than accidental full credit.
Missing components are also reported explicitly: the conservative score keeps
the missing component at zero, while an availability-normalized sensitivity
score excludes components that cannot be compared from its denominator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from medical_rag.similar_case.relevance import active_label_weights


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SEVERITY = {
    "mild", "minimal", "moderate", "marked", "severe", "large", "small",
    "borderline", "prominent", "extensive", "subtle", "trace", "massive",
}
_UNCERTAINTY = {"uncertain", "possible", "probable", "suggestive", "cannot exclude"}


def _tokens(value: object) -> set[str]:
    return set(_TOKEN_RE.findall(" ".join(str(value or "").lower().split())))


def _f1(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    if not overlap:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2.0 * precision * recall / (precision + recall)


def _problem_tokens(case: Mapping[str, Any]) -> set[str]:
    raw = case.get("problems")
    if raw is None:
        raw = case.get("mesh")
    if isinstance(raw, (list, tuple, set)):
        values = raw
    else:
        values = re.split(r"[;,]", str(raw or ""))
    result = set()
    for value in values:
        normalized = " ".join(str(value).lower().split())
        if normalized and normalized not in {"normal", "no indexing", "none", "nan"}:
            result.add(normalized)
    return result


def report_index_spectrum(case: Mapping[str, Any]) -> str:
    normalized = " ".join(str(case.get("problems", "")).lower().split())
    if normalized == "normal":
        return "report_indexed_normal"
    if normalized in {"", "no indexing"}:
        return "report_index_indeterminate"
    return "report_indexed_abnormal"


@dataclass(frozen=True)
class FactAttributes:
    lesion_type: frozenset[str]
    anatomy: frozenset[str]
    severity: frozenset[str]
    polarity: frozenset[str]
    uncertainty: frozenset[str]


@dataclass(frozen=True)
class PreparedQrelCase:
    """Cached case features for exact qrel-v2 scoring."""

    attributes: FactAttributes
    indication_tokens: frozenset[str]
    label_set: frozenset[str]
    spectrum: str
    fact_annotation_available: bool


def fact_attributes(facts: Sequence[str] | set[str] | frozenset[str]) -> FactAttributes:
    lesion: set[str] = set()
    anatomy: set[str] = set()
    severity: set[str] = set()
    polarity: set[str] = set()
    uncertainty: set[str] = set()
    for raw in facts:
        parts = [" ".join(str(part).lower().split()) for part in str(raw).split("|")]
        if not parts:
            continue
        if parts[0] == "entity" and len(parts) >= 3:
            text, label = parts[1], parts[2]
            label_parts = label.split("::", 1)
            family = label_parts[0]
            status = label_parts[1] if len(label_parts) == 2 else ""
            if family == "anatomy":
                anatomy.update(_tokens(text))
            else:
                lesion.update(_tokens(text))
            if status:
                polarity.add(status)
                if status in _UNCERTAINTY:
                    uncertainty.add(status)
            severity.update(_tokens(text) & _SEVERITY)
        elif parts[0] == "relation" and len(parts) >= 5:
            subject, label, relation, target = parts[1:5]
            if "observation" in label:
                lesion.update(_tokens(subject))
            if relation in {"located_at", "modify"}:
                anatomy.update(_tokens(target))
            severity.update(_tokens(subject) & _SEVERITY)
            label_parts = label.split("::", 1)
            if len(label_parts) == 2:
                polarity.add(label_parts[1])
                if label_parts[1] in _UNCERTAINTY:
                    uncertainty.add(label_parts[1])
    return FactAttributes(
        frozenset(lesion), frozenset(anatomy), frozenset(severity),
        frozenset(polarity), frozenset(uncertainty),
    )


def _label_sets(case: Mapping[str, Any]) -> set[str]:
    labels = case.get("labels", {})
    active = active_label_weights(labels) if isinstance(labels, Mapping) else {}
    active.update({f"problem:{value}": 1.0 for value in _problem_tokens(case)})
    return set(active)


def prepare_qrel_case(
    case: Mapping[str, Any],
    facts_by_case: Mapping[str, Sequence[str]] | None = None,
) -> PreparedQrelCase:
    facts_by_case = facts_by_case or {}
    facts = facts_by_case.get(str(case.get("case_id")), case.get("radgraph_facts", ()))
    return PreparedQrelCase(
        attributes=fact_attributes(facts),
        indication_tokens=frozenset(_tokens(case.get("indication"))),
        label_set=frozenset(_label_sets(case)),
        spectrum=report_index_spectrum(case),
        fact_annotation_available=bool(facts),
    )


def qrel_v2_profile_prepared(
    query: PreparedQrelCase,
    candidate: PreparedQrelCase,
) -> dict[str, float | str | dict[str, bool]]:
    q_attr = query.attributes
    c_attr = candidate.attributes
    components = {
        "lesion_type": _f1(set(q_attr.lesion_type), set(c_attr.lesion_type)),
        "anatomy": _f1(set(q_attr.anatomy), set(c_attr.anatomy)),
        "severity": _f1(set(q_attr.severity), set(c_attr.severity)),
        "polarity": _f1(set(q_attr.polarity), set(c_attr.polarity)),
        "uncertainty": _f1(set(q_attr.uncertainty), set(c_attr.uncertainty)),
        "indication": _f1(set(query.indication_tokens), set(candidate.indication_tokens)),
        "report_label": _f1(set(query.label_set), set(candidate.label_set)),
    }
    weights = {"lesion_type": 0.25, "anatomy": 0.20, "severity": 0.10, "polarity": 0.15, "uncertainty": 0.10, "indication": 0.10, "report_label": 0.10}
    availability = {
        "lesion_type": bool(q_attr.lesion_type and c_attr.lesion_type),
        "anatomy": bool(q_attr.anatomy and c_attr.anatomy),
        "severity": bool(q_attr.severity and c_attr.severity),
        "polarity": bool(q_attr.polarity and c_attr.polarity),
        "uncertainty": bool(q_attr.uncertainty and c_attr.uncertainty),
        "indication": bool(query.indication_tokens and candidate.indication_tokens),
        "report_label": bool(query.label_set and candidate.label_set),
    }
    conservative_score = sum(weights[name] * float(value) for name, value in components.items())
    available_weight = sum(weights[name] for name, is_available in availability.items() if is_available)
    normalized_score = (
        sum(weights[name] * float(components[name]) for name, is_available in availability.items() if is_available)
        / available_weight
        if available_weight
        else 0.0
    )
    return {
        **{name: float(value) for name, value in components.items()},
        "component_available": {name: bool(value) for name, value in availability.items()},
        "available_weight": float(available_weight),
        "availability_fraction": float(available_weight / sum(weights.values())),
        "qrel_v2": float(max(0.0, min(1.0, conservative_score))),
        "qrel_v2_available_normalized": float(max(0.0, min(1.0, normalized_score))),
        "query_spectrum": query.spectrum,
        "candidate_spectrum": candidate.spectrum,
        "fact_annotation_available": float(query.fact_annotation_available and candidate.fact_annotation_available),
    }


def qrel_v2_profile(query: Mapping[str, Any], candidate: Mapping[str, Any], facts_by_case: Mapping[str, Sequence[str]] | None = None) -> dict[str, float | str]:
    facts_by_case = facts_by_case or {}
    return qrel_v2_profile_prepared(
        prepare_qrel_case(query, facts_by_case),
        prepare_qrel_case(candidate, facts_by_case),
    )


def build_qrels_v2(query: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], facts_by_case: Mapping[str, Sequence[str]] | None = None) -> dict[str, float]:
    result: dict[str, float] = {}
    for candidate in candidates:
        case_id = str(candidate.get("case_id", "")).strip()
        if not case_id or case_id in result:
            raise ValueError(f"Invalid or duplicate candidate case_id: {case_id!r}")
        result[case_id] = float(qrel_v2_profile(query, candidate, facts_by_case)["qrel_v2"])
    return result


__all__ = [
    "FactAttributes", "PreparedQrelCase", "build_qrels_v2", "fact_attributes",
    "prepare_qrel_case", "qrel_v2_profile", "qrel_v2_profile_prepared",
    "report_index_spectrum",
]
