"""Hierarchical case-to-fact evidence selection for V11 development."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from medical_rag.similar_case.v10_evidence import EvidenceUnit, fact_units, normalized_text, rank_units, sentence_units
from medical_rag.similar_case.v11_question_planner import QuestionPlan


@dataclass(frozen=True)
class HierarchicalEvidence:
    units: tuple[EvidenceUnit, ...]
    retrieved_case_ids: tuple[str, ...]
    selector: str = "v11_case_to_fact_deterministic"

    def as_records(self) -> list[dict[str, Any]]:
        return [
            {
                "provenance_id": unit.provenance_id,
                "case_id": unit.case_id,
                "section": unit.section,
                "unit_type": unit.unit_type,
                "unit_index": unit.unit_index,
                "text": unit.text,
                "source_sha256": unit.source_sha256,
                "score": unit.score,
            }
            for unit in self.units
        ]


def _units_for_case(case: Mapping[str, Any], facts: Sequence[str]) -> list[EvidenceUnit]:
    case_id = normalized_text(case.get("case_id"))
    return (
        sentence_units(case_id, "findings", case.get("findings"))
        + sentence_units(case_id, "impression", case.get("impression"))
        + fact_units(case_id, facts)
    )


def select_case_facts(
    case: Mapping[str, Any], *, query: str, facts: Sequence[str], plan: QuestionPlan,
    maximum_units: int = 2, maximum_characters: int = 520,
) -> list[EvidenceUnit]:
    units = _units_for_case(case, facts)
    section_order = {name: index for index, name in enumerate(plan.evidence_preferences)}
    ranked = rank_units(query, units)
    ranked = sorted(ranked, key=lambda unit: (-unit.score, section_order.get(unit.section, 99), unit.provenance_id))
    selected: list[EvidenceUnit] = []
    characters = 0
    for unit in ranked:
        if len(selected) >= maximum_units:
            break
        if normalized_text(unit.text).lower() in {normalized_text(item.text).lower() for item in selected}:
            continue
        extra = len(unit.text) + (1 if selected else 0)
        if selected and characters + extra > maximum_characters:
            continue
        selected.append(unit)
        characters += extra
    return selected


def select_hierarchical_evidence(
    retrieved_cases: Sequence[Mapping[str, Any]], *, query: str, facts_by_case: Mapping[str, Sequence[str]], plan: QuestionPlan,
    maximum_cases: int = 3, maximum_units_per_case: int = 2, maximum_total_units: int = 6,
    maximum_characters: int = 1200,
) -> HierarchicalEvidence:
    selected: list[EvidenceUnit] = []
    case_ids: list[str] = []
    characters = 0
    for case in list(retrieved_cases)[:maximum_cases]:
        case_id = normalized_text(case.get("case_id"))
        if not case_id:
            continue
        units = select_case_facts(
            case, query=query, facts=facts_by_case.get(case_id, ()), plan=plan,
            maximum_units=maximum_units_per_case,
            maximum_characters=max(1, maximum_characters - characters),
        )
        if not units:
            continue
        for unit in units:
            if len(selected) >= maximum_total_units:
                break
            extra = len(unit.text) + (1 if selected else 0)
            if selected and characters + extra > maximum_characters:
                continue
            selected.append(unit)
            characters += extra
        if units:
            case_ids.append(case_id)
        if len(selected) >= maximum_total_units:
            break
    return HierarchicalEvidence(tuple(selected), tuple(case_ids))


def evidence_profile(evidence: Sequence[EvidenceUnit]) -> dict[str, float]:
    texts = [normalized_text(unit.text) for unit in evidence]
    return {
        "unit_count": float(len(evidence)),
        "case_count": float(len({unit.case_id for unit in evidence})),
        "sentence_count": float(sum(unit.unit_type == "sentence" for unit in evidence)),
        "fact_count": float(sum(unit.unit_type == "fact" for unit in evidence)),
        "character_count": float(sum(len(text) for text in texts)),
        "provenance_complete_rate": (
            sum(bool(unit.case_id and unit.section and unit.source_sha256) for unit in evidence) / len(evidence)
            if evidence else 1.0
        ),
        "duplicate_text_rate": (
            1.0 - len(set(texts)) / len(texts) if texts else 0.0
        ),
    }


__all__ = ["HierarchicalEvidence", "evidence_profile", "select_case_facts", "select_hierarchical_evidence"]
