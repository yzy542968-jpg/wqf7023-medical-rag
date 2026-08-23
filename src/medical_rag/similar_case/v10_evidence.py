from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def normalized_text(value: object) -> str:
    return " ".join(str(value or "").split())


def sentence_units(case_id: str, section: str, text: object) -> list["EvidenceUnit"]:
    cleaned = normalized_text(text)
    if not cleaned:
        return []
    parts = [part.strip() for part in SENTENCE_BOUNDARY.split(cleaned) if part.strip()]
    return [
        EvidenceUnit(
            case_id=str(case_id),
            section=section,
            unit_type="sentence",
            unit_index=index,
            text=part,
            source_sha256=hashlib.sha256(part.encode("utf-8")).hexdigest(),
            score=0.0,
        )
        for index, part in enumerate(parts)
    ]


def render_radgraph_fact(fact: str) -> str:
    parts = [normalized_text(part) for part in str(fact).split("|")]
    if len(parts) == 3 and parts[0] == "entity":
        return f"{parts[1]} [{parts[2]}]"
    if len(parts) == 5 and parts[0] == "relation":
        return f"{parts[1]} [{parts[2]}] {parts[3]} {parts[4]}"
    return normalized_text(fact)


def fact_units(case_id: str, facts: Iterable[str]) -> list["EvidenceUnit"]:
    rendered = sorted({render_radgraph_fact(fact) for fact in facts if normalized_text(fact)})
    return [
        EvidenceUnit(
            case_id=str(case_id),
            section="radgraph",
            unit_type="fact",
            unit_index=index,
            text=text,
            source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            score=0.0,
        )
        for index, text in enumerate(rendered)
    ]


@dataclass(frozen=True)
class EvidenceUnit:
    case_id: str
    section: str
    unit_type: str
    unit_index: int
    text: str
    source_sha256: str
    score: float

    @property
    def provenance_id(self) -> str:
        return f"{self.case_id}:{self.section}:{self.unit_type}:{self.unit_index}"

    def with_score(self, score: float) -> "EvidenceUnit":
        return EvidenceUnit(
            case_id=self.case_id,
            section=self.section,
            unit_type=self.unit_type,
            unit_index=self.unit_index,
            text=self.text,
            source_sha256=self.source_sha256,
            score=float(score),
        )


def rank_units(query: str, units: Sequence[EvidenceUnit]) -> list[EvidenceUnit]:
    if not units:
        return []
    query = normalized_text(query)
    if not query:
        return list(units)
    corpus = [query, *(unit.text for unit in units)]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)
    matrix = vectorizer.fit_transform(corpus)
    scores = (matrix[1:] @ matrix[0].T).toarray().reshape(-1)
    ranked = [unit.with_score(float(score)) for unit, score in zip(units, scores, strict=True)]
    return sorted(ranked, key=lambda unit: (-unit.score, unit.provenance_id))


def select_case_evidence(
    case: Mapping[str, object],
    *,
    query: str,
    facts: Iterable[str],
    policy: str,
) -> list[EvidenceUnit]:
    case_id = normalized_text(case.get("case_id"))
    if not case_id:
        raise ValueError("case lacks case_id")
    findings = sentence_units(case_id, "findings", case.get("findings"))
    impression = sentence_units(case_id, "impression", case.get("impression"))
    sentences = findings + impression
    if policy == "whole_report":
        result = []
        for section, text in (("findings", case.get("findings")), ("impression", case.get("impression"))):
            value = normalized_text(text)
            if value:
                result.extend(sentence_units(case_id, section, value))
        return result
    if policy == "sentence_top3":
        return rank_units(query, sentences)[:3]
    if policy == "sentence_top2_fact_top5":
        return rank_units(query, sentences)[:2] + rank_units(query, fact_units(case_id, facts))[:5]
    raise ValueError(f"unknown evidence policy: {policy}")


def evidence_diagnostics(units: Sequence[EvidenceUnit]) -> dict[str, float]:
    if not units:
        return {
            "unit_count": 0.0,
            "mean_score": 0.0,
            "max_score": 0.0,
            "positive_fact_fraction": 0.0,
            "negative_fact_fraction": 0.0,
            "uncertain_fact_fraction": 0.0,
            "redundancy": 0.0,
        }
    fact_texts = [unit.text.lower() for unit in units if unit.unit_type == "fact"]
    positive = sum("definitely present" in text for text in fact_texts)
    negative = sum("definitely absent" in text for text in fact_texts)
    uncertain = sum("uncertain" in text for text in fact_texts)
    fact_denominator = max(len(fact_texts), 1)
    normalized = [normalized_text(unit.text).lower() for unit in units]
    redundancy = 1.0 - len(set(normalized)) / len(normalized)
    return {
        "unit_count": float(len(units)),
        "mean_score": float(np.mean([unit.score for unit in units])),
        "max_score": float(max(unit.score for unit in units)),
        "positive_fact_fraction": positive / fact_denominator,
        "negative_fact_fraction": negative / fact_denominator,
        "uncertain_fact_fraction": uncertain / fact_denominator,
        "redundancy": float(redundancy),
    }


__all__ = [
    "EvidenceUnit",
    "evidence_diagnostics",
    "fact_units",
    "normalized_text",
    "rank_units",
    "render_radgraph_fact",
    "select_case_evidence",
    "sentence_units",
]

