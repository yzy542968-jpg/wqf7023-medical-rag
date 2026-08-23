from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from medical_rag.similar_case.v10_evidence import EvidenceUnit, normalized_text


def build_answer_prompt(
    *,
    indication: str,
    question: str,
    evidence: Sequence[EvidenceUnit],
    no_reliable_history: bool,
) -> str:
    evidence_lines = [
        f"[{unit.provenance_id}] {unit.text}" for unit in evidence
    ] or ["No historical evidence was selected."]
    history_instruction = (
        "Retrieval confidence was low. Do not rely on historical cases."
        if no_reliable_history
        else "Historical evidence is analogy only, not proof about the target patient."
    )
    return "\n".join(
        [
            "Answer the radiology question from the target image and indication.",
            history_instruction,
            "Return one short JSON object: {\"a\":\"answer\",\"u\":\"low|medium|high\"}.",
            "Use at most two concise sentences. Return JSON only.",
            f"Indication: {normalized_text(indication) or 'Not provided'}",
            f"Question: {normalized_text(question)}",
            "Historical evidence:",
            *evidence_lines,
        ]
    )


def build_support_prompt(answer: str, evidence: Sequence[EvidenceUnit]) -> str:
    lines = [f"[{unit.provenance_id}] {unit.text}" for unit in evidence]
    return "\n".join(
        [
            "Identify only historical statements that directly support the answer.",
            "Return {\"s\":[{\"p\":\"provenance_id\",\"t\":\"short support\"}]} or {\"s\":[]}.",
            "Use only listed provenance IDs. Historical evidence is not target-patient proof.",
            f"Answer: {normalized_text(answer)}",
            "Evidence:",
            *(lines or ["No evidence available."]),
        ]
    )


def _parse_object(text: str) -> tuple[dict[str, Any], bool]:
    raw = str(text or "").strip()
    decoder = json.JSONDecoder(strict=False)
    for start, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, True
    return {}, False


def parse_answer_stage(text: str) -> dict[str, Any]:
    parsed, valid = _parse_object(text)
    answer = normalized_text(parsed.get("a"))
    uncertainty = normalized_text(parsed.get("u")).lower()
    if uncertainty not in {"low", "medium", "high"}:
        uncertainty = "high"
        valid = False
    if not answer:
        answer = normalized_text(text)
        valid = False
    if not answer:
        answer = "Unable to provide a reliable answer from the available evidence."
    return {"answer": answer, "uncertainty": uncertainty, "answer_stage_valid": valid}


def parse_support_stage(
    text: str,
    evidence: Sequence[EvidenceUnit],
) -> dict[str, Any]:
    parsed, valid = _parse_object(text)
    allowed = {unit.provenance_id: unit for unit in evidence}
    raw_support = parsed.get("s", [])
    if not isinstance(raw_support, list):
        raw_support = []
        valid = False
    support = []
    for row in raw_support:
        if not isinstance(row, Mapping):
            valid = False
            continue
        provenance = normalized_text(row.get("p"))
        statement = normalized_text(row.get("t"))
        if provenance not in allowed or not statement:
            valid = False
            continue
        support.append(
            {
                "provenance_id": provenance,
                "case_id": allowed[provenance].case_id,
                "statement": statement,
                "source_sha256": allowed[provenance].source_sha256,
            }
        )
    return {"historical_support": support, "support_stage_valid": valid}


def assemble_output(
    answer_stage: Mapping[str, Any],
    support_stage: Mapping[str, Any],
    *,
    no_reliable_history: bool,
) -> dict[str, Any]:
    support = [] if no_reliable_history else list(support_stage.get("historical_support", []))
    return {
        "answer": normalized_text(answer_stage.get("answer")),
        "uncertainty": normalized_text(answer_stage.get("uncertainty")) or "high",
        "historical_support": support,
        "supporting_case_ids": sorted({str(row["case_id"]) for row in support}),
        "no_reliable_history": bool(no_reliable_history),
        "evidence_abstained": bool(no_reliable_history or not support),
        "answer_stage_valid": bool(answer_stage.get("answer_stage_valid", False)),
        "support_stage_valid": bool(support_stage.get("support_stage_valid", False)),
        "assembled_schema_valid": True,
    }


__all__ = [
    "assemble_output",
    "build_answer_prompt",
    "build_support_prompt",
    "parse_answer_stage",
    "parse_support_stage",
]
