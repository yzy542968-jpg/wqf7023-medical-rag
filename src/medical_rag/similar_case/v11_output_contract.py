"""Compact, provenance-safe output contract for V11 generation."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from medical_rag.similar_case.v10_evidence import EvidenceUnit, normalized_text


_COMPLETE_SENTENCE = re.compile(r".+?[.!?](?=\s|$)")


def compact_generation_prompt(*, indication: str, question: str, planner_instruction: str, evidence: Sequence[EvidenceUnit], abstain: bool) -> str:
    lines = [f"[{unit.provenance_id}] {unit.text}" for unit in evidence] or ["No historical evidence selected."]
    history = "Do not use historical evidence; state that evidence is insufficient." if abstain else "Historical evidence is analogy only; do not attribute it to the target patient."
    return "\n".join([
        "You are a cautious radiology QA assistant.", history,
        "Return JSON only with keys: answer, uncertainty, abstain, evidence.",
        "answer must be at most two complete sentences; uncertainty is low, medium, or high.",
        "evidence must contain only provenance IDs listed below.", planner_instruction,
        f"Indication: {normalized_text(indication) or 'Not provided'}",
        f"Question: {normalized_text(question)}", "Evidence:", *lines,
        'Schema: {"answer":"...","uncertainty":"high","abstain":false,"evidence":[]}',
    ])


def answer_only_generation_prompt(*, indication: str, question: str, planner_instruction: str, evidence: Sequence[EvidenceUnit], abstain: bool) -> str:
    """Prompt for the robust two-stage contract used by V11 development.

    The model emits only the answer. Provenance, abstention and source labels
    are attached by deterministic code after generation, which avoids asking a
    small multimodal model to serialize long evidence IDs at the token limit.
    """
    lines = [f"[{unit.provenance_id}] {unit.text}" for unit in evidence] or ["No historical evidence selected."]
    history = "Do not use historical evidence; answer that the available evidence is insufficient." if abstain else "Historical reports are analogies from other patients, not proof about the target patient."
    return "\n".join([
        "You are a cautious radiology QA assistant.", history,
        "Return only the concise answer in at most two complete sentences.",
        "Do not output JSON, field names, citations, analysis, or a preamble.",
        planner_instruction,
        f"Indication: {normalized_text(indication) or 'Not provided'}",
        f"Question: {normalized_text(question)}", "Historical evidence:", *lines,
    ])


def _decode_object(text: str) -> tuple[dict[str, Any], bool]:
    decoder = json.JSONDecoder(strict=False)
    raw = str(text or "").strip()
    for start, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, True
    return {}, False


def _decode_line_contract(text: str, allowed_ids: set[str]) -> tuple[dict[str, Any], bool]:
    """Recover a common key-per-line output without calling it valid JSON.

    MedGemma sometimes follows the semantic schema but emits YAML-like lines.
    This deterministic recovery keeps the answer usable for an engineering
    diagnostic while preserving ``raw_json_valid=False`` and
    ``parser_repaired=True`` in the returned audit record.
    """
    raw = str(text or "")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not any(line.lower().startswith("answer:") for line in lines):
        return {}, False
    parsed: dict[str, Any] = {}
    answer_parts: list[str] = []
    in_answer = False
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("answer:"):
            in_answer = True
            answer_parts.append(line.split(":", 1)[1].strip())
        elif lowered.startswith("uncertainty:"):
            in_answer = False
            parsed["uncertainty"] = line.split(":", 1)[1].strip()
        elif lowered.startswith("abstain:"):
            in_answer = False
            parsed["abstain"] = line.split(":", 1)[1].strip().lower() == "true"
        elif lowered.startswith("evidence:"):
            in_answer = False
        elif in_answer:
            answer_parts.append(line)
    parsed["answer"] = " ".join(answer_parts).strip()
    parsed["evidence"] = [identifier for identifier in sorted(allowed_ids) if identifier in raw]
    return parsed, True


def bound_complete_sentences(text: str, maximum: int = 2) -> str:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    cleaned = normalized_text(text)
    complete = [normalized_text(match.group(0)) for match in _COMPLETE_SENTENCE.finditer(cleaned)]
    return " ".join(complete[:maximum]) if complete else cleaned


def parse_compact_output(text: str, allowed_evidence: Sequence[EvidenceUnit], *, no_history_expected: bool = False) -> dict[str, Any]:
    parsed, valid = _decode_object(text)
    allowed = {unit.provenance_id: unit for unit in allowed_evidence}
    raw_json_valid = valid
    parser_repaired = False
    if not valid:
        parsed, parser_repaired = _decode_line_contract(str(text or ""), set(allowed))
        valid = False
    answer = bound_complete_sentences(str(parsed.get("answer", "")))
    uncertainty = normalized_text(parsed.get("uncertainty", "high")).lower()
    if uncertainty not in {"low", "medium", "high"}:
        uncertainty, valid = "high", False
    raw_ids = parsed.get("evidence", [])
    if not isinstance(raw_ids, list):
        raw_ids, valid = [], False
    evidence = [str(value) for value in raw_ids if str(value) in allowed]
    if len(evidence) != len(raw_ids):
        valid = False
    abstain = bool(parsed.get("abstain", False))
    if no_history_expected and evidence:
        evidence, valid = [], False
    if not answer:
        answer, valid = "Unable to provide a reliable answer from the available evidence.", False
    return {
        "answer": answer, "uncertainty": uncertainty, "abstain": abstain,
        "evidence": evidence, "supporting_case_ids": sorted({allowed[item].case_id for item in evidence}),
        "structured_output_valid": bool(valid),
        "raw_json_valid": bool(raw_json_valid),
        "parser_repaired": bool(parser_repaired),
        "normalized_output_usable": bool(answer and valid or (answer and parser_repaired)),
        "raw_output": str(text or ""),
        "truncated_to_complete_sentences": True,
    }


def assemble_provenance_output(answer: str, *, evidence: Sequence[EvidenceUnit], uncertainty: str = "high", abstain: bool = False) -> dict[str, Any]:
    bounded = bound_complete_sentences(answer)
    selected = [] if abstain else list(evidence)
    return {
        "answer": bounded,
        "uncertainty": uncertainty if uncertainty in {"low", "medium", "high"} else "high",
        "abstain": bool(abstain),
        "evidence": [
            {"provenance_id": unit.provenance_id, "case_id": unit.case_id, "section": unit.section, "text": unit.text, "source_sha256": unit.source_sha256}
            for unit in selected
        ],
        "supporting_case_ids": sorted({unit.case_id for unit in selected}),
        "deterministic_provenance": True,
    }


__all__ = [
    "answer_only_generation_prompt", "assemble_provenance_output",
    "bound_complete_sentences", "compact_generation_prompt", "parse_compact_output",
]
