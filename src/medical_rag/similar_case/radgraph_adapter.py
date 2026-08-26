from __future__ import annotations

import json
import re
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RadGraphCaseRecord:
    case_id: str
    status: str
    facts: frozenset[str]
    report_text_sha256: str
    model_type: str


def _normalized(value: object) -> str:
    return " ".join(str(value or "").lower().split())


def _report_text_key(value: object) -> str:
    """Normalize whitespace without depending on the optional RadGraph package."""

    collapsed = " ".join(str(value or "").split())
    return re.sub(r"\s+([.,!?;:])", r"\1", collapsed)


def unwrap_radgraph_annotation(annotation: Mapping[str, Any]) -> Mapping[str, Any]:
    if "entities" in annotation and "text" in annotation:
        return annotation
    if "0" in annotation and isinstance(annotation["0"], Mapping):
        inner = annotation["0"]
        if "entities" in inner and "text" in inner:
            return inner
    raise ValueError("Unsupported RadGraph annotation structure.")


def radgraph_annotation_facts(annotation: Mapping[str, Any]) -> frozenset[str]:
    """Flatten RadGraph entities/relations using complete-reward semantics."""

    row = unwrap_radgraph_annotation(annotation)
    entities = row.get("entities")
    if not isinstance(entities, Mapping):
        raise ValueError("RadGraph entities must be a mapping.")
    facts: set[str] = set()
    for entity_id, entity in entities.items():
        if not isinstance(entity, Mapping):
            raise ValueError(f"Invalid RadGraph entity: {entity_id}")
        tokens = _normalized(entity.get("tokens"))
        label = _normalized(entity.get("label"))
        if not tokens or not label:
            raise ValueError(f"RadGraph entity {entity_id} lacks tokens or label.")
        relations = entity.get("relations") or []
        if not relations:
            facts.add(f"entity|{tokens}|{label}")
            continue
        for relation in relations:
            if not isinstance(relation, (list, tuple)) or len(relation) != 2:
                raise ValueError(f"Invalid relation on RadGraph entity {entity_id}.")
            relation_label, target_id = relation
            target = entities.get(str(target_id))
            if not isinstance(target, Mapping):
                raise ValueError(
                    f"RadGraph relation on entity {entity_id} has missing target."
                )
            target_tokens = _normalized(target.get("tokens"))
            if not target_tokens:
                raise ValueError("RadGraph relation target lacks tokens.")
            facts.add(
                "relation|"
                f"{tokens}|{label}|{_normalized(relation_label)}|{target_tokens}"
            )
    return frozenset(facts)


def read_radgraph_facts_by_text(path: Path) -> dict[str, frozenset[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    annotations = raw if isinstance(raw, list) else list(raw.values())
    facts_by_text: dict[str, frozenset[str]] = {}
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise ValueError(f"Invalid RadGraph annotation at index {index}.")
        row = unwrap_radgraph_annotation(annotation)
        text = _report_text_key(row.get("text", ""))
        if not text:
            raise ValueError(f"RadGraph annotation {index} lacks text.")
        facts = radgraph_annotation_facts(annotation)
        if text in facts_by_text and facts_by_text[text] != facts:
            raise ValueError(f"Conflicting RadGraph annotations for text: {text[:80]}")
        facts_by_text[text] = facts
    return facts_by_text


def match_radgraph_facts(
    report_text: str,
    facts_by_text: Mapping[str, frozenset[str]],
) -> tuple[frozenset[str], bool]:
    cleaned = _report_text_key(report_text)
    if cleaned in facts_by_text:
        return facts_by_text[cleaned], True
    try:
        from radgraph.utils import radgraph_xl_preprocess_report
    except ImportError:
        return frozenset(), False
    preprocessed = _report_text_key(radgraph_xl_preprocess_report(cleaned))
    if preprocessed in facts_by_text:
        return facts_by_text[preprocessed], True
    return frozenset(), False


def read_radgraph_case_records(path: Path) -> dict[str, RadGraphCaseRecord]:
    """Read V9 checkpointed annotations without exposing raw annotations downstream."""

    if not path.exists():
        raise FileNotFoundError(path)
    records: dict[str, RadGraphCaseRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                case_id = str(row["case_id"]).strip()
                status = str(row["status"]).strip()
                report_text_sha256 = str(row["report_text_sha256"]).strip()
                model_type = str(row["model_type"]).strip()
                raw_facts = row.get("facts", [])
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid RadGraph case record at line {line_number}: {exc}"
                ) from exc
            if not case_id or not report_text_sha256 or not model_type:
                raise ValueError(f"Incomplete RadGraph case record at line {line_number}.")
            if status not in {"ok", "empty_report", "error"}:
                raise ValueError(f"Unsupported RadGraph status for {case_id}: {status}")
            if not isinstance(raw_facts, list):
                raise ValueError(f"RadGraph facts for {case_id} must be a list.")
            facts = frozenset(
                _normalized(value) for value in raw_facts if str(value).strip()
            )
            if status != "ok" and facts:
                raise ValueError(f"Unavailable RadGraph record {case_id} contains facts.")
            if case_id in records:
                raise ValueError(f"Duplicate RadGraph case record: {case_id}")
            records[case_id] = RadGraphCaseRecord(
                case_id=case_id,
                status=status,
                facts=facts,
                report_text_sha256=report_text_sha256,
                model_type=model_type,
            )
    return records
