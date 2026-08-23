from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _normalized(value: object) -> str:
    return " ".join(str(value or "").lower().split())


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
        text = " ".join(str(row.get("text", "")).split())
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
    cleaned = " ".join(report_text.split())
    if cleaned in facts_by_text:
        return facts_by_text[cleaned], True
    try:
        from radgraph.utils import radgraph_xl_preprocess_report
    except ImportError:
        return frozenset(), False
    preprocessed = radgraph_xl_preprocess_report(cleaned)
    if preprocessed in facts_by_text:
        return facts_by_text[preprocessed], True
    return frozenset(), False
