from __future__ import annotations

import csv
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from medical_rag.similar_case.schema import PairedCase
from medical_rag.similar_case.radgraph_adapter import match_radgraph_facts


CHEXBERT_LABEL_COLUMNS = (
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
)


def _clean_text(value: object) -> str:
    normalized = " ".join(str(value or "").split())
    return "" if normalized.lower() in {"nan", "none"} else normalized


def canonical_chexpert_path(value: object) -> str:
    normalized = str(value or "").strip().replace("\\", "/")
    if not normalized:
        raise ValueError("path_to_image cannot be empty.")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("path_to_image must be a safe relative path.")
    return path.as_posix()


def parse_chexpert_patient_study(path_to_image: object) -> tuple[str, str]:
    path = canonical_chexpert_path(path_to_image)
    patient = next(
        (part for part in PurePosixPath(path).parts if re.fullmatch(r"patient\d+", part)),
        None,
    )
    study = next(
        (part for part in PurePosixPath(path).parts if re.fullmatch(r"study\d+", part)),
        None,
    )
    if patient is None or study is None:
        raise ValueError(f"Cannot parse patient/study IDs from path: {path}")
    return patient, f"{patient}/{study}"


def read_chexbert_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels_by_path: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                image_path = canonical_chexpert_path(row["path_to_image"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid CheXbert JSONL row at line {line_number}: {exc}"
                ) from exc
            if image_path in labels_by_path:
                raise ValueError(f"Duplicate CheXbert image path: {image_path}")
            labels_by_path[image_path] = {
                label: row.get(label) for label in CHEXBERT_LABEL_COLUMNS
            }
    return labels_by_path


def _merge_consistent_text(current: str, incoming: str, *, field_name: str) -> str:
    if not current:
        return incoming
    if not incoming or incoming == current:
        return current
    raise ValueError(f"Conflicting {field_name} text within one study.")


def _merge_consistent_labels(
    current: dict[str, Any] | None,
    incoming: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if incoming is None:
        return current
    normalized = dict(incoming)
    if current is None:
        return normalized
    for key in CHEXBERT_LABEL_COLUMNS:
        left = current.get(key)
        right = normalized.get(key)
        if left is None:
            current[key] = right
        elif right is not None and right != left:
            raise ValueError(f"Conflicting CheXbert label {key!r} within one study.")
    return current


def read_chexpert_plus_cases(
    csv_path: Path,
    *,
    image_root: Path,
    chexbert_labels_path: Path | None = None,
    radgraph_facts_by_findings: Mapping[str, Sequence[str]] | None = None,
    require_image_files: bool = True,
) -> list[PairedCase]:
    """Load official CheXpert Plus rows and aggregate multiple views by study."""

    labels_by_path = (
        read_chexbert_labels(chexbert_labels_path)
        if chexbert_labels_path is not None
        else {}
    )
    groups: dict[str, dict[str, Any]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path_to_image", "section_findings", "section_impression"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CheXpert Plus CSV is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                relative_path = canonical_chexpert_path(row["path_to_image"])
                patient_id, study_id = parse_chexpert_patient_study(relative_path)
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid CheXpert Plus row {row_number}: {exc}") from exc
            image_path = image_root.joinpath(*PurePosixPath(relative_path).parts)
            if require_image_files and not image_path.is_file():
                raise FileNotFoundError(image_path)

            group = groups.setdefault(
                study_id,
                {
                    "patient_id": patient_id,
                    "image_paths": [],
                    "indication": "",
                    "findings": "",
                    "impression": "",
                    "labels": None,
                },
            )
            if group["patient_id"] != patient_id:
                raise ValueError(f"Conflicting patient ID for study {study_id}.")
            group["image_paths"].append(str(image_path))
            indication = _clean_text(
                row.get("section_history") or row.get("section_indication")
            )
            for field_name, incoming in (
                ("indication", indication),
                ("findings", _clean_text(row.get("section_findings"))),
                ("impression", _clean_text(row.get("section_impression"))),
            ):
                group[field_name] = _merge_consistent_text(
                    group[field_name], incoming, field_name=field_name
                )
            group["labels"] = _merge_consistent_labels(
                group["labels"], labels_by_path.get(relative_path)
            )

    cases: list[PairedCase] = []
    for study_id in sorted(groups):
        group = groups[study_id]
        if not group["findings"] and not group["impression"]:
            continue
        facts: Sequence[str] = ()
        radgraph_available = False
        if radgraph_facts_by_findings is not None:
            facts, radgraph_available = match_radgraph_facts(
                group["findings"], radgraph_facts_by_findings
            )
        labels = group["labels"] or {}
        cases.append(
            PairedCase(
                study_id=study_id,
                patient_id=group["patient_id"],
                image_paths=tuple(sorted(set(group["image_paths"]))),
                indication=group["indication"],
                findings=group["findings"],
                impression=group["impression"],
                labels=labels,
                radgraph_facts=frozenset(facts),
                source="chexpert_plus_official",
                metadata={
                    "label_annotation_available": bool(labels),
                    "radgraph_annotation_available": radgraph_available,
                    "view_count": len(set(group["image_paths"])),
                    "released_patient_identifier_available": True,
                    "patient_key_basis": "released_patient_identifier",
                },
            )
        )
    return cases
