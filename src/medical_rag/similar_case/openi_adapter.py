from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from medical_rag.similar_case.schema import PairedCase
from medical_rag.similar_case.radgraph_adapter import (
    RadGraphCaseRecord,
    read_radgraph_case_records,
)


def _openi_report_index_class(value: object) -> str:
    normalized = " ".join(str(value or "").lower().split())
    if normalized == "normal":
        return "normal"
    if not normalized or normalized == "no indexing":
        return "indeterminate"
    return "abnormal"


def _openi_problem_labels(value: object) -> dict[str, float]:
    labels = [" ".join(part.lower().split()) for part in str(value or "").split(";")]
    labels = [label for label in labels if label and label not in {"normal", "no indexing"}]
    return {label: 1.0 for label in labels}


def openi_row_to_paired_case(
    row: Mapping[str, Any],
    *,
    image_root: Path | None = None,
    source_unique_patient: bool = False,
    radgraph_record: RadGraphCaseRecord | None = None,
) -> PairedCase:
    image_paths = []
    for image in row.get("images", []):
        filename = str(image.get("filename", "")).strip()
        if not filename:
            continue
        image_paths.append(str(image_root / filename) if image_root else filename)
    labels = _openi_problem_labels(row.get("problems"))
    report_index_class = _openi_report_index_class(row.get("problems"))
    study_id = str(row.get("case_id", ""))
    if radgraph_record is not None and radgraph_record.case_id != study_id:
        raise ValueError(
            f"RadGraph case {radgraph_record.case_id} does not match OpenI case {study_id}."
        )
    formal_radgraph = radgraph_record is not None
    radgraph_facts = radgraph_record.facts if formal_radgraph else frozenset(labels)
    radgraph_available = (
        radgraph_record.status == "ok"
        if formal_radgraph
        else report_index_class != "indeterminate"
    )
    patient_id = f"openi-source-unique:{study_id}" if source_unique_patient else None
    return PairedCase(
        study_id=study_id,
        patient_id=patient_id,
        image_paths=tuple(image_paths),
        indication=str(row.get("indication", "")),
        findings=str(row.get("findings", "")),
        impression=str(row.get("impression", "")),
        labels=labels,
        radgraph_facts=radgraph_facts,
        source=(
            "openi_iu_xray_primary_source"
            if source_unique_patient
            else "openi_engineering_smoke_only"
        ),
        metadata={
            "problems": row.get("problems", ""),
            "mesh": row.get("mesh", ""),
            "report_index_class": report_index_class,
            "label_annotation_available": report_index_class != "indeterminate",
            "radgraph_annotation_available": radgraph_available,
            "radgraph_annotation_source": (
                radgraph_record.model_type
                if formal_radgraph
                else "problem_label_proxy_not_formal_radgraph"
            ),
            "radgraph_report_text_sha256": (
                radgraph_record.report_text_sha256 if formal_radgraph else None
            ),
            "released_patient_identifier_available": False,
            "source_collection_one_study_per_patient": source_unique_patient,
            "patient_key_basis": (
                "source_design_one_study_per_patient"
                if source_unique_patient
                else "unavailable"
            ),
        },
    )


def read_openi_paired_cases(
    path: Path,
    *,
    image_root: Path | None = None,
    source_unique_patient: bool = False,
    radgraph_path: Path | None = None,
) -> list[PairedCase]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[PairedCase] = []
    radgraph_records = (
        read_radgraph_case_records(radgraph_path) if radgraph_path is not None else None
    )
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                case_id = str(raw.get("case_id", "")).strip()
                if radgraph_records is not None and case_id not in radgraph_records:
                    raise ValueError(f"OpenI case {case_id} lacks a RadGraph record.")
                rows.append(
                    openi_row_to_paired_case(
                        raw,
                        image_root=image_root,
                        source_unique_patient=source_unique_patient,
                        radgraph_record=(
                            radgraph_records[case_id]
                            if radgraph_records is not None
                            else None
                        ),
                    )
                )
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid OpenI row at line {line_number}: {exc}") from exc
    if radgraph_records is not None:
        source_ids = {case.study_id for case in rows}
        extra = set(radgraph_records) - source_ids
        if extra:
            raise ValueError(
                "RadGraph records contain unknown OpenI cases: "
                + ", ".join(sorted(extra)[:5])
            )
    return rows


def iter_openi_question_rows(cases: Iterable[PairedCase]) -> Iterable[dict[str, str]]:
    questions = {
        "findings": "What are the main radiographic findings?",
        "impression": "What is the most likely radiographic impression?",
        "acute": "Is there an acute cardiopulmonary abnormality? Explain briefly.",
    }
    for case in cases:
        for question_type, question in questions.items():
            yield {
                "qid": f"{case.study_id}:{question_type}",
                "study_id": case.study_id,
                "question_type": question_type,
                "question": question,
            }
