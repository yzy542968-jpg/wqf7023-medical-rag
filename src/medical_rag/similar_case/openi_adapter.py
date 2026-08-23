from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from medical_rag.similar_case.schema import PairedCase


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
) -> PairedCase:
    image_paths = []
    for image in row.get("images", []):
        filename = str(image.get("filename", "")).strip()
        if not filename:
            continue
        image_paths.append(str(image_root / filename) if image_root else filename)
    labels = _openi_problem_labels(row.get("problems"))
    report_index_class = _openi_report_index_class(row.get("problems"))
    return PairedCase(
        study_id=str(row.get("case_id", "")),
        patient_id=None,
        image_paths=tuple(image_paths),
        indication=str(row.get("indication", "")),
        findings=str(row.get("findings", "")),
        impression=str(row.get("impression", "")),
        labels=labels,
        radgraph_facts=frozenset(labels),
        source="openi_engineering_smoke_only",
        metadata={
            "problems": row.get("problems", ""),
            "mesh": row.get("mesh", ""),
            "report_index_class": report_index_class,
            "label_annotation_available": report_index_class != "indeterminate",
            "patient_level_independence_available": False,
        },
    )


def read_openi_paired_cases(
    path: Path,
    *,
    image_root: Path | None = None,
) -> list[PairedCase]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[PairedCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                rows.append(openi_row_to_paired_case(raw, image_root=image_root))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid OpenI row at line {line_number}: {exc}") from exc
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
