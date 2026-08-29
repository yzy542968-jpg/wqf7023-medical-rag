from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


RADRESTRUCT_SPLITS = ("train", "val", "test")
_CASE_ID_RE = re.compile(r"^(?:CXR)?0*(\d+)$", re.IGNORECASE)


def canonical_openi_case_id(value: object) -> str:
    """Return the local OpenI identifier used throughout this repository."""

    text = str(value).strip()
    match = _CASE_ID_RE.fullmatch(text)
    if not match:
        raise ValueError(f"Unsupported IU-Xray/OpenI report identifier: {value!r}")
    return f"CXR{int(match.group(1))}"


def _clean_strings(values: Sequence[object]) -> tuple[str, ...]:
    return tuple(text for value in values if (text := " ".join(str(value).split())))


@dataclass(frozen=True)
class RadReStructQuestion:
    question: str
    answers: tuple[str, ...]
    history: tuple[object, ...]
    answer_type: str
    options: tuple[str, ...]
    path: str

    @classmethod
    def from_json(cls, row: object) -> "RadReStructQuestion":
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("Each Rad-ReStruct QA row must contain four elements")
        question, answers, history, metadata = row
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Rad-ReStruct question text must be non-empty")
        if not isinstance(answers, list) or not isinstance(history, list):
            raise ValueError("Rad-ReStruct answers and history must be lists")
        if not isinstance(metadata, Mapping):
            raise ValueError("Rad-ReStruct question metadata must be an object")
        options = metadata.get("options", [])
        if not isinstance(options, list):
            raise ValueError("Rad-ReStruct answer options must be a list")
        return cls(
            question=" ".join(question.split()),
            answers=_clean_strings(answers),
            history=tuple(history),
            answer_type=str(metadata.get("answer_type", "")).strip(),
            options=_clean_strings(options),
            path=str(metadata.get("path", "")).strip(),
        )


@dataclass(frozen=True)
class RadReStructCase:
    source_report_id: str
    case_id: str
    official_split: str
    image_ids: tuple[str, ...]
    questions: tuple[RadReStructQuestion, ...]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_image_mapping(root: Path) -> dict[str, tuple[str, ...]]:
    raw = _load_json(root / "id_to_img_mapping_frontal_reports.json")
    if not isinstance(raw, dict):
        raise ValueError("Rad-ReStruct image mapping must be a JSON object")
    mapping: dict[str, tuple[str, ...]] = {}
    for report_id, image_ids in raw.items():
        if not isinstance(image_ids, list):
            raise ValueError(f"Image mapping for report {report_id!r} must be a list")
        mapping[str(report_id)] = _clean_strings(image_ids)
    return mapping


def iter_radrestruct_cases(dataset_root: str | Path) -> Iterator[RadReStructCase]:
    """Yield official Rad-ReStruct cases without changing the authors' splits."""

    root = Path(dataset_root)
    image_mapping = _load_image_mapping(root)
    observed: set[str] = set()
    for split in RADRESTRUCT_SPLITS:
        report_ids = _load_json(root / f"{split}_ids.json")
        if not isinstance(report_ids, list):
            raise ValueError(f"{split}_ids.json must contain a list")
        for raw_report_id in report_ids:
            report_id = str(raw_report_id)
            case_id = canonical_openi_case_id(report_id)
            if case_id in observed:
                raise ValueError(f"Duplicate Rad-ReStruct report across splits: {case_id}")
            observed.add(case_id)
            qa_path = root / f"{split}_qa_pairs" / f"{report_id}.json"
            questions_raw = _load_json(qa_path)
            if not isinstance(questions_raw, list):
                raise ValueError(f"QA file must contain a list: {qa_path}")
            yield RadReStructCase(
                source_report_id=report_id,
                case_id=case_id,
                official_split=split,
                image_ids=image_mapping.get(report_id, ()),
                questions=tuple(
                    RadReStructQuestion.from_json(row) for row in questions_raw
                ),
            )
