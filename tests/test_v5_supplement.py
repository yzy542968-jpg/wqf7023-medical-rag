from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from scripts.build_v5_reproducibility_supplement import DEPENDENCIES


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_v5_confirmation_split_is_case_disjoint_and_has_three_questions_per_case() -> None:
    cohort = read_json(ROOT / "data/processed/openi_multimodal_v5_cohort.json")
    development = set(cohort["split"]["development"]["case_ids"])
    confirmation = set(cohort["split"]["confirmation"]["case_ids"])
    assert len(development) == 120
    assert len(confirmation) == 120
    assert development.isdisjoint(confirmation)
    assert len(cohort["case_ids"]) == 240
    assert len(cohort["questions"]) == 720

    confirmation_questions = [
        row for row in cohort["questions"] if row["case_id"] in confirmation
    ]
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in confirmation_questions:
        by_case[str(row["case_id"])].append(row)
    assert len(confirmation_questions) == 360
    assert {len(rows) for rows in by_case.values()} == {3}
    assert Counter(row["question_type"] for row in confirmation_questions) == Counter(
        {
            "case_scoped_findings": 120,
            "case_scoped_impression": 120,
            "case_scoped_summary": 120,
        }
    )


def test_v5_reference_duplication_and_patient_identifier_boundary_are_explicit() -> None:
    cohort = read_json(ROOT / "data/processed/openi_multimodal_v5_cohort.json")
    confirmation = set(cohort["split"]["confirmation"]["case_ids"])
    source_path = ROOT / "data/processed/openi_cases.jsonl"
    if not source_path.is_file():
        pytest.skip("requires local OpenI source artifact excluded from Git")
    source_cases = read_jsonl(source_path)
    source_fields = set(source_cases[0])
    assert not any("patient" in field.lower() or "subject" in field.lower() for field in source_fields)

    by_case: dict[str, dict[str, str]] = defaultdict(dict)
    for row in cohort["questions"]:
        if row["case_id"] in confirmation:
            by_case[str(row["case_id"])][str(row["question_type"])] = str(row["reference_answer"])
    assert len(by_case) == 120
    assert all(
        values["case_scoped_impression"] == values["case_scoped_summary"]
        for values in by_case.values()
    )


def test_v5_supplemental_dependency_closure_exists() -> None:
    source_path = ROOT / "data/processed/openi_cases.jsonl"
    if not source_path.is_file():
        pytest.skip("requires local OpenI source artifact excluded from Git")
    missing = [relative for relative in DEPENDENCIES if not (ROOT / relative).is_file()]
    assert not missing, f"Missing V5 dependency closure entries: {missing}"
