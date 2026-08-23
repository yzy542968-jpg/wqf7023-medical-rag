from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence


RATING_FIELDS = (
    "retrieval_similarity_1_5",
    "target_answer_consistency_1_5",
    "historical_usefulness_1_5",
    "potential_harm_0_2",
    "preference_rank",
    "reviewer_note",
)


def _digest(domain: str, seed: int, value: str) -> str:
    return hashlib.sha256(f"{domain}|{seed}|{value}".encode("utf-8")).hexdigest()


def build_blinded_review_rows(
    cases: Sequence[Mapping[str, Any]],
    *,
    system_names: Sequence[str],
    case_count: int = 100,
    seed: int = 7047,
) -> list[dict[str, Any]]:
    if len(system_names) < 2 or len(system_names) != len(set(system_names)):
        raise ValueError("at least two unique systems are required")
    ordered_cases = sorted(
        cases,
        key=lambda row: (
            _digest("v10-clinical-case", seed, str(row["case_id"])),
            str(row["case_id"]),
        ),
    )[:case_count]
    rows = []
    for package_index, case in enumerate(ordered_cases, start=1):
        case_id = str(case["case_id"])
        blinded_systems = sorted(
            system_names,
            key=lambda system: (_digest(f"v10-clinical-system|{case_id}", seed, system), system),
        )
        for system_index, system in enumerate(blinded_systems, start=1):
            answers = case.get("answers") or {}
            retrieval = case.get("retrieval") or {}
            row = {
                "package_case_id": f"V10R{package_index:03d}",
                "presentation_code": chr(64 + system_index),
                "question": str(case.get("question", "")),
                "indication": str(case.get("indication", "")),
                "target_image_reference": str(case.get("target_image_reference", "")),
                "answer": str(answers.get(system, "")),
                "retrieved_evidence": str(retrieval.get(system, "")),
                "system_key_private": system,
                **{field: "" for field in RATING_FIELDS},
            }
            rows.append(row)
    return rows


def validate_completed_review(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("review package is empty")
    for index, row in enumerate(rows, start=1):
        missing = [field for field in RATING_FIELDS if str(row.get(field, "")).strip() == ""]
        if missing:
            raise ValueError(f"review row {index} is incomplete: {', '.join(missing)}")


def public_review_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "system_key_private"}
        for row in rows
    ]


__all__ = [
    "RATING_FIELDS",
    "build_blinded_review_rows",
    "public_review_rows",
    "validate_completed_review",
]

