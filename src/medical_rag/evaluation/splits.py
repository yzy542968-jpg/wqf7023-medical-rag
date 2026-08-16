from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from typing import Any


def _question_fingerprint(questions: list[dict[str, Any]]) -> str:
    canonical = [
        {
            "qid": str(item["qid"]),
            "case_id": str(item["case_id"]),
            "question_type": str(item.get("question_type", "")),
        }
        for item in sorted(questions, key=lambda value: str(value["qid"]))
    ]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _split_summary(questions: list[dict[str, Any]], case_ids: set[str]) -> dict[str, Any]:
    selected = [item for item in questions if str(item["case_id"]) in case_ids]
    type_counts = Counter(str(item.get("question_type", "unknown")) for item in selected)
    return {
        "case_count": len(case_ids),
        "question_count": len(selected),
        "case_ids": sorted(case_ids),
        "qids": sorted(str(item["qid"]) for item in selected),
        "question_type_counts": dict(sorted(type_counts.items())),
    }


def build_grouped_case_split(
    questions: list[dict[str, Any]],
    *,
    development_fraction: float = 0.70,
    seed: int = 7023,
) -> dict[str, Any]:
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must be between 0 and 1")
    if not questions:
        raise ValueError("questions must not be empty")

    qids = [str(item["qid"]) for item in questions]
    if len(qids) != len(set(qids)):
        raise ValueError("question IDs must be unique")

    case_ids = sorted({str(item["case_id"]) for item in questions})
    if len(case_ids) < 2:
        raise ValueError("at least two cases are required for a grouped split")

    shuffled_case_ids = case_ids.copy()
    random.Random(seed).shuffle(shuffled_case_ids)
    development_count = max(1, min(len(case_ids) - 1, round(len(case_ids) * development_fraction)))
    development_cases = set(shuffled_case_ids[:development_count])
    test_cases = set(shuffled_case_ids[development_count:])

    if development_cases.intersection(test_cases):
        raise AssertionError("case leakage detected between development and test splits")

    development = _split_summary(questions, development_cases)
    test = _split_summary(questions, test_cases)
    if set(development["qids"]).intersection(test["qids"]):
        raise AssertionError("question leakage detected between development and test splits")
    if len(development["qids"]) + len(test["qids"]) != len(questions):
        raise AssertionError("split does not cover every question")

    return {
        "split_type": "grouped_by_case_id",
        "seed": seed,
        "development_fraction": development_fraction,
        "dataset_fingerprint_sha256": _question_fingerprint(questions),
        "total_case_count": len(case_ids),
        "total_question_count": len(questions),
        "development": development,
        "test": test,
    }


def filter_questions_for_split(
    questions: list[dict[str, Any]], split: dict[str, Any], split_name: str
) -> list[dict[str, Any]]:
    if split_name not in {"development", "test"}:
        raise ValueError("split_name must be development or test")
    selected_qids = set(split[split_name]["qids"])
    return [item for item in questions if str(item["qid"]) in selected_qids]
