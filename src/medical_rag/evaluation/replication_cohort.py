from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Iterable

from medical_rag.evaluation.case_scoped_benchmark import clean_text, is_clean_eligible_case


def _problem_hint(case: dict[str, Any]) -> str:
    problems = clean_text(case.get("problems", ""))
    return problems.replace(";", ", ") if problems else "the main radiology finding"


def build_replication_questions(case: dict[str, Any]) -> list[dict[str, Any]]:
    case_id = str(case["case_id"])
    indication = clean_text(case.get("indication", "")) or "not provided"
    findings = clean_text(case.get("findings", ""))
    impression = clean_text(case.get("impression", ""))
    return [
        {
            "qid": f"{case_id}_rep_impression",
            "case_id": case_id,
            "question_type": "impression_from_indication",
            "question": f"For a chest X-ray case with the indication '{indication}', what is the radiology impression?",
            "reference_answer": impression,
            "relevant_case_ids": [case_id],
        },
        {
            "qid": f"{case_id}_rep_findings",
            "case_id": case_id,
            "question_type": "findings_from_indication",
            "question": f"For a chest X-ray case with the indication '{indication}', what are the main report findings?",
            "reference_answer": findings,
            "relevant_case_ids": [case_id],
        },
        {
            "qid": f"{case_id}_rep_summary",
            "case_id": case_id,
            "question_type": "abnormality_summary",
            "question": f"What does the chest X-ray report say about {_problem_hint(case)}?",
            "reference_answer": impression or findings,
            "relevant_case_ids": [case_id],
        },
    ]


def build_replication_cohort(
    cases: Iterable[dict[str, Any]],
    excluded_case_ids: set[str],
    max_cases: int = 300,
    seed: int = 47023,
) -> dict[str, Any]:
    eligible = [
        case
        for case in cases
        if str(case["case_id"]) not in excluded_case_ids and is_clean_eligible_case(case)
    ]
    random.Random(seed).shuffle(eligible)
    selected = sorted(eligible[:max_cases], key=lambda row: str(row["case_id"]))
    if len(selected) < max_cases:
        raise ValueError(f"Requested {max_cases} cases but only {len(selected)} are eligible.")
    questions = [question for case in selected for question in build_replication_questions(case)]
    canonical = json.dumps(questions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    case_ids = [str(case["case_id"]) for case in selected]
    return {
        "cohort": "OpenI locked-system untouched replication",
        "version": "1.0",
        "seed": seed,
        "case_count": len(case_ids),
        "question_count": len(questions),
        "excluded_prior_case_count": len(excluded_case_ids),
        "case_ids": case_ids,
        "case_id_fingerprint_sha256": hashlib.sha256(
            "\n".join(case_ids).encode("utf-8")
        ).hexdigest(),
        "content_fingerprint_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "questions": questions,
    }
