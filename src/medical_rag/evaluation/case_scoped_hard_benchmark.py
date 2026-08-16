from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Iterable

from medical_rag.evaluation.case_scoped_benchmark import (
    build_case_chunks,
    clean_text,
    content_fingerprint,
    is_clean_eligible_case,
)


FINDINGS_QUESTIONS = (
    "Describe the observations visible on this examination.",
    "What abnormalities are documented on the images?",
    "What did the radiographic examination show?",
    "Which imaging observations were recorded in this report?",
)

IMPRESSION_QUESTIONS = (
    "What conclusion did the radiologist reach?",
    "Give the final assessment from this report.",
    "Please summarize this report.",
    "What is the diagnostic bottom line of the examination?",
)

UNANSWERABLE_QUESTIONS = (
    "What was the patient's serum troponin concentration?",
    "Which medication was administered after the examination?",
    "What did the pathology specimen show?",
    "What treatment was prescribed at discharge?",
    "What was the measured CT attenuation in Hounsfield units?",
    "What was the patient's oxygen saturation after treatment?",
)

NEAR_DOMAIN_CONDITIONS = (
    "pneumothorax",
    "pleural effusion",
    "pulmonary edema",
    "cardiomegaly",
    "rib fracture",
    "hiatal hernia",
    "pulmonary nodule",
    "mediastinal widening",
)

FACT_STOPWORDS = {
    "chest",
    "disease",
    "finding",
    "findings",
    "left",
    "normal",
    "present",
    "right",
    "there",
    "unchanged",
    "without",
}


def _stable_index(case_id: str, namespace: str, size: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % size


def _partition(case_ids: list[str], seed: int) -> dict[str, list[str]]:
    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)
    development_end = round(len(shuffled) * 0.50)
    calibration_end = development_end + round(len(shuffled) * 0.20)
    return {
        "development": sorted(shuffled[:development_end]),
        "calibration": sorted(shuffled[development_end:calibration_end]),
        "test": sorted(shuffled[calibration_end:]),
    }


def _fact_probe(
    case: dict[str, Any], chunks: list[dict[str, Any]]
) -> tuple[str, str, list[str]]:
    text_by_section = {
        section: clean_text(case.get(section, ""))
        for section in ("findings", "impression")
    }
    source = " ".join(
        [clean_text(case.get("problems", "")), text_by_section["impression"], text_by_section["findings"]]
    )
    candidates = [
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z-]{4,}", source)
        if token.lower() not in FACT_STOPWORDS
    ]
    for token in candidates:
        relevant = [
            str(chunk["chunk_id"])
            for chunk in chunks
            if chunk["section"] in {"findings", "impression"}
            and token in str(chunk["text"]).lower()
        ]
        if relevant:
            first = next(chunk for chunk in chunks if chunk["chunk_id"] == relevant[0])
            return token, str(first["section"]), relevant
    findings_ids = [
        str(chunk["chunk_id"]) for chunk in chunks if chunk["section"] == "findings"
    ]
    return "the principal abnormality", "findings", findings_ids


def _absent_condition(case: dict[str, Any]) -> str:
    report = " ".join(
        clean_text(case.get(section, "")).lower()
        for section in ("indication", "comparison", "findings", "impression", "problems")
    )
    start = _stable_index(str(case["case_id"]), "absent-condition", len(NEAR_DOMAIN_CONDITIONS))
    for offset in range(len(NEAR_DOMAIN_CONDITIONS)):
        condition = NEAR_DOMAIN_CONDITIONS[(start + offset) % len(NEAR_DOMAIN_CONDITIONS)]
        if condition not in report:
            return condition
    return "measured pulmonary artery pressure"


def build_hard_questions(
    case: dict[str, Any], chunks: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_section: dict[str, list[str]] = {}
    for chunk in chunks:
        by_section.setdefault(str(chunk["section"]), []).append(str(chunk["chunk_id"]))

    case_id = str(case["case_id"])
    first_negative = _stable_index(case_id, "negative-a", len(UNANSWERABLE_QUESTIONS))
    second_negative = _stable_index(case_id, "negative-b", len(UNANSWERABLE_QUESTIONS))
    if second_negative == first_negative:
        second_negative = (second_negative + 1) % len(UNANSWERABLE_QUESTIONS)
    fact_term, fact_section, fact_chunk_ids = _fact_probe(case, chunks)
    absent_condition = _absent_condition(case)

    common = {
        "case_id": case_id,
        "scope_case_id": case_id,
        "images": case.get("images", []),
    }
    return [
        {
            **common,
            "qid": f"{case_id}_v21_observation",
            "question": FINDINGS_QUESTIONS[
                _stable_index(case_id, "findings", len(FINDINGS_QUESTIONS))
            ],
            "is_answerable": True,
            "expected_intent": "findings",
            "reference_answer": clean_text(case.get("findings", "")),
            "relevant_chunk_ids": by_section.get("findings", []),
        },
        {
            **common,
            "qid": f"{case_id}_v21_conclusion",
            "question": IMPRESSION_QUESTIONS[
                _stable_index(case_id, "impression", len(IMPRESSION_QUESTIONS))
            ],
            "is_answerable": True,
            "expected_intent": "impression",
            "reference_answer": clean_text(case.get("impression", "")),
            "relevant_chunk_ids": by_section.get("impression", []),
        },
        {
            **common,
            "qid": f"{case_id}_v21_fact_probe",
            "question": f"What does this report state about {fact_term}?",
            "is_answerable": True,
            "expected_intent": fact_section,
            "reference_answer": " ".join(
                str(chunk["text"]) for chunk in chunks if chunk["chunk_id"] in fact_chunk_ids
            ),
            "relevant_chunk_ids": fact_chunk_ids,
        },
        {
            **common,
            "qid": f"{case_id}_v21_unanswerable_a",
            "question": UNANSWERABLE_QUESTIONS[first_negative],
            "is_answerable": False,
            "expected_intent": "unavailable",
            "reference_answer": "NOT ANSWERABLE",
            "relevant_chunk_ids": [],
        },
        {
            **common,
            "qid": f"{case_id}_v21_unanswerable_b",
            "question": UNANSWERABLE_QUESTIONS[second_negative],
            "is_answerable": False,
            "expected_intent": "unavailable",
            "reference_answer": "NOT ANSWERABLE",
            "relevant_chunk_ids": [],
        },
        {
            **common,
            "qid": f"{case_id}_v21_near_domain_negative",
            "question": f"What does this report state about {absent_condition}?",
            "is_answerable": False,
            "expected_intent": "unavailable",
            "reference_answer": "NOT ANSWERABLE",
            "relevant_chunk_ids": [],
        },
    ]


def build_case_scoped_hard_benchmark(
    cases: Iterable[dict[str, Any]],
    excluded_case_ids: set[str],
    max_cases: int = 240,
    seed: int = 27023,
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

    chunks: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    for case in selected:
        case_chunks = build_case_chunks(case)
        chunks.extend(case_chunks)
        questions.extend(build_hard_questions(case, case_chunks))

    partitions = _partition([str(case["case_id"]) for case in selected], seed + 1)
    split_by_case = {
        case_id: name for name, case_ids in partitions.items() for case_id in case_ids
    }
    for question in questions:
        question["split"] = split_by_case[question["case_id"]]

    split: dict[str, Any] = {}
    for name, case_ids in partitions.items():
        case_set = set(case_ids)
        split[name] = {
            "case_count": len(case_ids),
            "question_count": sum(row["case_id"] in case_set for row in questions),
            "case_ids": case_ids,
            "qids": [row["qid"] for row in questions if row["case_id"] in case_set],
        }

    case_fingerprint = "\n".join(sorted(str(case["case_id"]) for case in selected))
    return {
        "benchmark": "OpenI case-scoped hard evidence QA",
        "version": "2.1",
        "seed": seed,
        "design": {
            "answerable_questions_per_case": 3,
            "unanswerable_questions_per_case": 3,
            "same_case_distractors": ["indication", "comparison", "non-target report section"],
            "agent_input_fields": ["scope_case_id", "question"],
            "gold_route_not_exposed_to_agent": True,
        },
        "excluded_prior_case_count": len(excluded_case_ids),
        "case_count": len(selected),
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "answerable_count": sum(bool(row["is_answerable"]) for row in questions),
        "unanswerable_count": sum(not bool(row["is_answerable"]) for row in questions),
        "case_id_fingerprint_sha256": hashlib.sha256(
            case_fingerprint.encode("utf-8")
        ).hexdigest(),
        "content_fingerprint_sha256": content_fingerprint(questions, chunks),
        "questions": questions,
        "chunks": chunks,
        "split": split,
    }


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)
