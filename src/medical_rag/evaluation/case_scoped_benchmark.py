from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from typing import Any, Iterable


SECTION_BY_QUESTION_TYPE = {
    "case_scoped_findings": "findings",
    "case_scoped_impression": "impression",
    "case_scoped_summary": "impression",
}


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def placeholder_ratio(text: str) -> float:
    tokens = clean_text(text).split()
    if not tokens:
        return 1.0
    return sum("XXXX" in token.upper() for token in tokens) / len(tokens)


def is_clean_eligible_case(case: dict[str, Any]) -> bool:
    return (
        bool(case.get("images"))
        and len(clean_text(case.get("findings", ""))) >= 40
        and len(clean_text(case.get("impression", ""))) >= 8
        and placeholder_ratio(case.get("indication", "")) <= 0.5
        and clean_text(case.get("problems", "")).lower() not in {"", "normal"}
    )


def split_sentences(text: str) -> list[str]:
    cleaned = clean_text(text)
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", cleaned)
    return [piece.strip() for piece in pieces if piece.strip()]


def build_case_chunks(case: dict[str, Any]) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for section in ("indication", "comparison", "findings", "impression"):
        for index, sentence in enumerate(split_sentences(case.get(section, "")), start=1):
            chunks.append(
                {
                    "chunk_id": f"{case['case_id']}::{section}::{index:03d}",
                    "case_id": case["case_id"],
                    "section": section,
                    "position": index,
                    "text": sentence,
                }
            )
    return chunks


def build_case_questions(case: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_section: dict[str, list[str]] = {}
    for chunk in chunks:
        by_section.setdefault(chunk["section"], []).append(chunk["chunk_id"])

    common = {
        "case_id": case["case_id"],
        "scope_case_id": case["case_id"],
        "indication": clean_text(case.get("indication", "")),
        "problems": clean_text(case.get("problems", "")),
        "images": case.get("images", []),
    }
    return [
        {
            **common,
            "qid": f"{case['case_id']}_v2_findings",
            "question_type": "case_scoped_findings",
            "question": "What radiographic findings are documented for this examination?",
            "reference_answer": clean_text(case.get("findings", "")),
            "answer_source": "findings",
            "relevant_chunk_ids": by_section.get("findings", []),
        },
        {
            **common,
            "qid": f"{case['case_id']}_v2_impression",
            "question_type": "case_scoped_impression",
            "question": "What is the final radiology impression for this examination?",
            "reference_answer": clean_text(case.get("impression", "")),
            "answer_source": "impression",
            "relevant_chunk_ids": by_section.get("impression", []),
        },
        {
            **common,
            "qid": f"{case['case_id']}_v2_summary",
            "question_type": "case_scoped_summary",
            "question": "Summarize the principal abnormality or conclusion in this report.",
            "reference_answer": clean_text(case.get("impression", "")),
            "answer_source": "impression",
            "relevant_chunk_ids": by_section.get("impression", []),
        },
    ]


def _case_partition(case_ids: list[str], seed: int) -> dict[str, list[str]]:
    shuffled = list(case_ids)
    random.Random(seed).shuffle(shuffled)
    development_end = round(len(shuffled) * 0.60)
    calibration_end = development_end + round(len(shuffled) * 0.20)
    return {
        "development": sorted(shuffled[:development_end]),
        "calibration": sorted(shuffled[development_end:calibration_end]),
        "test": sorted(shuffled[calibration_end:]),
    }


def content_fingerprint(
    questions: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> str:
    """Hash the benchmark content, not only the selected case identifiers."""
    payload = {
        "questions": questions,
        "chunks": chunks,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_case_scoped_benchmark(
    cases: Iterable[dict[str, Any]],
    excluded_case_ids: set[str],
    max_cases: int = 600,
    seed: int = 7023,
) -> dict[str, Any]:
    eligible = [
        case
        for case in cases
        if case["case_id"] not in excluded_case_ids and is_clean_eligible_case(case)
    ]
    random.Random(seed).shuffle(eligible)
    selected = sorted(eligible[:max_cases], key=lambda item: item["case_id"])
    if len(selected) < max_cases:
        raise ValueError(f"Requested {max_cases} cases but only {len(selected)} are eligible.")

    chunks: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    for case in selected:
        case_chunks = build_case_chunks(case)
        chunks.extend(case_chunks)
        questions.extend(build_case_questions(case, case_chunks))

    partitions = _case_partition([case["case_id"] for case in selected], seed + 1)
    split: dict[str, Any] = {}
    for name, case_ids in partitions.items():
        selected_ids = set(case_ids)
        qids = [row["qid"] for row in questions if row["case_id"] in selected_ids]
        chunk_ids = [row["chunk_id"] for row in chunks if row["case_id"] in selected_ids]
        split[name] = {
            "case_count": len(case_ids),
            "question_count": len(qids),
            "chunk_count": len(chunk_ids),
            "case_ids": case_ids,
            "qids": qids,
            "chunk_ids": chunk_ids,
        }

    fingerprint_source = "\n".join(sorted(case["case_id"] for case in selected))
    return {
        "benchmark": "OpenI case-scoped evidence QA v2",
        "version": "2.0",
        "seed": seed,
        "excluded_prior_case_count": len(excluded_case_ids),
        "case_count": len(selected),
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "case_id_fingerprint_sha256": hashlib.sha256(fingerprint_source.encode("utf-8")).hexdigest(),
        "content_fingerprint_sha256": content_fingerprint(questions, chunks),
        "questions": questions,
        "chunks": chunks,
        "split": split,
    }


def expected_section(question_type: str) -> str:
    try:
        return SECTION_BY_QUESTION_TYPE[question_type]
    except KeyError as exc:
        raise ValueError(f"Unknown v2 question type: {question_type}") from exc


def benchmark_summary(payload: dict[str, Any]) -> dict[str, Any]:
    questions = payload["questions"]
    query_targets: dict[str, set[str]] = {}
    for row in questions:
        query_targets.setdefault(row["question"], set()).add(row["case_id"])
    globally_ambiguous = sum(
        1 for row in questions if len(query_targets[row["question"]]) > 1
    )
    scoped_keys = {(row["scope_case_id"], row["question"]) for row in questions}
    return {
        "case_count": payload["case_count"],
        "question_count": payload["question_count"],
        "chunk_count": payload["chunk_count"],
        "global_duplicate_query_rate": globally_ambiguous / len(questions),
        "case_scoped_unique_query_rate": len(scoped_keys) / len(questions),
        "split_counts": {
            name: {
                "cases": part["case_count"],
                "questions": part["question_count"],
                "chunks": part["chunk_count"],
            }
            for name, part in payload["split"].items()
        },
        "fingerprint": payload["case_id_fingerprint_sha256"],
        "content_fingerprint": payload.get("content_fingerprint_sha256"),
    }


def dumps_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)
