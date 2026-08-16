from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def clean_text(text: str) -> str:
    return " ".join((text or "").split())


def infer_section(document_id: str) -> str:
    suffix = document_id.rsplit("_", 1)[-1].upper()
    if suffix == "I":
        return "impression"
    if suffix in {"F", "O"}:
        return "findings"
    return "report"


def report_id_from_document_id(document_id: str) -> str:
    return re.sub(r"_(?:I|F|O)$", "", document_id, flags=re.IGNORECASE)


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    if not text:
        return []
    boundaries = list(re.finditer(r"(?<=[.!?])\s+(?=[A-Z0-9])", text))
    raw_spans: list[tuple[int, int]] = []
    start = 0
    for boundary in boundaries:
        raw_spans.append((start, boundary.start()))
        start = boundary.end()
    raw_spans.append((start, len(text)))

    spans: list[tuple[int, int, str]] = []
    for raw_start, raw_end in raw_spans:
        segment = text[raw_start:raw_end]
        left = len(segment) - len(segment.lstrip())
        right = len(segment.rstrip())
        span_start = raw_start + left
        span_end = raw_start + right
        if span_start < span_end:
            spans.append((span_start, span_end, clean_text(text[span_start:span_end])))
    return spans


def answer_overlaps_chunk(answer_start: int, answer_text: str, start: int, end: int) -> bool:
    answer_end = answer_start + len(answer_text)
    return answer_start < end and answer_end > start


def content_fingerprint(questions: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        {"questions": questions, "chunks": chunks},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _iter_articles(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("RadQA file must contain a SQuAD-style 'data' list.")
    return data


def normalize_radqa_split(payload: dict[str, Any], split: str) -> dict[str, Any]:
    questions: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for article in _iter_articles(payload):
        patient_id = str(article.get("title", "")).strip()
        if not patient_id:
            raise ValueError(f"RadQA {split} article is missing a patient title.")
        paragraphs = article.get("paragraphs", [])
        for paragraph_index, paragraph in enumerate(paragraphs, start=1):
            context = str(paragraph.get("context", ""))
            document_id = str(
                paragraph.get("document_id") or f"{patient_id}_P{paragraph_index:04d}"
            )
            report_id = report_id_from_document_id(document_id)
            section = infer_section(document_id)
            paragraph_chunks: list[dict[str, Any]] = []
            for position, (start, end, text) in enumerate(sentence_spans(context), start=1):
                chunk_id = f"{split}::{document_id}::{position:03d}"
                if chunk_id in seen_chunk_ids:
                    raise ValueError(f"Duplicate RadQA chunk ID: {chunk_id}")
                seen_chunk_ids.add(chunk_id)
                chunk = {
                    "chunk_id": chunk_id,
                    "case_id": report_id,
                    "report_id": report_id,
                    "patient_id": patient_id,
                    "document_id": document_id,
                    "section": section,
                    "position": position,
                    "char_start": start,
                    "char_end": end,
                    "text": text,
                }
                chunks.append(chunk)
                paragraph_chunks.append(chunk)

            for qa in paragraph.get("qas", []):
                qid = str(qa.get("id", "")).strip()
                if not qid:
                    raise ValueError(f"RadQA {split} question is missing an ID.")
                answers = [
                    {
                        "text": str(answer.get("text", "")),
                        "answer_start": int(answer["answer_start"]),
                    }
                    for answer in qa.get("answers", [])
                    if str(answer.get("text", "")).strip()
                    and answer.get("answer_start") is not None
                ]
                is_impossible = bool(qa.get("is_impossible", False))
                relevant_chunk_ids = sorted(
                    {
                        chunk["chunk_id"]
                        for answer in answers
                        for chunk in paragraph_chunks
                        if answer_overlaps_chunk(
                            answer["answer_start"],
                            answer["text"],
                            chunk["char_start"],
                            chunk["char_end"],
                        )
                    }
                )
                questions.append(
                    {
                        "qid": qid,
                        "split": split,
                        "patient_id": patient_id,
                        "report_id": report_id,
                        "document_id": document_id,
                        "source_section": section,
                        "question": clean_text(str(qa.get("question", ""))),
                        "is_answerable": not is_impossible,
                        "answers": answers,
                        "reference_answers": [clean_text(answer["text"]) for answer in answers],
                        "relevant_chunk_ids": relevant_chunk_ids,
                        "paragraph_chunk_ids": [chunk["chunk_id"] for chunk in paragraph_chunks],
                    }
                )

    validate_normalized_split(questions, chunks, split)
    return {"questions": questions, "chunks": chunks}


def validate_normalized_split(
    questions: list[dict[str, Any]], chunks: list[dict[str, Any]], split: str
) -> None:
    qids = [row["qid"] for row in questions]
    if len(qids) != len(set(qids)):
        duplicates = [qid for qid, count in Counter(qids).items() if count > 1]
        raise ValueError(f"Duplicate RadQA qids in {split}: {duplicates[:3]}")
    chunk_ids = {row["chunk_id"] for row in chunks}
    for row in questions:
        relevant = set(row["relevant_chunk_ids"])
        paragraph = set(row["paragraph_chunk_ids"])
        if not relevant <= chunk_ids or not paragraph <= chunk_ids:
            raise ValueError(f"Question {row['qid']} references unknown chunks.")
        if row["is_answerable"] and not row["answers"]:
            raise ValueError(f"Answerable question {row['qid']} has no answer spans.")
        if row["is_answerable"] and not relevant:
            raise ValueError(f"Answerable question {row['qid']} has no sentence-level qrels.")
        if not row["is_answerable"] and (row["answers"] or relevant):
            raise ValueError(f"Impossible question {row['qid']} contains answer evidence.")


def build_radqa_benchmark(split_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    all_questions: list[dict[str, Any]] = []
    all_chunks: list[dict[str, Any]] = []
    split_manifest: dict[str, Any] = {}
    patient_sets: dict[str, set[str]] = {}
    for split in ("train", "dev", "test"):
        normalized = normalize_radqa_split(split_payloads[split], split)
        questions = normalized["questions"]
        chunks = normalized["chunks"]
        all_questions.extend(questions)
        all_chunks.extend(chunks)
        patients = {row["patient_id"] for row in questions}
        patient_sets[split] = patients
        split_manifest[split] = {
            "patient_count": len(patients),
            "report_count": len({row["report_id"] for row in questions}),
            "question_count": len(questions),
            "answerable_count": sum(row["is_answerable"] for row in questions),
            "unanswerable_count": sum(not row["is_answerable"] for row in questions),
            "chunk_count": len(chunks),
            "qids": [row["qid"] for row in questions],
            "chunk_ids": [row["chunk_id"] for row in chunks],
        }
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        overlap = patient_sets[left] & patient_sets[right]
        if overlap:
            raise ValueError(f"RadQA patient overlap between {left} and {right}: {sorted(overlap)[:3]}")

    return {
        "benchmark": "RadQA natural-question evidence retrieval v3",
        "version": "3.0",
        "source_format": "official SQuAD-style RadQA train/dev/test JSON",
        "task": "report-scoped natural-question evidence retrieval and answerability",
        "questions": all_questions,
        "chunks": all_chunks,
        "split": split_manifest,
        "content_fingerprint_sha256": content_fingerprint(all_questions, all_chunks),
    }


def load_radqa_files(input_dir: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for split in ("train", "dev", "test"):
        path = input_dir / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing credentialed RadQA file: {path}. Obtain it through the official "
                "PhysioNet access process; this project does not download restricted files."
            )
        payloads[split] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def benchmark_summary(payload: dict[str, Any]) -> dict[str, Any]:
    questions = payload["questions"]
    answerable = [row for row in questions if row["is_answerable"]]
    candidate_sizes = [len(row["paragraph_chunk_ids"]) for row in questions]
    candidate_equals_qrels = [
        set(row["paragraph_chunk_ids"]) == set(row["relevant_chunk_ids"])
        for row in answerable
    ]
    return {
        "question_count": len(questions),
        "answerable_count": len(answerable),
        "unanswerable_count": len(questions) - len(answerable),
        "unique_question_count": len({row["question"] for row in questions}),
        "chunk_count": len(payload["chunks"]),
        "mean_report_candidate_chunks": sum(candidate_sizes) / len(candidate_sizes),
        "candidate_pool_equals_qrels_rate": (
            sum(candidate_equals_qrels) / len(candidate_equals_qrels) if candidate_equals_qrels else 0.0
        ),
        "split": payload["split"],
        "content_fingerprint_sha256": payload["content_fingerprint_sha256"],
    }

