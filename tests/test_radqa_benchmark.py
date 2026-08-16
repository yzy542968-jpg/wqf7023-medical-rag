from __future__ import annotations

import copy

import pytest

from medical_rag.evaluation.radqa_benchmark import (
    benchmark_summary,
    build_radqa_benchmark,
    infer_section,
    normalize_radqa_split,
    report_id_from_document_id,
    sentence_spans,
)


def split_payload(patient: str, report: str) -> dict:
    context = "FINDINGS: The heart is enlarged. There is no pleural effusion."
    answer = "heart is enlarged"
    return {
        "data": [
            {
                "title": patient,
                "paragraphs": [
                    {
                        "document_id": f"{report}_O",
                        "context": context,
                        "qas": [
                            {
                                "id": f"{report}_answerable",
                                "question": "Did the cardiac silhouette enlarge?",
                                "answers": [
                                    {"text": answer, "answer_start": context.index(answer)}
                                ],
                                "is_impossible": False,
                            },
                            {
                                "id": f"{report}_impossible",
                                "question": "Is a pneumothorax present?",
                                "answers": [],
                                "is_impossible": True,
                            },
                        ],
                    },
                    {
                        "document_id": f"{report}_I",
                        "context": "IMPRESSION: Cardiomegaly. No acute pulmonary disease.",
                        "qas": [],
                    },
                ],
            }
        ],
        "version": "1.0",
    }


def test_radqa_normalization_maps_answer_spans_and_hard_negatives() -> None:
    normalized = normalize_radqa_split(split_payload("P1", "R1"), "train")
    answerable = next(row for row in normalized["questions"] if row["is_answerable"])
    impossible = next(row for row in normalized["questions"] if not row["is_answerable"])
    assert len(answerable["relevant_chunk_ids"]) == 1
    assert len(answerable["paragraph_chunk_ids"]) == 2
    assert set(answerable["relevant_chunk_ids"]) < set(answerable["paragraph_chunk_ids"])
    assert impossible["relevant_chunk_ids"] == []
    assert infer_section("123_I") == "impression"
    assert infer_section("123_O") == "findings"
    assert report_id_from_document_id("123_O") == "123"
    assert len(sentence_spans("One sentence. Another sentence.")) == 2


def test_radqa_builder_enforces_patient_disjoint_splits() -> None:
    payload = build_radqa_benchmark(
        {
            "train": split_payload("P1", "R1"),
            "dev": split_payload("P2", "R2"),
            "test": split_payload("P3", "R3"),
        }
    )
    summary = benchmark_summary(payload)
    assert summary["question_count"] == 6
    assert summary["answerable_count"] == 3
    assert summary["unanswerable_count"] == 3
    assert summary["candidate_pool_equals_qrels_rate"] == 0.0
    overlapping = copy.deepcopy(split_payload("P1", "R4"))
    with pytest.raises(ValueError, match="patient overlap"):
        build_radqa_benchmark(
            {
                "train": split_payload("P1", "R1"),
                "dev": overlapping,
                "test": split_payload("P3", "R3"),
            }
        )
