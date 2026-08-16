from __future__ import annotations

import csv

from medical_rag.evaluation.benchmark_v2_validity import audit_cohort, audit_human_evaluation
from medical_rag.evaluation.case_scoped_benchmark import build_case_chunks, build_case_questions


def sample_benchmark() -> dict:
    case = {
        "case_id": "CXR1",
        "indication": "Cough.",
        "comparison": "None.",
        "findings": "Focal opacity. No pleural effusion.",
        "impression": "Focal pulmonary opacity.",
        "problems": "Opacity",
        "images": [{"filename": "1.png", "projection": "Frontal"}],
    }
    chunks = build_case_chunks(case)
    questions = build_case_questions(case, chunks)
    return {
        "questions": questions,
        "chunks": chunks,
        "split": {
            "test": {
                "case_count": 1,
                "qids": [row["qid"] for row in questions],
                "chunk_ids": [row["chunk_id"] for row in chunks],
            }
        },
    }


def test_audit_exposes_gold_route_and_extractive_advantage() -> None:
    benchmark = sample_benchmark()
    prompts = []
    retrieval = []
    generations = []
    for question in benchmark["questions"]:
        relevant = [
            row for row in benchmark["chunks"] if row["chunk_id"] in question["relevant_chunk_ids"]
        ]
        context = "\n".join(row["text"] for row in relevant)
        prompts.append({"qid": question["qid"], "retrieved_context": context})
        retrieval.append(
            {
                "qid": question["qid"],
                "system": "case_scoped_agent_routed_bm25",
                "scores": [0.0] * len(relevant),
            }
        )
        generations.append(
            {
                "qid": question["qid"],
                "question_type": question["question_type"],
                "final_token_f1": 0.5,
            }
        )
    audit = audit_cohort(benchmark, "test", prompts, retrieval, generations)
    assert audit["routed_candidate_pool_equals_qrels_rate"] == 1.0
    assert audit["routed_all_zero_score_query_rate"] == 1.0
    assert audit["extractive_retrieved_context_token_f1"] == 1.0
    assert audit["qwen_minus_extractive_token_f1"] == -0.5


def test_human_audit_requires_all_ratings(tmp_path) -> None:
    path = tmp_path / "ratings.csv"
    columns = [
        "sample_id",
        *[
            f"{letter}_{metric}"
            for letter in "abcd"
            for metric in (
                "correctness_1_5",
                "evidence_grounding_1_5",
                "potentially_harmful_0_1",
            )
        ],
        "best_response_A_B_C_D_or_tie",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerow({column: "1" for column in columns})
        writer.writerow({"sample_id": "incomplete"})
    audit = audit_human_evaluation(path)
    assert audit["completed_rows"] == 1
    assert audit["completion_rate"] == 0.5
