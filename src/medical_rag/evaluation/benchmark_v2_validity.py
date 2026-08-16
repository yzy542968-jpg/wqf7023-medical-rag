from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from medical_rag.evaluation.answer_metrics import token_f1
from medical_rag.evaluation.case_scoped_benchmark import expected_section


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _mean_by_type(rows: Iterable[dict[str, Any]], value_key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_type"])].append(float(row[value_key]))
    return {name: mean(values) for name, values in sorted(grouped.items())}


def audit_cohort(
    benchmark: dict[str, Any],
    split_name: str,
    prompt_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    generation_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    split_qids = set(benchmark["split"][split_name]["qids"])
    questions = [row for row in benchmark["questions"] if row["qid"] in split_qids]
    chunks = [
        row
        for row in benchmark["chunks"]
        if row["chunk_id"] in set(benchmark["split"][split_name]["chunk_ids"])
    ]
    candidates: dict[tuple[str, str], set[str]] = defaultdict(set)
    for chunk in chunks:
        candidates[(str(chunk["case_id"]), str(chunk["section"]))].add(
            str(chunk["chunk_id"])
        )

    candidate_equals_qrels = []
    for question in questions:
        key = (str(question["case_id"]), expected_section(str(question["question_type"])))
        candidate_equals_qrels.append(
            candidates[key] == set(question["relevant_chunk_ids"])
        )

    routed_rows = [
        row
        for row in retrieval_rows
        if row.get("system") == "case_scoped_agent_routed_bm25"
        and row.get("qid") in split_qids
    ]
    score_lists = [[float(score) for score in row.get("scores", [])] for row in routed_rows]
    flat_scores = [score for scores in score_lists for score in scores]

    prompt_by_qid = {str(row["qid"]): row for row in prompt_rows}
    generation_by_qid = {str(row["qid"]): row for row in generation_rows}
    baseline_rows = []
    for question in questions:
        qid = str(question["qid"])
        prompt = prompt_by_qid[qid]
        generation = generation_by_qid[qid]
        baseline_rows.append(
            {
                "qid": qid,
                "question_type": question["question_type"],
                "extractive_token_f1": token_f1(
                    str(prompt["retrieved_context"]), str(question["reference_answer"])
                ),
                "qwen_token_f1": float(generation["final_token_f1"]),
            }
        )

    extractive_f1 = mean(row["extractive_token_f1"] for row in baseline_rows)
    qwen_f1 = mean(row["qwen_token_f1"] for row in baseline_rows)
    return {
        "case_count": benchmark["split"][split_name]["case_count"],
        "question_count": len(questions),
        "unique_question_string_count": len({row["question"] for row in questions}),
        "scope_case_matches_target_rate": mean(
            row["scope_case_id"] == row["case_id"] for row in questions
        ),
        "routed_candidate_pool_equals_qrels_rate": mean(candidate_equals_qrels),
        "routed_all_zero_score_query_rate": mean(
            bool(scores) and all(abs(score) < 1e-12 for score in scores)
            for scores in score_lists
        ),
        "routed_zero_score_result_rate": (
            mean(abs(score) < 1e-12 for score in flat_scores) if flat_scores else 0.0
        ),
        "routed_mean_top1_score": mean(scores[0] for scores in score_lists if scores),
        "extractive_retrieved_context_token_f1": extractive_f1,
        "qwen_verified_token_f1": qwen_f1,
        "qwen_minus_extractive_token_f1": qwen_f1 - extractive_f1,
        "extractive_token_f1_by_question_type": _mean_by_type(
            baseline_rows, "extractive_token_f1"
        ),
        "qwen_token_f1_by_question_type": _mean_by_type(
            baseline_rows, "qwen_token_f1"
        ),
    }


def audit_human_evaluation(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rating_columns = [
        f"{letter}_{metric}"
        for letter in "abcd"
        for metric in (
            "correctness_1_5",
            "evidence_grounding_1_5",
            "potentially_harmful_0_1",
        )
    ]
    required = [*rating_columns, "best_response_A_B_C_D_or_tie"]
    completed = sum(
        all(str(row.get(column, "")).strip() for column in required) for row in rows
    )
    filled = sum(
        bool(str(row.get(column, "")).strip()) for row in rows for column in required
    )
    return {
        "rows": len(rows),
        "completed_rows": completed,
        "completion_rate": completed / len(rows) if rows else 0.0,
        "filled_required_cells": filled,
        "required_cells": len(rows) * len(required),
    }

