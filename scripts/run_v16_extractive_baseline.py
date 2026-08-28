"""Run a leakage-audited extractive historical-evidence diagnostic.

This is not a clinical diagnosis system and is not a replacement for the
generative V16 arms.  It returns the question-matched section from the
retrieved Top-1 historical case, so that report-style transfer can be
measured separately from multimodal generation.  The target report is never
used to construct the answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402


QUESTION_TYPES = ("findings", "impression")
RANKING_NAME = "rrf_lambdamart"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def canonical_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def sha256_ids(values: Sequence[str]) -> str:
    payload = "\n".join(sorted({str(value).strip() for value in values}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def paired_bootstrap(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int | bool]:
    left_by_key = {
        (str(row["case_id"]), str(row["question_type"])): float(row["token_f1"])
        for row in left
    }
    right_by_key = {
        (str(row["case_id"]), str(row["question_type"])): float(row["token_f1"])
        for row in right
    }
    if set(left_by_key) != set(right_by_key):
        raise RuntimeError("Extractive and comparison rows do not share the same keys")
    by_case: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left_by_key):
        by_case[key[0]].append(left_by_key[key] - right_by_key[key])
    case_values = np.asarray(
        [mean(by_case[case_id]) for case_id in sorted(by_case)], dtype=np.float64
    )
    if case_values.size == 0:
        raise RuntimeError("No paired cases available")
    rng = np.random.default_rng(seed)
    draws = case_values[rng.integers(0, len(case_values), size=(iterations, len(case_values)))].mean(axis=1)
    low = float(np.quantile(draws, 0.025))
    high = float(np.quantile(draws, 0.975))
    return {
        "case_count": int(case_values.size),
        "mean_difference": float(case_values.mean()),
        "ci_95_low": low,
        "ci_95_high": high,
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "iterations": iterations,
        "seed": seed,
    }


def cluster_map(split: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(case_id): str(cluster["cluster_id"])
        for cluster in split.get("clusters", [])
        for case_id in cluster.get("case_ids", [])
    }


def build_rows(
    cases: Mapping[str, Mapping[str, Any]],
    selected_case_ids: Sequence[str],
    ranking_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    clusters: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_id in selected_case_ids:
        target = cases[case_id]
        for question_type in QUESTION_TYPES:
            ranking = ranking_rows[(case_id, question_type)]["rankings"][RANKING_NAME]
            if not ranking:
                raise RuntimeError(f"Empty ranking for {case_id}/{question_type}")
            top1_id = str(ranking[0])
            if top1_id == case_id:
                raise RuntimeError(f"Target case leaked into Top-1 ranking: {case_id}")
            if top1_id not in cases:
                raise RuntimeError(f"Top-1 case is absent from case bank: {top1_id}")
            source = cases[top1_id]
            answer = canonical_text(source.get(question_type))
            reference = canonical_text(target.get(question_type))
            rows.append(
                {
                    "case_id": case_id,
                    "question_type": question_type,
                    "condition": "retrieved_history",
                    "retrieved_case_ids": [top1_id],
                    "retrieved_top1_case_id": top1_id,
                    "retrieved_top1_cluster_id": clusters.get(top1_id),
                    "target_cluster_id": clusters.get(case_id),
                    "same_cluster_top1": float(
                        bool(clusters.get(top1_id))
                        and clusters.get(top1_id) == clusters.get(case_id)
                    ),
                    "answer": answer,
                    "reference_answer": reference,
                    "reference_available": float(bool(reference)),
                    "token_f1": token_f1(answer, reference) if reference else None,
                    "answer_only_contract_valid": float(bool(answer)),
                    "evidence_provenance_valid": 1.0,
                    "hit_token_ceiling": 0.0,
                    "input_tokens": 0,
                    "output_tokens": len(answer.split()),
                    "latency_seconds": 0.0,
                    "baseline_type": "retrieval_copy_diagnostic",
                }
            )
    return rows


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    scored = [row for row in selected if bool(row["reference_available"])]
    return {
        "row_count": len(selected),
        "case_count": len({str(row["case_id"]) for row in selected}),
        "reference_available_row_count": len(scored),
        "reference_available_rate": len(scored) / len(selected) if selected else 0.0,
        "token_f1": mean([float(row["token_f1"]) for row in scored]),
        "answer_only_contract_valid_rate": mean(
            [float(row["answer_only_contract_valid"]) for row in scored]
        ),
        "evidence_provenance_valid_rate": mean(
            [float(row["evidence_provenance_valid"]) for row in scored]
        ),
        "same_cluster_top1_rate": mean(
            [float(row["same_cluster_top1"]) for row in selected]
        ),
        "by_question_type": {
            question_type: {
                "row_count": sum(
                    row["question_type"] == question_type and bool(row["reference_available"])
                    for row in selected
                ),
                "token_f1": mean(
                    [
                        float(row["token_f1"])
                        for row in scored
                        if row["question_type"] == question_type
                    ]
                ),
            }
            for question_type in QUESTION_TYPES
        },
    }


def run(args: argparse.Namespace) -> None:
    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    split = read_json(args.split)
    clusters = cluster_map(split)
    selected_case_ids = sorted(
        {str(row["case_id"]) for row in read_jsonl(args.selection_rows)}
    )
    if len(selected_case_ids) != 48:
        raise RuntimeError("The diagnostic must use the preselected 48-case Validation cohort")
    ranking_rows = {
        (str(row["case_id"]), str(row["question_type"])): row
        for row in read_jsonl(args.ranking_rows)
        if str(row["case_id"]) in set(selected_case_ids)
        and str(row["question_type"]) in QUESTION_TYPES
    }
    expected_keys = {
        (case_id, question_type)
        for case_id in selected_case_ids
        for question_type in QUESTION_TYPES
    }
    if set(ranking_rows) != expected_keys:
        raise RuntimeError("Ranking rows do not cover the fixed 48 x 2 matrix")
    rows = build_rows(cases, selected_case_ids, ranking_rows, clusters)
    if len(rows) != 96:
        raise RuntimeError(f"Unexpected diagnostic row count: {len(rows)}")

    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    base_rows = [
        row
        for row in read_jsonl(args.base_rows)
        if str(row["condition"]) == "retrieved_history"
    ]
    valid_keys = {
        (str(row["case_id"]), str(row["question_type"]))
        for row in rows
        if bool(row["reference_available"])
    }
    base_by_key = {
        (str(row["case_id"]), str(row["question_type"])): row
        for row in base_rows
    }
    comparison = paired_bootstrap(
        [row for row in rows if (str(row["case_id"]), str(row["question_type"])) in valid_keys],
        [base_by_key[key] for key in sorted(valid_keys)],
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    summary = {
        "study": "V16 retrieval-copy diagnostic",
        "status": "validation_only_no_test_evaluation",
        "baseline_type": (
            "The answer is copied from the question-matched section of the saved V12 Top-1 "
            "historical ranking. This is an extractive report-transfer diagnostic, not a "
            "generative or clinical-diagnosis system."
        ),
        "counts": {
            "cases": len(selected_case_ids),
            "rows": len(rows),
            "target_top1_leak_count": sum(row["retrieved_top1_case_id"] == row["case_id"] for row in rows),
            "same_cluster_top1_count": int(sum(float(row["same_cluster_top1"]) for row in rows)),
            "missing_target_reference_count": sum(not bool(row["reference_available"]) for row in rows),
        },
        "metrics": summarize(rows),
        "comparison_with_frozen_base_retrieved_generator": {
            "comparison_rows": str(args.base_rows.resolve().relative_to(ROOT).as_posix()),
            "token_f1_copy_minus_generator": comparison,
        },
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "split_sha256": file_sha256(args.split),
            "selection_rows_sha256": file_sha256(args.selection_rows),
            "ranking_rows_sha256": file_sha256(args.ranking_rows),
            "base_rows_sha256": file_sha256(args.base_rows),
            "selected_case_ids_sha256": sha256_ids(selected_case_ids),
            "ranking_name": RANKING_NAME,
        },
        "outputs": {
            "rows": str(args.rows_output.resolve().relative_to(ROOT).as_posix()),
            "rows_sha256": file_sha256(args.rows_output),
        },
        "runtime": {
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "claim_boundary": (
            "The copied historical section is evidence-transfer diagnostic only. It does not "
            "establish that the historical case matches the target patient, clinical diagnostic "
            "accuracy, clinical safety, physician agreement, or external validation."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--ranking-rows", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_qwen3_validation_rankings_rows.jsonl")
    parser.add_argument("--selection-rows", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_selection_rows.jsonl")
    parser.add_argument("--base-rows", type=Path, default=ROOT / "experiments/v16_adaptation/generation_base_batched.jsonl")
    parser.add_argument("--rows-output", type=Path, default=ROOT / "experiments/v16_adaptation/generation_extractive_copy_top1_rows.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/splits/v16/v16_extractive_copy_baseline_summary.json")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1919)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
