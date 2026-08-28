"""Evaluate two paired V16 generation row files without assuming model arms."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))


def validate(rows: Sequence[Mapping[str, Any]], label: str) -> None:
    if not rows:
        raise RuntimeError(f"No rows found for {label}")
    keys = [row_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate rows in {label} output")
    if {key[1] for key in keys} != {"findings", "impression"}:
        raise RuntimeError(f"Unexpected question types in {label} output")
    if {key[2] for key in keys} != {"no_history", "retrieved_history", "random_history"}:
        raise RuntimeError(f"Unexpected conditions in {label} output")


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    result: dict[str, Any] = {}
    for condition, selected in sorted(by_condition.items()):
        result[condition] = {
            "row_count": len(selected),
            "case_count": len({str(row["case_id"]) for row in selected}),
            "token_f1": mean([float(row["token_f1"]) for row in selected]),
            "answer_only_contract_valid_rate": mean(
                [float(row["answer_only_contract_valid"]) for row in selected]
            ),
            "provenance_valid_rate": mean(
                [float(row["evidence_provenance_valid"]) for row in selected]
            ),
            "token_ceiling_rate": mean([float(row["hit_token_ceiling"]) for row in selected]),
            "mean_input_tokens": mean([float(row["input_tokens"]) for row in selected]),
            "mean_output_tokens": mean([float(row["output_tokens"]) for row in selected]),
            "by_question_type": {
                question_type: {
                    "token_f1": mean([
                        float(row["token_f1"])
                        for row in selected
                        if str(row["question_type"]) == question_type
                    ]),
                    "token_ceiling_rate": mean([
                        float(row["hit_token_ceiling"])
                        for row in selected
                        if str(row["question_type"]) == question_type
                    ]),
                    "rows": sum(
                        str(row["question_type"]) == question_type for row in selected
                    ),
                }
                for question_type in ("findings", "impression")
            },
        }
    return result


def paired_bootstrap(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int | bool]:
    left_by_key = {row_key(row): float(row[metric]) for row in left}
    right_by_key = {row_key(row): float(row[metric]) for row in right}
    if set(left_by_key) != set(right_by_key):
        raise RuntimeError(f"Paired keys differ for {metric}")
    by_case: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left_by_key):
        by_case[key[0]].append(left_by_key[key] - right_by_key[key])
    values = np.asarray([mean(by_case[key]) for key in sorted(by_case)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sample = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    low = float(np.quantile(sample, 0.025))
    high = float(np.quantile(sample, 0.975))
    return {
        "case_count": len(values),
        "mean_difference": float(values.mean()),
        "ci_95_low": low,
        "ci_95_high": high,
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "iterations": iterations,
        "seed": seed,
    }


def run(args: argparse.Namespace) -> None:
    left = read_jsonl(args.left_rows)
    right = read_jsonl(args.right_rows)
    validate(left, args.left_label)
    validate(right, args.right_label)
    if {row_key(row) for row in left} != {row_key(row) for row in right}:
        raise RuntimeError("Paired row matrices differ")
    metrics = (
        "token_f1",
        "answer_only_contract_valid",
        "evidence_provenance_valid",
        "hit_token_ceiling",
    )
    comparisons: dict[str, dict[str, Any]] = {}
    for metric_index, metric in enumerate(metrics):
        comparisons[metric] = {
            condition: paired_bootstrap(
                [row for row in left if str(row["condition"]) == condition],
                [row for row in right if str(row["condition"]) == condition],
                metric,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + metric_index * 100,
            )
            for condition in ("no_history", "retrieved_history", "random_history")
        }
    output = {
        "study": "V16 paired row evaluation",
        "status": "validation_evaluation_complete_no_retuning",
        "no_test_evaluation": True,
        "counts": {
            "cases": len({str(row["case_id"]) for row in left}),
            "rows_per_arm": len(left),
        },
        "arms": {
            args.left_label: summarize(left),
            args.right_label: summarize(right),
        },
        f"{args.left_label}_minus_{args.right_label}": comparisons,
        "runtime": {
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "claim_boundary": (
            "Automated Validation answer-reference consistency only; no diagnosis, "
            "clinical correctness, clinical safety, or external validation claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-rows", type=Path, required=True)
    parser.add_argument("--right-rows", type=Path, required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1619)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
