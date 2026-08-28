"""Evaluate paired V16 base/QLoRA generation rows on Validation only."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def grouped_mean(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(float(row[metric]))
    return {key: mean(values) for key, values in grouped.items()}


def paired_bootstrap(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    iterations: int,
    seed: int,
    include_condition: bool = True,
) -> dict[str, float | int | bool]:
    def key(row: Mapping[str, Any]) -> tuple[str, ...]:
        fields = (str(row["case_id"]), str(row["question_type"]))
        if include_condition:
            return (*fields, str(row["condition"]))
        return fields

    left_by_key = {key(row): float(row[metric]) for row in left}
    right_by_key = {key(row): float(row[metric]) for row in right}
    if set(left_by_key) != set(right_by_key):
        raise RuntimeError(f"V16 paired keys differ for {metric}")
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


def paired_condition_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    left_condition: str,
    right_condition: str,
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int | bool]:
    left = [row for row in rows if row["condition"] == left_condition]
    right = [row for row in rows if row["condition"] == right_condition]
    return paired_bootstrap(
        left,
        right,
        metric,
        iterations=iterations,
        seed=seed,
        include_condition=False,
    )


def validate_rows(rows: Sequence[Mapping[str, Any]], expected_arm: str) -> None:
    if not rows:
        raise RuntimeError(f"No rows found for {expected_arm}")
    if {str(row.get("model_arm")) for row in rows} != {expected_arm}:
        raise RuntimeError(f"Unexpected model arm in {expected_arm} rows")
    keys = [
        (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate rows in {expected_arm} output")
    if {key[1] for key in keys} != {"findings", "impression"}:
        raise RuntimeError(f"Unexpected question types in {expected_arm} output")
    if {key[2] for key in keys} != {"no_history", "retrieved_history", "random_history"}:
        raise RuntimeError(f"Unexpected conditions in {expected_arm} output")


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
                        if row["question_type"] == question_type
                    ]),
                    "rows": sum(row["question_type"] == question_type for row in selected),
                }
                for question_type in ("findings", "impression")
            },
        }
    return result


def run(args: argparse.Namespace) -> None:
    base = read_jsonl(args.base_rows)
    qlora = read_jsonl(args.qlora_rows)
    validate_rows(base, "base")
    validate_rows(qlora, "qlora")
    base_keys = {
        (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))
        for row in base
    }
    qlora_keys = {
        (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))
        for row in qlora
    }
    if base_keys != qlora_keys:
        raise RuntimeError("Base and QLoRA generation matrices differ")
    metrics = (
        "token_f1",
        "answer_only_contract_valid",
        "evidence_provenance_valid",
        "hit_token_ceiling",
    )
    arm_summary = {"base": summarize(base), "qlora": summarize(qlora)}
    comparisons: dict[str, Any] = {}
    for index, metric in enumerate(metrics):
        comparisons[metric] = {
            condition: paired_bootstrap(
                [row for row in qlora if row["condition"] == condition],
                [row for row in base if row["condition"] == condition],
                metric,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + index * 100,
            )
            for condition in ("no_history", "retrieved_history", "random_history")
        }
    transfer = {
        arm: paired_condition_bootstrap(
            rows,
            "retrieved_history",
            "no_history",
            "token_f1",
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + offset,
        )
        for offset, (arm, rows) in enumerate((("base", base), ("qlora", qlora)), start=1000)
    }
    random_gap = {
        arm: paired_condition_bootstrap(
            rows,
            "retrieved_history",
            "random_history",
            "token_f1",
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + offset,
        )
        for offset, (arm, rows) in enumerate((("base", base), ("qlora", qlora)), start=2000)
    }
    output = {
        "study": "V16 QLoRA paired generation pilot evaluation",
        "status": "validation_evaluation_complete_no_retuning",
        "no_test_evaluation": True,
        "counts": {
            "cases": len({str(row["case_id"]) for row in base}),
            "rows_per_arm": len(base),
        },
        "arms": arm_summary,
        "qlora_minus_base": comparisons,
        "history_transfer_token_f1": transfer,
        "retrieved_minus_random_token_f1": random_gap,
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
    parser.add_argument("--base-rows", type=Path, required=True)
    parser.add_argument("--qlora-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1619)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
