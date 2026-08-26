"""Evaluate the predeclared question-type evidence router on V12 rows."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUTER = {
    "findings": "whole_report",
    "impression": "whole_report",
    "acute": "case_to_fact",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def bootstrap_difference(
    rows: Sequence[Mapping[str, Any]],
    first_policy: str,
    second_policy: str,
    metric: str,
    *,
    iterations: int = 10000,
    seed: int = 1212,
) -> dict[str, float | int | bool]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["case_id"])][str(row["policy"])].append(float(row[metric]))
    differences = np.asarray(
        [
            mean(grouped[case_id][first_policy]) - mean(grouped[case_id][second_policy])
            for case_id in sorted(grouped)
            if grouped[case_id].get(first_policy) and grouped[case_id].get(second_policy)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(0, len(differences), size=(iterations, len(differences)))].mean(axis=1)
    low = float(np.quantile(samples, 0.025))
    high = float(np.quantile(samples, 0.975))
    return {
        "difference": float(differences.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "case_count": len(differences),
        "iterations": iterations,
        "seed": seed,
    }


def summarize(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row["question_type"]) in ROUTER]
    result: dict[str, Any] = {
        "all_rows": mean([float(row[metric]) for row in selected]),
        "non_proxy_rows": mean([float(row[metric]) for row in selected if not row["reference_is_proxy"]]),
        "by_question_type": {},
    }
    for question_type, policy in ROUTER.items():
        chosen = [row for row in selected if row["question_type"] == question_type and row["policy"] == policy]
        result["by_question_type"][question_type] = {
            "policy": policy,
            "rows": len(chosen),
            "all_rows": mean([float(row[metric]) for row in chosen]),
            "non_proxy_rows": mean([float(row[metric]) for row in chosen if not row["reference_is_proxy"]]),
        }
    return result


def summarize_fixed_policy(rows: Sequence[Mapping[str, Any]], policy: str, metric: str) -> dict[str, Any]:
    selected = [row for row in rows if str(row["policy"]) == policy]
    return {
        "policy": policy,
        "all_rows": mean([float(row[metric]) for row in selected]),
        "non_proxy_rows": mean([float(row[metric]) for row in selected if not row["reference_is_proxy"]]),
        "by_question_type": {
            question_type: {
                "rows": sum(str(row["question_type"]) == question_type for row in selected),
                "all_rows": mean([float(row[metric]) for row in selected if row["question_type"] == question_type]),
                "non_proxy_rows": mean([
                    float(row[metric])
                    for row in selected
                    if row["question_type"] == question_type and not row["reference_is_proxy"]
                ]),
            }
            for question_type in ROUTER
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_96_rows.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_question_router_analysis.json")
    args = parser.parse_args()
    rows = read_jsonl(args.rows)
    expected = {"findings", "impression", "acute"}
    if {str(row["question_type"]) for row in rows} != expected:
        raise RuntimeError("Question router input does not contain exactly the three V12 question types.")
    routed_rows = [
        row
        for row in rows
        if str(row["policy"]) == ROUTER.get(str(row["question_type"]))
    ]
    paired = {
        metric: bootstrap_difference(rows, "case_to_fact", "whole_report", metric)
        for metric in ("token_f1", "answer_only_contract_valid")
    }
    output = {
        "study": "V12 question-type evidence router analysis",
        "status": "validation_only_post_hoc_development",
        "no_test_evaluation": True,
        "input_rows": str(args.rows.resolve().relative_to(ROOT)),
        "router": ROUTER,
        "selection_information": "question_type only; no reference answer, target report, or Token-F1 is used at routing time",
        "rows": len(routed_rows),
        "cases": len({str(row["case_id"]) for row in routed_rows}),
        "metrics": {
            "token_f1": summarize(routed_rows, "token_f1"),
            "answer_only_contract_valid": summarize(routed_rows, "answer_only_contract_valid"),
            "input_tokens": summarize(routed_rows, "input_tokens"),
            "evidence_characters": summarize(routed_rows, "evidence_character_count"),
        },
        "fixed_policy_comparators": {
            "whole_report": summarize_fixed_policy(rows, "whole_report", "token_f1"),
            "case_to_fact": summarize_fixed_policy(rows, "case_to_fact", "token_f1"),
        },
        "paired_case_to_fact_minus_whole_report": paired,
        "claim_boundary": "This is a deterministic Validation-only exploratory router. Its rule was derived from the observed V12 pilot and must be prospectively frozen before any future confirmation study.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
