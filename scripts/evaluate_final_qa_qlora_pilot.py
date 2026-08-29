"""Audit the paired Calibration effect of the Final-QA QLoRA pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE_CONDITIONS = {
    "no_history": "b3_no_history_r2",
    "relevant_history": "p1_top3_image_neighbors_question_conditioned_evidence",
}
BOOTSTRAP_SEED = 7033
BOOTSTRAP_REPLICATES = 10_000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyed(rows: Iterable[dict[str, Any]], condition: str) -> dict[tuple[str, int], dict[str, Any]]:
    selected = {
        (str(row["case_id"]), int(row["question_index"])): row
        for row in rows
        if row["condition"] == condition
    }
    if len(selected) != 256:
        raise RuntimeError(f"Expected 256 rows for {condition}, found {len(selected)}")
    return selected


def metrics(rows: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    rows = list(rows)
    tp = fp = fn = 0
    exact = valid = 0
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        gold = set(int(value) for value in row["gold_indices"])
        pred = set(int(value) for value in row["predicted_indices"])
        is_exact = float(gold == pred)
        exact += int(is_exact)
        valid += int(bool(row["contract_valid"]))
        by_type[str(row["answer_type"])].append(is_exact)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    denominator = 2 * tp + fp + fn
    result: dict[str, float | int] = {
        "row_count": len(rows),
        "option_micro_f1": 2 * tp / denominator if denominator else 0.0,
        "exact_answer_set_accuracy": exact / len(rows),
        "contract_valid_rate": valid / len(rows),
    }
    for answer_type, values in sorted(by_type.items()):
        result[f"{answer_type}_exact_accuracy"] = float(np.mean(values))
    return result


def bootstrap_difference(
    left: dict[tuple[str, int], dict[str, Any]],
    right: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, float | int]:
    if set(left) != set(right):
        raise RuntimeError("Paired row keys differ between compared arms")
    by_case: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for key in sorted(left):
        by_case[key[0]].append(key)
    case_ids = sorted(by_case)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    f1_differences = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    exact_differences = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(case_ids, size=len(case_ids), replace=True)
        keys = [key for case_id in sampled for key in by_case[str(case_id)]]
        left_metrics = metrics(left[key] for key in keys)
        right_metrics = metrics(right[key] for key in keys)
        f1_differences[index] = float(left_metrics["option_micro_f1"]) - float(
            right_metrics["option_micro_f1"]
        )
        exact_differences[index] = float(
            left_metrics["exact_answer_set_accuracy"]
        ) - float(right_metrics["exact_answer_set_accuracy"])
    return {
        "case_count": len(case_ids),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "option_micro_f1_difference_ci95_low": float(np.quantile(f1_differences, 0.025)),
        "option_micro_f1_difference_ci95_high": float(np.quantile(f1_differences, 0.975)),
        "exact_accuracy_difference_ci95_low": float(np.quantile(exact_differences, 0.025)),
        "exact_accuracy_difference_ci95_high": float(np.quantile(exact_differences, 0.975)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    base_rows = read_jsonl(args.base_rows)
    adapter_rows = read_jsonl(args.adapter_rows)
    training_rows = read_jsonl(args.training_rows)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    training_case_ids = {str(row["case_id"]) for row in training_rows}
    calibration_cases = manifest["roles"]["calibration"]["cases"]
    calibration_case_ids = {str(row["case_id"]) for row in calibration_cases}
    train_cluster_by_case = {
        str(row["case_id"]): str(row["cluster_id"])
        for row in manifest["roles"]["train"]["cases"]
    }
    training_clusters = {train_cluster_by_case[case_id] for case_id in training_case_ids}
    calibration_clusters = {str(row["cluster_id"]) for row in calibration_cases}
    case_overlap = sorted(training_case_ids & calibration_case_ids)
    cluster_overlap = sorted(training_clusters & calibration_clusters)
    if case_overlap or cluster_overlap:
        raise RuntimeError("Train/Calibration case or duplicate-cluster leakage detected")
    comparison: dict[str, Any] = {}
    adapter_by_condition: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for label, condition in BASE_CONDITIONS.items():
        base = keyed(base_rows, condition)
        adapter = keyed(adapter_rows, condition)
        if set(base) != set(adapter):
            raise RuntimeError(f"Base/adapter row keys differ for {label}")
        base_metrics = metrics(base.values())
        adapter_metrics = metrics(adapter.values())
        comparison[label] = {
            "base": base_metrics,
            "adapter": adapter_metrics,
            "adapter_minus_base": {
                "option_micro_f1": float(adapter_metrics["option_micro_f1"])
                - float(base_metrics["option_micro_f1"]),
                "exact_answer_set_accuracy": float(
                    adapter_metrics["exact_answer_set_accuracy"]
                )
                - float(base_metrics["exact_answer_set_accuracy"]),
                "contract_valid_rate": float(adapter_metrics["contract_valid_rate"])
                - float(base_metrics["contract_valid_rate"]),
            },
            "case_grouped_bootstrap": bootstrap_difference(adapter, base),
        }
        adapter_by_condition[label] = adapter

    no_history = adapter_by_condition["no_history"]
    relevant = adapter_by_condition["relevant_history"]
    history_effect = {
        "no_history": metrics(no_history.values()),
        "relevant_history": metrics(relevant.values()),
        "relevant_minus_no_history": {
            "option_micro_f1": float(metrics(relevant.values())["option_micro_f1"])
            - float(metrics(no_history.values())["option_micro_f1"]),
            "exact_answer_set_accuracy": float(
                metrics(relevant.values())["exact_answer_set_accuracy"]
            )
            - float(metrics(no_history.values())["exact_answer_set_accuracy"]),
        },
        "case_grouped_bootstrap": bootstrap_difference(relevant, no_history),
    }
    promotion = all(
        comparison[label]["adapter_minus_base"]["option_micro_f1"] > 0
        and comparison[label]["adapter_minus_base"]["contract_valid_rate"] >= -0.01
        for label in BASE_CONDITIONS
    )
    summary = {
        "study": "Final QA QLoRA paired Calibration evaluation",
        "status": "calibration_development_evaluation_complete",
        "base_rows_sha256": file_sha256(args.base_rows),
        "adapter_rows_sha256": file_sha256(args.adapter_rows),
        "isolation_audit": {
            "training_case_count": len(training_case_ids),
            "calibration_case_count": len(calibration_case_ids),
            "case_id_overlap_count": len(case_overlap),
            "duplicate_cluster_overlap_count": len(cluster_overlap),
            "training_rows_sha256": file_sha256(args.training_rows),
            "manifest_sha256": file_sha256(args.manifest),
        },
        "paired_row_count_per_condition": 256,
        "comparison": comparison,
        "adapter_relevant_history_effect": history_effect,
        "prespecified_promotion_rule_passed": promotion,
        "boundary": (
            "Calibration development evidence only. Validation and Test were not accessed; "
            "metrics are structured-answer agreement, not clinical accuracy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-rows",
        type=Path,
        default=ROOT / "experiments/final_qa_development/report_text_rag_pilot_rows.jsonl",
    )
    parser.add_argument(
        "--adapter-rows",
        type=Path,
        default=ROOT / "experiments/final_qa_development/final_qa_qlora_calibration_rows.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/final_qa_development/final_qa_qlora_paired_evaluation.json",
    )
    parser.add_argument(
        "--training-rows",
        type=Path,
        default=ROOT / "experiments/final_qa_development/final_qa_qlora_pilot_examples.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json",
    )
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
