"""Evaluate V16 base/QLoRA rows with Token-F1, CheXbert, and RadGraph."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_v10_pathology_utility import (  # noqa: E402
    label_unique_texts,
    resolve_checkpoint,
    text_sha256,
)
from evaluate_v13_concept_qa_pilot import (  # noqa: E402
    RADGRAPH_METRICS,
    score_radgraph,
)
from medical_rag.evaluation.chexbert_pathology import (  # noqa: E402
    METRIC_NAMES as CHEXBERT_METRICS,
    build_case_statistics,
    metrics_from_case_statistics,
    paired_case_bootstrap,
)
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402


ARMS = ("base", "qlora")
CONDITIONS = ("no_history", "retrieved_history", "random_history")
QUESTION_TYPES = ("findings", "impression")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def validate_matrix(rows: Sequence[Mapping[str, Any]], arm: str) -> None:
    if {str(row.get("model_arm")) for row in rows} != {arm}:
        raise RuntimeError(f"Unexpected model arm in {arm} rows")
    keys = [
        (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))
        for row in rows
    ]
    expected = {
        (case_id, question_type, condition)
        for case_id in {key[0] for key in keys}
        for question_type in QUESTION_TYPES
        for condition in CONDITIONS
    }
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError(f"Incomplete or duplicated {arm} matrix")


def grouped_case_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(float(row[metric]))
    return {case_id: mean(values) for case_id, values in grouped.items()}


def paired_linear_bootstrap(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int | bool]:
    left_by_key = {
        (str(row["case_id"]), str(row["question_type"])): float(row[metric])
        for row in left
    }
    right_by_key = {
        (str(row["case_id"]), str(row["question_type"])): float(row[metric])
        for row in right
    }
    if set(left_by_key) != set(right_by_key):
        raise RuntimeError(f"Paired V16 rows differ for {metric}")
    by_case: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left_by_key):
        by_case[key[0]].append(left_by_key[key] - right_by_key[key])
    values = np.asarray([mean(by_case[key]) for key in sorted(by_case)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    low = float(np.quantile(draws, 0.025))
    high = float(np.quantile(draws, 0.975))
    return {
        "case_count": len(values),
        "mean_difference": float(values.mean()),
        "ci_95_low": low,
        "ci_95_high": high,
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "iterations": iterations,
        "seed": seed,
    }


def save_radgraph_subset(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "question_type", "condition", *RADGRAPH_METRICS],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    base = read_jsonl(args.base_rows)
    qlora = read_jsonl(args.qlora_rows)
    validate_matrix(base, "base")
    validate_matrix(qlora, "qlora")
    base_keys = {(str(row["case_id"]), str(row["question_type"]), str(row["condition"])) for row in base}
    qlora_keys = {(str(row["case_id"]), str(row["question_type"]), str(row["condition"])) for row in qlora}
    if base_keys != qlora_keys:
        raise RuntimeError("Base and QLoRA matrices differ")

    checkpoint = resolve_checkpoint()
    labels = label_unique_texts(
        [*base, *qlora],
        cache_path=args.chexbert_cache,
        checkpoint_hash=file_sha256(checkpoint),
        device=args.device,
        batch_size=args.chexbert_batch_size,
    )
    enriched: dict[str, list[dict[str, Any]]] = {}
    for arm, rows in (("base", base), ("qlora", qlora)):
        enriched[arm] = []
        for source in rows:
            row = dict(source)
            row["reference_labels"] = np.asarray(
                labels[text_sha256(str(row.get("reference_answer") or ""))], dtype=np.int8
            )
            row["prediction_labels"] = np.asarray(
                labels[text_sha256(str(row.get("answer") or ""))], dtype=np.int8
            )
            enriched[arm].append(row)

    radgraph_by_arm: dict[str, dict[tuple[str, str, str], dict[str, float]]] = {}
    for arm in ARMS:
        rows = enriched[arm]
        cache_path = args.radgraph_output.with_name(
            f"{args.radgraph_output.stem}_{arm}{args.radgraph_output.suffix}"
        )
        scored = score_radgraph(
            rows,
            output_path=cache_path,
            model_type=args.radgraph_model,
            batch_size=args.radgraph_batch_size,
            cuda=args.cuda,
        )
        radgraph_by_arm[arm] = scored
        for row in rows:
            key = (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))
            row.update(scored[key])

    arm_condition_metrics: dict[str, Any] = {}
    for arm in ARMS:
        arm_condition_metrics[arm] = {}
        for condition in CONDITIONS:
            selected = [row for row in enriched[arm] if row["condition"] == condition]
            stats = build_case_statistics(
                [str(row["case_id"]) for row in selected],
                np.stack([row["reference_labels"] for row in selected]),
                np.stack([row["prediction_labels"] for row in selected]),
            )
            point = metrics_from_case_statistics(stats)
            linear = {
                metric: mean([float(row[metric]) for row in selected])
                for metric in ("token_f1", *RADGRAPH_METRICS)
            }
            arm_condition_metrics[arm][condition] = {
                "row_count": len(selected),
                "chexbert": point,
                "linear": linear,
                "answer_only_contract_valid_rate": mean(
                    [float(row["answer_only_contract_valid"]) for row in selected]
                ),
                "provenance_valid_rate": mean(
                    [float(row["evidence_provenance_valid"]) for row in selected]
                ),
                "token_ceiling_rate": mean(
                    [float(row["hit_token_ceiling"]) for row in selected]
                ),
            }

    qlora_minus_base: dict[str, Any] = {}
    for condition_index, condition in enumerate(CONDITIONS):
        q_rows = [row for row in enriched["qlora"] if row["condition"] == condition]
        b_rows = [row for row in enriched["base"] if row["condition"] == condition]
        linear_metrics = {metric: paired_linear_bootstrap(
            q_rows,
            b_rows,
            metric,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + condition_index * 100 + metric_index,
        ) for metric_index, metric in enumerate(("token_f1", *RADGRAPH_METRICS))}
        q_stats = build_case_statistics(
            [str(row["case_id"]) for row in q_rows],
            np.stack([row["reference_labels"] for row in q_rows]),
            np.stack([row["prediction_labels"] for row in q_rows]),
        )
        b_stats = build_case_statistics(
            [str(row["case_id"]) for row in b_rows],
            np.stack([row["reference_labels"] for row in b_rows]),
            np.stack([row["prediction_labels"] for row in b_rows]),
        )
        qlora_minus_base[condition] = {
            "linear": linear_metrics,
            "chexbert": paired_case_bootstrap(
                q_stats,
                b_stats,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + condition_index * 100 + 50,
            ),
        }

    within_arm_comparisons: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        within_arm_comparisons[arm] = {}
        for comparison_index, (left_condition, right_condition) in enumerate(
            (("retrieved_history", "no_history"), ("retrieved_history", "random_history"))
        ):
            left_rows = [
                row for row in enriched[arm] if row["condition"] == left_condition
            ]
            right_rows = [
                row for row in enriched[arm] if row["condition"] == right_condition
            ]
            linear = {
                metric: paired_linear_bootstrap(
                    left_rows,
                    right_rows,
                    metric,
                    iterations=args.bootstrap_iterations,
                    seed=args.bootstrap_seed + 400 + comparison_index * 100 + metric_index,
                )
                for metric_index, metric in enumerate(("token_f1", *RADGRAPH_METRICS))
            }
            left_stats = build_case_statistics(
                [str(row["case_id"]) for row in left_rows],
                np.stack([row["reference_labels"] for row in left_rows]),
                np.stack([row["prediction_labels"] for row in left_rows]),
            )
            right_stats = build_case_statistics(
                [str(row["case_id"]) for row in right_rows],
                np.stack([row["reference_labels"] for row in right_rows]),
                np.stack([row["prediction_labels"] for row in right_rows]),
            )
            comparison_key = f"{left_condition}_minus_{right_condition}"
            within_arm_comparisons[arm][comparison_key] = {
                "linear": linear,
                "chexbert": paired_case_bootstrap(
                    left_stats,
                    right_stats,
                    iterations=args.bootstrap_iterations,
                    seed=args.bootstrap_seed + 450 + comparison_index * 100,
                ),
            }

    output = {
        "study": "V16 paired QLoRA generation metrics",
        "status": "validation_evaluation_complete_no_retuning",
        "no_test_evaluation": True,
        "counts": {
            "cases": len({str(row["case_id"]) for row in base}),
            "rows_per_arm": len(base),
        },
        "arms": arm_condition_metrics,
        "qlora_minus_base": qlora_minus_base,
        "within_arm_comparisons": within_arm_comparisons,
        "runtime": {
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "radgraph_model": args.radgraph_model,
            "f1chexbert_version": "0.0.2",
            "f1chexbert_checkpoint_sha256": file_sha256(checkpoint),
        },
        "inputs": {
            "base_rows_sha256": file_sha256(args.base_rows),
            "qlora_rows_sha256": file_sha256(args.qlora_rows),
        },
        "claim_boundary": (
            "Validation-only automated report-reference consistency; CheXbert and "
            "RadGraph are not clinical diagnosis accuracy, safety, physician agreement, "
            "or external validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-rows", type=Path, required=True)
    parser.add_argument("--qlora-rows", type=Path, required=True)
    parser.add_argument("--chexbert-cache", type=Path, default=ROOT / "experiments/v16_adaptation/v16_chexbert_cache.json")
    parser.add_argument("--radgraph-output", type=Path, default=ROOT / "experiments/v16_adaptation/v16_radgraph.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chexbert-batch-size", type=int, default=128)
    parser.add_argument("--radgraph-model", default="modern-radgraph-xl")
    parser.add_argument("--radgraph-batch-size", type=int, default=8)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1620)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
