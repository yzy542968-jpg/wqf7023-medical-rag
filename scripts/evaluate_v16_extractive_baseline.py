"""Evaluate the V16 extractive report-transfer diagnostic.

The extractive system copies the question-matched section from the saved
Top-1 historical case.  It is intentionally evaluated separately from the
generative arms because it measures report-style evidence transfer rather
than target-patient diagnosis or answer generation.
"""

from __future__ import annotations

import argparse
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
from evaluate_v13_concept_qa_pilot import RADGRAPH_METRICS, score_radgraph  # noqa: E402
from medical_rag.evaluation.chexbert_pathology import (  # noqa: E402
    build_case_statistics,
    metrics_from_case_statistics,
    paired_case_bootstrap,
)
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402


QUESTION_TYPES = ("findings", "impression")


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


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = [
        (str(row["case_id"]), str(row["question_type"]))
        for row in rows
        if bool(row.get("reference_available"))
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate extractive baseline rows")
    if any(str(row["question_type"]) not in QUESTION_TYPES for row in rows):
        raise RuntimeError("Unexpected extractive question type")
    if any(str(row.get("condition")) != "retrieved_history" for row in rows):
        raise RuntimeError("Extractive baseline condition is not retrieved_history")


def linear_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "case_count": len({str(row["case_id"]) for row in rows}),
        "token_f1": mean([float(row["token_f1"]) for row in rows]),
        **{
            metric: mean([float(row[metric]) for row in rows])
            for metric in RADGRAPH_METRICS
        },
        "by_question_type": {
            question_type: {
                "row_count": sum(row["question_type"] == question_type for row in rows),
                "token_f1": mean(
                    [
                        float(row["token_f1"])
                        for row in rows
                        if row["question_type"] == question_type
                    ]
                ),
            }
            for question_type in QUESTION_TYPES
        },
    }


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
        raise RuntimeError(f"Extractive paired keys differ for {metric}")
    grouped: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left_by_key):
        grouped[key[0]].append(left_by_key[key] - right_by_key[key])
    values = np.asarray(
        [mean(grouped[case_id]) for case_id in sorted(grouped)], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    low = float(np.quantile(draws, 0.025))
    high = float(np.quantile(draws, 0.975))
    return {
        "case_count": int(len(values)),
        "mean_difference": float(values.mean()),
        "ci_95_low": low,
        "ci_95_high": high,
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "iterations": iterations,
        "seed": seed,
    }


def run(args: argparse.Namespace) -> None:
    raw = read_jsonl(args.rows)
    validate_rows(raw)
    rows = [row for row in raw if bool(row.get("reference_available"))]
    base = [
        row
        for row in read_jsonl(args.base_rows)
        if str(row["condition"]) == "retrieved_history"
        and (str(row["case_id"]), str(row["question_type"]))
        in {(str(value["case_id"]), str(value["question_type"])) for value in rows}
    ]
    if len(base) != len(rows):
        raise RuntimeError("Frozen base retrieved rows do not cover extractive rows")

    checkpoint = resolve_checkpoint()
    labels = label_unique_texts(
        [*rows, *base],
        cache_path=args.chexbert_cache,
        checkpoint_hash=file_sha256(checkpoint),
        device=args.device,
        batch_size=args.chexbert_batch_size,
    )

    enriched: dict[str, list[dict[str, Any]]] = {}
    for name, source_rows in (("extractive", rows), ("base_generator", base)):
        enriched[name] = []
        for source in source_rows:
            row = dict(source)
            row["reference_labels"] = np.asarray(
                labels[text_sha256(str(row.get("reference_answer") or ""))], dtype=np.int8
            )
            row["prediction_labels"] = np.asarray(
                labels[text_sha256(str(row.get("answer") or ""))], dtype=np.int8
            )
            enriched[name].append(row)

    radgraph: dict[str, dict[tuple[str, str, str], dict[str, float]]] = {}
    for name, source_rows in enriched.items():
        cache_path = args.radgraph_output.with_name(
            f"{args.radgraph_output.stem}_{name}{args.radgraph_output.suffix}"
        )
        scored = score_radgraph(
            source_rows,
            output_path=cache_path,
            model_type=args.radgraph_model,
            batch_size=args.radgraph_batch_size,
            cuda=args.cuda,
        )
        radgraph[name] = scored
        for row in source_rows:
            key = (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))
            row.update(scored[key])

    statistics_by_name: dict[str, Any] = {}
    for name, source_rows in enriched.items():
        stats = build_case_statistics(
            [str(row["case_id"]) for row in source_rows],
            np.stack([row["reference_labels"] for row in source_rows]),
            np.stack([row["prediction_labels"] for row in source_rows]),
        )
        statistics_by_name[name] = stats

    extractive_stats = statistics_by_name["extractive"]
    base_stats = statistics_by_name["base_generator"]
    paired = {
        "chexbert": paired_case_bootstrap(
            extractive_stats,
            base_stats,
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed,
        ),
        "linear": {
            metric: paired_linear_bootstrap(
                enriched["extractive"],
                enriched["base_generator"],
                metric,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + index + 1,
            )
            for index, metric in enumerate(("token_f1", *RADGRAPH_METRICS))
        },
    }
    output = {
        "study": "V16 retrieval-copy diagnostic metrics",
        "status": "validation_only_no_test_evaluation",
        "scored_reference_rows": len(rows),
        "dropped_missing_reference_rows": len(raw) - len(rows),
        "arms": {
            "extractive_copy_top1": {
                "chexbert": metrics_from_case_statistics(extractive_stats),
                "linear": linear_summary(enriched["extractive"]),
            },
            "frozen_base_retrieved_generator": {
                "chexbert": metrics_from_case_statistics(base_stats),
                "linear": linear_summary(enriched["base_generator"]),
            },
        },
        "extractive_minus_base_generator": paired,
        "inputs": {
            "rows_sha256": file_sha256(args.rows),
            "base_rows_sha256": file_sha256(args.base_rows),
            "chexbert_checkpoint_sha256": file_sha256(checkpoint),
            "radgraph_model": args.radgraph_model,
        },
        "runtime": {
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "claim_boundary": (
            "This is an extractive report-transfer diagnostic. It does not establish target "
            "patient alignment, clinical diagnostic accuracy, safety, physician agreement, "
            "or external validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=ROOT / "experiments/v16_adaptation/generation_extractive_copy_top1_rows.jsonl")
    parser.add_argument("--base-rows", type=Path, default=ROOT / "experiments/v16_adaptation/generation_base_batched.jsonl")
    parser.add_argument("--chexbert-cache", type=Path, default=ROOT / "experiments/v16_adaptation/v16_chexbert_cache.json")
    parser.add_argument("--radgraph-output", type=Path, default=ROOT / "experiments/v16_adaptation/v16_radgraph_extractive.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/splits/v16/v16_extractive_copy_metrics.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chexbert-batch-size", type=int, default=64)
    parser.add_argument("--radgraph-model", default="modern-radgraph-xl")
    parser.add_argument("--radgraph-batch-size", type=int, default=8)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1920)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
