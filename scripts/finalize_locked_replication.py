from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        raise ValueError("n must be positive")
    proportion = successes / n
    denominator = 1.0 + z * z / n
    centre = (proportion + z * z / (2 * n)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / n + z * z / (4 * n * n)
    ) / denominator
    return [max(0.0, centre - half_width), min(1.0, centre + half_width)]


def grouped_bootstrap(
    rows: list[dict[str, Any]], iterations: int = 5000, seed: int = 57023
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    case_ids = sorted(by_case)
    rng = np.random.default_rng(seed)
    metrics = ("draft_token_f1", "final_token_f1", "support_rate")
    samples = {metric: [] for metric in metrics}
    for _ in range(iterations):
        selected = rng.choice(case_ids, size=len(case_ids), replace=True)
        sampled_rows = [row for case_id in selected for row in by_case[str(case_id)]]
        for metric in metrics:
            samples[metric].append(
                float(np.mean([float(row[metric]) for row in sampled_rows]))
            )
    return {
        "grouping_unit": "case_id",
        "case_count": len(case_ids),
        "iterations": iterations,
        "seed": seed,
        "confidence_intervals_95": {
            metric: [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ]
            for metric, values in samples.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize locked replication evidence.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "experiments" / "locked_replication" / "summary.json",
    )
    parser.add_argument(
        "--cohort",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_locked_replication_cohort.json",
    )
    parser.add_argument(
        "--generations",
        type=Path,
        default=ROOT / "experiments" / "locked_replication" / "generations_qwen15.jsonl",
    )
    parser.add_argument(
        "--semantic-summary",
        type=Path,
        default=ROOT
        / "experiments"
        / "locked_replication"
        / "semantic_final"
        / "final_optimized_test_summary.json",
    )
    parser.add_argument(
        "--semantic-rows",
        type=Path,
        default=ROOT
        / "experiments"
        / "locked_replication"
        / "semantic_final"
        / "final_optimized_test_rows.jsonl",
    )
    args = parser.parse_args()

    summary = _read_json(args.summary)
    for key in ("decisions_path", "prompt_pack"):
        value = Path(str(summary[key]))
        if value.is_absolute():
            summary[key] = str(value.relative_to(ROOT))
    cohort = _read_json(args.cohort)
    generations = _read_jsonl(args.generations)
    semantic_summary = _read_json(args.semantic_summary)
    semantic_rows = _read_jsonl(args.semantic_rows)
    expected_qids = {str(row["qid"]) for row in cohort["questions"]}
    generation_qids = [str(row["qid"]) for row in generations]
    semantic_qids = [str(row["qid"]) for row in semantic_rows]
    if len(generation_qids) != len(set(generation_qids)):
        raise ValueError("Generation output contains duplicate QIDs.")
    if set(generation_qids) != expected_qids or set(semantic_qids) != expected_qids:
        raise ValueError("Generation and semantic outputs must exactly match the cohort QIDs.")

    replication_accuracy = float(summary["retrieval"]["adaptive"]["hit@1"])
    original = _read_json(
        ROOT
        / "experiments"
        / "final_optimized"
        / "adaptive_retrieval"
        / "adaptive_policy_selection.json"
    )["held_out_test"]
    summary.update(
        {
            "status": "complete",
            "generation": {
                "model": generations[0]["model"],
                "prompt_mode": "direct",
                "temperature": 0.0,
                "max_new_tokens": 256,
                "record_count": len(generations),
                "unique_qid_count": len(set(generation_qids)),
                "sha256": _sha256(args.generations),
            },
            "semantic_evaluation": semantic_summary,
            "grouped_bootstrap": grouped_bootstrap(semantic_rows),
            "retrieval_replication_comparison": {
                "original_held_out": {
                    "n": int(original["n"]),
                    "adaptive_top1": float(original["overall_accuracy"]),
                    "wilson_95_ci": wilson_interval(
                        round(float(original["overall_accuracy"]) * int(original["n"])),
                        int(original["n"]),
                    ),
                },
                "untouched_replication": {
                    "n": len(generations),
                    "adaptive_top1": replication_accuracy,
                    "wilson_95_ci": wilson_interval(
                        round(replication_accuracy * len(generations)), len(generations)
                    ),
                },
                "absolute_difference": replication_accuracy
                - float(original["overall_accuracy"]),
            },
        }
    )
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
