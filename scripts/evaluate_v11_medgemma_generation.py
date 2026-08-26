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

from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402


DEFAULT_ROWS = (
    ROOT
    / "experiments"
    / "v11_development"
    / "v11_medgemma_generation_48_clean_rows.jsonl"
)
DEFAULT_GENERATION_SUMMARY = (
    ROOT / "data" / "splits" / "v11" / "v11_medgemma_generation_48_clean_summary.json"
)
DEFAULT_METRIC_ROWS = (
    ROOT / "experiments" / "v11_development" / "v11_medgemma_generation_48_metric_rows.csv"
)
DEFAULT_OUTPUT = (
    ROOT / "data" / "splits" / "v11" / "v11_medgemma_generation_48_statistical_summary.json"
)
POLICIES = ("whole_report", "sentence_only", "case_to_fact")
METRICS = (
    "token_f1",
    "f1_radgraph_entity",
    "f1_radgraph_entity_relation",
    "f1_radgraph_complete",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def grouped_values(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, list[float]]:
    output: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        output[str(row["case_id"])].append(float(row[metric]))
    return dict(output)


def grouped_bootstrap_ci(
    values: Mapping[str, Sequence[float]], *, iterations: int, seed: int
) -> dict[str, float | int]:
    case_ids = sorted(values)
    case_means = np.asarray(
        [statistics.fmean(values[case_id]) for case_id in case_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(case_means), size=(iterations, len(case_means)))
    draws = case_means[indices].mean(axis=1)
    return {
        "case_count": int(len(case_ids)),
        "mean": float(case_means.mean()),
        "ci_95_low": float(np.quantile(draws, 0.025)),
        "ci_95_high": float(np.quantile(draws, 0.975)),
        "iterations": int(iterations),
        "seed": int(seed),
    }


def paired_grouped_bootstrap(
    first: Mapping[str, Sequence[float]],
    second: Mapping[str, Sequence[float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int | bool]:
    case_ids = sorted(set(first) & set(second))
    differences = np.asarray(
        [
            statistics.fmean(first[case_id])
            - statistics.fmean(second[case_id])
            for case_id in case_ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(iterations, len(differences)))
    draws = differences[indices].mean(axis=1)
    low = float(np.quantile(draws, 0.025))
    high = float(np.quantile(draws, 0.975))
    return {
        "case_count": int(len(case_ids)),
        "mean_difference": float(differences.mean()),
        "ci_95_low": low,
        "ci_95_high": high,
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "probability_greater_than_zero": float(np.mean(draws > 0.0)),
        "iterations": int(iterations),
        "seed": int(seed),
    }


def score_radgraph(rows: list[dict[str, Any]], *, model_type: str, batch_size: int, cuda: int) -> None:
    from radgraph import F1RadGraph

    scoreable = [index for index, row in enumerate(rows) if str(row.get("reference_answer", "")).strip()]
    scorer = F1RadGraph(
        reward_level="all",
        model_type=model_type,
        batch_size=batch_size,
        cuda=cuda,
    )
    _, reward_lists, _, _ = scorer(
        hyps=[str(rows[index]["assembled_output"]["answer"]) for index in scoreable],
        refs=[str(rows[index]["reference_answer"]) for index in scoreable],
    )
    for row in rows:
        row["f1_radgraph_entity"] = 0.0
        row["f1_radgraph_entity_relation"] = 0.0
        row["f1_radgraph_complete"] = 0.0
        row["radgraph_scoreable_reference"] = bool(str(row.get("reference_answer", "")).strip())
    for index, entity, entity_relation, complete in zip(scoreable, *reward_lists, strict=True):
        rows[index]["f1_radgraph_entity"] = float(entity)
        rows[index]["f1_radgraph_entity_relation"] = float(entity_relation)
        rows[index]["f1_radgraph_complete"] = float(complete)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen V11 48-case MedGemma generation rows.")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--generation-summary", type=Path, default=DEFAULT_GENERATION_SUMMARY)
    parser.add_argument("--metric-rows-output", type=Path, default=DEFAULT_METRIC_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-type", default="modern-radgraph-xl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=7111)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_summary = read_json(args.generation_summary)
    observed_hash = file_sha256(args.rows)
    if observed_hash != str(generation_summary["generation_rows_sha256"]):
        raise RuntimeError("V11 generation rows differ from the frozen clean summary")
    rows = read_jsonl(args.rows)
    if len(rows) != 432:
        raise RuntimeError(f"expected 432 frozen rows, observed {len(rows)}")
    if sorted({str(row["policy"]) for row in rows}) != sorted(POLICIES):
        raise RuntimeError("V11 evidence policies are incomplete")
    matrix = {(str(row["case_id"]), str(row["question_type"]), str(row["policy"])) for row in rows}
    if len(matrix) != len(rows):
        raise RuntimeError("duplicate case/question/policy rows detected")

    score_radgraph(rows, model_type=args.model_type, batch_size=args.batch_size, cuda=args.cuda)

    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    policy_summaries: dict[str, Any] = {}
    for policy_index, policy in enumerate(POLICIES):
        selected = [row for row in rows if row["policy"] == policy]
        policy_summaries[policy] = {
            "case_count": len({str(row["case_id"]) for row in selected}),
            "row_count": len(selected),
            "nonempty_reference_rows": sum(bool(row["radgraph_scoreable_reference"]) for row in selected),
            "metrics": {},
        }
        for metric_index, metric in enumerate(METRICS):
            values = grouped_values(selected, metric)
            grouped[(policy, metric)] = values
            policy_summaries[policy]["metrics"][metric] = grouped_bootstrap_ci(
                values,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + policy_index * 10 + metric_index,
            )

    primary = {
        metric: paired_grouped_bootstrap(
            grouped[("case_to_fact", metric)],
            grouped[("whole_report", metric)],
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + 100 + metric_index,
        )
        for metric_index, metric in enumerate(METRICS)
    }
    secondary = {
        metric: paired_grouped_bootstrap(
            grouped[("sentence_only", metric)],
            grouped[("whole_report", metric)],
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + 200 + metric_index,
        )
        for metric_index, metric in enumerate(METRICS)
    }

    args.metric_rows_output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "question_type",
        "policy",
        "radgraph_scoreable_reference",
        *METRICS,
    ]
    with args.metric_rows_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)

    output = {
        "study": "V11 clean 48-case MedGemma generation statistical audit",
        "status": "development_only_complete_no_retuning",
        "source_generation_rows_sha256": observed_hash,
        "model_type": args.model_type,
        "empty_reference_policy": (
            "Rows with an empty report reference are retained and assigned zero automated overlap; "
            "they are not removed after outcome inspection."
        ),
        "counts": {
            "cases": len({str(row["case_id"]) for row in rows}),
            "rows": len(rows),
            "scoreable_reference_rows": sum(bool(row["radgraph_scoreable_reference"]) for row in rows),
            "empty_reference_rows": sum(not bool(row["radgraph_scoreable_reference"]) for row in rows),
        },
        "policies": policy_summaries,
        "primary_case_to_fact_minus_whole_report": primary,
        "secondary_sentence_only_minus_whole_report": secondary,
        "metric_rows": {
            "path": args.metric_rows_output.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(args.metric_rows_output),
            "committed_to_public_repository": False,
        },
        "claim_boundary": (
            "F1RadGraph and Token-F1 are automated report-reference consistency metrics. "
            "This Validation-only audit is not physician-adjudicated correctness, clinical safety, "
            "or independent confirmation."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
