from __future__ import annotations

import argparse
import csv
import importlib.util
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


DEFAULT_ROWS = ROOT / "experiments" / "v10_publication" / "v10_confirmation_qa_rows.jsonl"
DEFAULT_QA_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_confirmation_qa_summary.json"
DEFAULT_PER_ROW = ROOT / "experiments" / "v10_publication" / "v10_radgraph_metric_rows.csv"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_radgraph_metrics_summary.json"
SYSTEMS = (
    "g0_target_image",
    "g1_whole_report",
    "g2_hierarchical",
    "g3_selective",
)
METRICS = (
    "f1_radgraph_entity",
    "f1_radgraph_entity_relation",
    "f1_radgraph_complete",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def grouped_values(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, list[float]]:
    output: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        output[str(row["case_id"])].append(float(row[metric]))
    return dict(output)


def grouped_bootstrap_ci(
    values: Mapping[str, Sequence[float]], *, iterations: int, seed: int
) -> dict[str, float]:
    case_ids = sorted(values)
    means = np.asarray(
        [statistics.fmean(values[case_id]) for case_id in case_ids], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draws[index] = means[rng.integers(0, len(means), len(means))].mean()
    return {
        "mean": float(means.mean()),
        "ci_95_low": float(np.quantile(draws, 0.025)),
        "ci_95_high": float(np.quantile(draws, 0.975)),
    }


def paired_bootstrap(
    left: Mapping[str, Sequence[float]],
    right: Mapping[str, Sequence[float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    case_ids = sorted(set(left) & set(right))
    differences = np.asarray(
        [
            statistics.fmean(left[case_id]) - statistics.fmean(right[case_id])
            for case_id in case_ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draws[index] = differences[
            rng.integers(0, len(differences), len(differences))
        ].mean()
    return {
        "case_count": float(len(case_ids)),
        "mean_difference": float(differences.mean()),
        "ci_95_low": float(np.quantile(draws, 0.025)),
        "ci_95_high": float(np.quantile(draws, 0.975)),
    }


def chexbert_status() -> dict[str, Any]:
    probes = {
        name: importlib.util.find_spec(name) is not None
        for name in ("f1chexbert", "rad_eval", "radeval", "chexbert")
    }
    return {
        "available": any(probes.values()),
        "module_probes": probes,
        "status": "available_not_run" if any(probes.values()) else "unavailable_local_dependency",
        "reason": (
            "A compatible local implementation requires a separate frozen adapter."
            if any(probes.values())
            else "No compatible local F1CheXbert implementation was installed; no proxy was substituted."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen V10 QA rows with F1RadGraph.")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--qa-summary", type=Path, default=DEFAULT_QA_SUMMARY)
    parser.add_argument("--per-row-output", type=Path, default=DEFAULT_PER_ROW)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model-type", default="modern-radgraph-xl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=7052)
    args = parser.parse_args()

    qa_summary = read_json(args.qa_summary)
    observed_hash = file_sha256(args.rows)
    if qa_summary.get("status") != "confirmation_complete_no_retuning":
        raise RuntimeError("V10 QA confirmation is not complete")
    if observed_hash != str(qa_summary["qa_rows_sha256"]):
        raise RuntimeError("V10 QA rows differ from their frozen summary")
    rows = read_jsonl(args.rows)
    if sorted({str(row["system"]) for row in rows}) != sorted(SYSTEMS):
        raise RuntimeError("V10 QA systems are incomplete")

    from radgraph import F1RadGraph

    scorer = F1RadGraph(
        reward_level="all",
        model_type=args.model_type,
        batch_size=args.batch_size,
        cuda=args.cuda,
    )
    _, reward_lists, _, _ = scorer(
        hyps=[str(row.get("answer") or "") for row in rows],
        refs=[str(row.get("reference_answer") or "") for row in rows],
    )
    for row, entity, entity_relation, complete in zip(rows, *reward_lists, strict=True):
        row["f1_radgraph_entity"] = float(entity)
        row["f1_radgraph_entity_relation"] = float(entity_relation)
        row["f1_radgraph_complete"] = float(complete)

    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    summaries: dict[str, Any] = {}
    for system_index, system in enumerate(SYSTEMS):
        selected = [row for row in rows if row["system"] == system]
        summaries[system] = {
            "case_count": len({row["case_id"] for row in selected}),
            "row_count": len(selected),
            "metrics": {},
        }
        for metric_index, metric in enumerate(METRICS):
            values = grouped_values(selected, metric)
            grouped[(system, metric)] = values
            summaries[system]["metrics"][metric] = grouped_bootstrap_ci(
                values,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + system_index * 10 + metric_index,
            )

    comparisons = {}
    for comparison_index, baseline in enumerate(("g0_target_image", "g1_whole_report")):
        comparisons[f"g2_hierarchical_minus_{baseline}"] = {
            metric: paired_bootstrap(
                grouped[("g2_hierarchical", metric)],
                grouped[(baseline, metric)],
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + 100 + comparison_index * 10 + metric_index,
            )
            for metric_index, metric in enumerate(METRICS)
        }

    args.per_row_output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["system", "case_id", "question_type", *METRICS]
    with args.per_row_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)

    output = {
        "study": "V10 cluster-disjoint clinical generation metrics",
        "status": "confirmation_secondary_metric_complete_no_retuning",
        "source_rows_sha256": observed_hash,
        "model_type": args.model_type,
        "row_count": len(rows),
        "systems": summaries,
        "paired_comparisons": comparisons,
        "f1_chexbert": chexbert_status(),
        "per_row_output": {
            "sha256": file_sha256(args.per_row_output),
            "committed_to_public_repository": False,
        },
        "claim_boundary": (
            "F1RadGraph is automated graph overlap, not physician-adjudicated clinical correctness."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
