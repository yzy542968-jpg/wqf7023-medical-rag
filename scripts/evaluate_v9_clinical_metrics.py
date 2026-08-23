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
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import file_sha256, read_json  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v9_supplemental_validity.json"
DEFAULT_ROWS = ROOT / "experiments" / "post_submission_v9" / "v9_qa_raw_rows.jsonl"
DEFAULT_PER_ANSWER = (
    ROOT / "experiments" / "post_submission_v9" / "v9_clinical_metric_rows.csv"
)
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v9" / "v9_clinical_metrics_summary.json"

PROTOCOL_TO_SOURCE_SYSTEM = {
    "g1_bm25_report_rag": "g1_bm25_rag",
}


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
    values_by_case: Mapping[str, Sequence[float]], *, iterations: int, seed: int
) -> dict[str, float]:
    case_ids = sorted(values_by_case)
    case_means = np.asarray(
        [statistics.fmean(values_by_case[case_id]) for case_id in case_ids],
        dtype=np.float64,
    )
    observed = float(case_means.mean())
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draws[index] = case_means[
            rng.integers(0, len(case_means), size=len(case_means))
        ].mean()
    return {
        "mean": observed,
        "ci_95_low": float(np.quantile(draws, 0.025)),
        "ci_95_high": float(np.quantile(draws, 0.975)),
    }


def paired_grouped_bootstrap(
    first: Mapping[str, Sequence[float]],
    second: Mapping[str, Sequence[float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    case_ids = sorted(set(first) & set(second))
    differences = np.asarray(
        [
            statistics.fmean(first[case_id]) - statistics.fmean(second[case_id])
            for case_id in case_ids
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draws[index] = differences[
            rng.integers(0, len(differences), size=len(differences))
        ].mean()
    return {
        "case_count": len(case_ids),
        "difference": float(differences.mean()),
        "ci_95_low": float(np.quantile(draws, 0.025)),
        "ci_95_high": float(np.quantile(draws, 0.975)),
    }


def dependency_status() -> dict[str, Any]:
    chexbert_specs = {
        name: importlib.util.find_spec(name) is not None
        for name in ("f1chexbert", "rad_eval", "radeval")
    }
    available = any(chexbert_specs.values())
    return {
        "available": available,
        "module_probes": chexbert_specs,
        "status": "available_not_run" if available else "unavailable_local_dependency",
        "reason": (
            "A local F1CheXbert implementation was not installed; no proxy metric "
            "was substituted."
            if not available
            else "A local module was detected but requires an explicit compatible adapter."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen V9 QA answers with clinical generation metrics."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--per-answer-output", type=Path, default=DEFAULT_PER_ANSWER)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model-type", default="modern-radgraph-xl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cuda", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    metric_config = config["clinical_generation_metrics"]
    rows = read_jsonl(args.rows)
    expected_systems = {
        PROTOCOL_TO_SOURCE_SYSTEM.get(system, system)
        for system in metric_config["systems"]
    }
    rows = [row for row in rows if str(row["system"]) in expected_systems]
    systems = sorted({str(row["system"]) for row in rows})
    if set(systems) != expected_systems:
        raise RuntimeError("Frozen QA systems differ from the supplemental protocol.")

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
    entity_scores, entity_relation_scores, complete_scores = reward_lists
    for row, entity, entity_relation, complete in zip(
        rows,
        entity_scores,
        entity_relation_scores,
        complete_scores,
        strict=True,
    ):
        row["f1_radgraph_entity"] = float(entity)
        row["f1_radgraph_entity_relation"] = float(entity_relation)
        row["f1_radgraph_complete"] = float(complete)

    metrics = (
        "f1_radgraph_entity",
        "f1_radgraph_entity_relation",
        "f1_radgraph_complete",
    )
    iterations = int(metric_config["bootstrap_iterations"])
    seed = int(metric_config["bootstrap_seed"])
    summaries: dict[str, Any] = {}
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for system_index, system in enumerate(systems):
        system_rows = [row for row in rows if str(row["system"]) == system]
        metric_summary: dict[str, Any] = {}
        for metric_index, metric in enumerate(metrics):
            values = grouped_values(system_rows, metric)
            grouped[(system, metric)] = values
            metric_summary[metric] = grouped_bootstrap_ci(
                values,
                iterations=iterations,
                seed=seed + system_index * 10 + metric_index,
            )
        summaries[system] = {
            "case_count": len({str(row["case_id"]) for row in system_rows}),
            "question_count": len(system_rows),
            "metrics": metric_summary,
        }

    comparisons: dict[str, Any] = {}
    reference_system = "g3_learned_multimodal_rag"
    for comparison_index, baseline in enumerate(
        ("g0_no_retrieval", "g1_bm25_rag", "g2_fixed_multimodal_rag")
    ):
        comparisons[f"{reference_system}_minus_{baseline}"] = {
            metric: paired_grouped_bootstrap(
                grouped[(reference_system, metric)],
                grouped[(baseline, metric)],
                iterations=iterations,
                seed=seed + 100 + comparison_index * 10 + metric_index,
            )
            for metric_index, metric in enumerate(metrics)
        }

    args.per_answer_output.parent.mkdir(parents=True, exist_ok=True)
    public_fields = [
        "system",
        "case_id",
        "qid",
        "question_type",
        *metrics,
    ]
    with args.per_answer_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in public_fields} for row in rows)

    output = {
        "study": "V9 clinical generation metric audit",
        "status": "post_hoc_exploratory_complete",
        "source_rows_sha256": file_sha256(args.rows),
        "config_sha256": file_sha256(args.config),
        "model_type": args.model_type,
        "row_count": len(rows),
        "protocol_system_aliases": PROTOCOL_TO_SOURCE_SYSTEM,
        "protocol_deviation": (
            "The supplemental protocol used the descriptive G1 key "
            "g1_bm25_report_rag; the immutable frozen source rows use "
            "g1_bm25_rag. They denote the same BM25 report-RAG condition."
        ),
        "systems": summaries,
        "paired_comparisons": comparisons,
        "f1_chexbert": dependency_status(),
        "per_answer_output": {
            "path": args.per_answer_output.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(args.per_answer_output),
            "committed_to_public_repository": False,
        },
        "claim_boundary": (
            "F1-RadGraph is an automated semantic-overlap metric and is not a "
            "clinical correctness or safety adjudication."
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
