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
from medical_rag.evaluation.chexbert_pathology import (  # noqa: E402
    METRIC_NAMES as CHEXBERT_METRICS,
    build_case_statistics,
    metrics_from_case_statistics,
    paired_case_bootstrap,
)
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from run_v10_evidence_generation_development import read_json, read_jsonl  # noqa: E402


DEFAULT_ROWS = ROOT / "experiments/v13_target_concept/v13_concept_qa_rows.jsonl"
DEFAULT_GENERATION_SUMMARY = ROOT / "data/splits/v13/v13_concept_qa_generation_summary.json"
DEFAULT_CHEX_CACHE = ROOT / "experiments/v13_target_concept/v13_concept_qa_chexbert_cache.json"
DEFAULT_RADGRAPH = ROOT / "experiments/v13_target_concept/v13_concept_qa_radgraph.csv"
DEFAULT_OUTPUT = ROOT / "data/splits/v13/v13_concept_qa_evaluation_summary.json"
CONDITIONS = ("concept_off", "concept_on")
QUESTION_TYPES = ("findings", "impression")
RADGRAPH_METRICS = (
    "f1_radgraph_entity",
    "f1_radgraph_entity_relation",
    "f1_radgraph_complete",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_rows(
    rows: Sequence[Mapping[str, Any]], generation_summary: Mapping[str, Any]
) -> None:
    if generation_summary.get("status") != "validation_generation_complete_no_retuning":
        raise RuntimeError("V13 concept QA generation is not complete")
    if generation_summary["artifacts"]["output_rows_sha256"] != file_sha256(DEFAULT_ROWS):
        raise RuntimeError("V13 concept QA rows differ from their generation summary")
    keys = [
        (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))
        for row in rows
    ]
    if len(rows) != 384 or len(keys) != len(set(keys)):
        raise RuntimeError("V13 concept QA rows are incomplete or duplicated")
    if {key[2] for key in keys} != set(CONDITIONS):
        raise RuntimeError("V13 concept QA condition coverage differs")


def score_radgraph(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    model_type: str,
    batch_size: int,
    cuda: int,
) -> dict[tuple[str, str, str], dict[str, float]]:
    expected_keys = {
        (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))
        for row in rows
    }
    if output_path.exists():
        cached = read_csv(output_path)
        by_key = {
            (str(row["case_id"]), str(row["question_type"]), str(row["condition"])): {
                metric: float(row[metric]) for metric in RADGRAPH_METRICS
            }
            for row in cached
        }
        if set(by_key) != expected_keys:
            raise RuntimeError("Cached V13 RadGraph rows are incomplete")
        return by_key

    from radgraph import F1RadGraph

    scorer = F1RadGraph(
        reward_level="all", model_type=model_type, batch_size=batch_size, cuda=cuda
    )
    _, reward_lists, _, _ = scorer(
        hyps=[str(row.get("answer") or "") for row in rows],
        refs=[str(row.get("reference_answer") or "") for row in rows],
    )
    by_key = {}
    output_rows = []
    for row, entity, entity_relation, complete in zip(rows, *reward_lists, strict=True):
        key = (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))
        values = {
            "f1_radgraph_entity": float(entity),
            "f1_radgraph_entity_relation": float(entity_relation),
            "f1_radgraph_complete": float(complete),
        }
        by_key[key] = values
        output_rows.append(
            {
                "case_id": key[0],
                "question_type": key[1],
                "condition": key[2],
                **values,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "question_type", "condition", *RADGRAPH_METRICS]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    return by_key


def grouped_values(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, float]:
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(float(row[metric]))
    return {case_id: statistics.fmean(values) for case_id, values in by_case.items()}


def paired_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    if set(left) != set(right):
        raise RuntimeError("V13 paired metric case coverage differs")
    case_ids = sorted(left)
    differences = np.asarray([left[key] - right[key] for key in case_ids])
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(case_ids), len(case_ids))
        draws[index] = differences[sample].mean()
    return {
        "case_count": len(case_ids),
        "mean_difference": float(differences.mean()),
        "ci_95_low": float(np.quantile(draws, 0.025)),
        "ci_95_high": float(np.quantile(draws, 0.975)),
    }


def select_scope(
    rows: Sequence[dict[str, Any]], scope: str
) -> list[dict[str, Any]]:
    return list(rows) if scope == "all" else [row for row in rows if row["question_type"] == scope]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired V13 concept QA.")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--generation-summary", type=Path, default=DEFAULT_GENERATION_SUMMARY)
    parser.add_argument("--chexbert-cache", type=Path, default=DEFAULT_CHEX_CACHE)
    parser.add_argument("--radgraph-output", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chexbert-batch-size", type=int, default=128)
    parser.add_argument("--radgraph-model", default="modern-radgraph-xl")
    parser.add_argument("--radgraph-batch-size", type=int, default=8)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=7145)
    args = parser.parse_args()

    rows = read_jsonl(args.rows)
    generation_summary = read_json(args.generation_summary)
    if args.rows != DEFAULT_ROWS:
        raise RuntimeError("Custom V13 QA row paths require a separately frozen evaluator")
    validate_rows(rows, generation_summary)

    checkpoint = resolve_checkpoint()
    labels_by_hash = label_unique_texts(
        rows,
        cache_path=args.chexbert_cache,
        checkpoint_hash=file_sha256(checkpoint),
        device=args.device,
        batch_size=args.chexbert_batch_size,
    )
    radgraph_by_key = score_radgraph(
        rows,
        output_path=args.radgraph_output,
        model_type=args.radgraph_model,
        batch_size=args.radgraph_batch_size,
        cuda=args.cuda,
    )
    enriched = []
    for source in rows:
        row = dict(source)
        key = (str(row["case_id"]), str(row["question_type"]), str(row["condition"]))
        row.update(radgraph_by_key[key])
        row["reference_labels"] = np.asarray(
            labels_by_hash[text_sha256(str(row.get("reference_answer") or ""))],
            dtype=np.int8,
        )
        row["prediction_labels"] = np.asarray(
            labels_by_hash[text_sha256(str(row.get("answer") or ""))], dtype=np.int8
        )
        enriched.append(row)

    scopes = {}
    for scope_index, scope in enumerate(("all", *QUESTION_TYPES)):
        scoped = select_scope(enriched, scope)
        by_condition = {
            condition: [row for row in scoped if row["condition"] == condition]
            for condition in CONDITIONS
        }
        linear_metrics = ("token_f1", *RADGRAPH_METRICS)
        systems = {
            metric: {
                condition: statistics.fmean(
                    float(row[metric]) for row in by_condition[condition]
                )
                for condition in CONDITIONS
            }
            for metric in linear_metrics
        }
        comparisons = {
            metric: paired_bootstrap(
                grouped_values(by_condition["concept_on"], metric),
                grouped_values(by_condition["concept_off"], metric),
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + scope_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(linear_metrics)
        }

        chex_stats = {}
        chex_metrics = {}
        for condition in CONDITIONS:
            selected = by_condition[condition]
            statistics_by_case = build_case_statistics(
                [str(row["case_id"]) for row in selected],
                np.stack([row["reference_labels"] for row in selected]),
                np.stack([row["prediction_labels"] for row in selected]),
            )
            chex_stats[condition] = statistics_by_case
            chex_metrics[condition] = metrics_from_case_statistics(statistics_by_case)
        chexbert_systems = {
            metric: {
                condition: float(chex_metrics[condition][metric])
                for condition in CONDITIONS
            }
            for metric in CHEXBERT_METRICS
        }
        chexbert_comparison = paired_case_bootstrap(
            chex_stats["concept_on"],
            chex_stats["concept_off"],
            iterations=args.bootstrap_iterations,
            seed=args.bootstrap_seed + scope_index * 100 + 20,
        )
        scopes[scope] = {
            "linear_metrics": systems,
            "concept_on_minus_off": comparisons,
            "chexbert_metrics": chexbert_systems,
            "chexbert_concept_on_minus_off": chexbert_comparison,
        }

    integrity = {
        condition: {
            "answer_contract_valid_rate": statistics.fmean(
                float(row["assembled_schema_valid"])
                for row in enriched
                if row["condition"] == condition
            ),
            "citation_valid_rate": statistics.fmean(
                float(row["citation_valid"])
                for row in enriched
                if row["condition"] == condition
            ),
            "token_ceiling_rate": statistics.fmean(
                float(row["hit_token_ceiling"])
                for row in enriched
                if row["condition"] == condition
            ),
            "mean_input_tokens": statistics.fmean(
                float(row["input_tokens"])
                for row in enriched
                if row["condition"] == condition
            ),
        }
        for condition in CONDITIONS
    }
    concept_rows_by_case = {
        str(row["case_id"]): row
        for row in enriched
        if row["condition"] == "concept_on" and row["question_type"] == "findings"
    }
    concept_counts = [len(row["predicted_concepts"]) for row in concept_rows_by_case.values()]
    label_counts: dict[str, int] = defaultdict(int)
    for row in concept_rows_by_case.values():
        for concept in row["predicted_concepts"]:
            label_counts[str(concept["label"])] += 1
    concept_distribution = {
        "case_count": len(concept_rows_by_case),
        "cases_without_passing_concept": sum(count == 0 for count in concept_counts),
        "mean_concepts_per_case": statistics.fmean(concept_counts),
        "maximum_concepts_per_case": max(concept_counts),
        "label_case_counts": dict(sorted(label_counts.items())),
    }
    all_scope = scopes["all"]
    promotion = {
        "five_observation_micro_f1_ci_above_zero": (
            all_scope["chexbert_concept_on_minus_off"]["micro_f1_5"]["ci_95_low"]
            > 0
        ),
        "token_f1_point_difference_nonnegative": (
            all_scope["concept_on_minus_off"]["token_f1"]["mean_difference"] >= 0
        ),
        "complete_radgraph_point_difference_nonnegative": (
            all_scope["concept_on_minus_off"]["f1_radgraph_complete"][
                "mean_difference"
            ]
            >= 0
        ),
        "contract_and_citation_not_lower": all(
            integrity["concept_on"][metric] >= integrity["concept_off"][metric]
            for metric in ("answer_contract_valid_rate", "citation_valid_rate")
        ),
    }
    output = {
        "study": "V13 paired concept-on/off QA evaluation",
        "status": "validation_evaluation_complete_no_retuning",
        "test_outcomes_inspected": False,
        "counts": {"cases": 96, "questions_per_case": 2, "rows": len(rows)},
        "scopes": scopes,
        "integrity": integrity,
        "concept_distribution": concept_distribution,
        "promotion_diagnostics": promotion,
        "runtime": {
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "radgraph_model": args.radgraph_model,
            "f1chexbert_version": "0.0.2",
            "f1chexbert_checkpoint_sha256": file_sha256(checkpoint),
        },
        "artifacts": {
            "qa_rows_sha256": file_sha256(args.rows),
            "generation_summary_sha256": file_sha256(args.generation_summary),
            "chexbert_cache_sha256": file_sha256(args.chexbert_cache),
            "radgraph_rows_sha256": file_sha256(args.radgraph_output),
            "script_sha256": file_sha256(Path(__file__)),
        },
        "claim_boundary": (
            "Validation-only automated answer-reference consistency; not diagnostic "
            "accuracy, clinical safety, physician utility, or confirmation evidence."
        ),
    }
    args.summary_output.write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
