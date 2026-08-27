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
    random_control_case_bootstrap,
)
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from run_v10_evidence_generation_development import read_json, read_jsonl  # noqa: E402


DEFAULT_FROZEN_ROWS = ROOT / "experiments/v10_publication/v10_confirmation_qa_rows.jsonl"
DEFAULT_FROZEN_QA_SUMMARY = ROOT / "data/splits/v10/v10_confirmation_qa_summary.json"
DEFAULT_FROZEN_RADGRAPH = ROOT / "experiments/v10_publication/v10_radgraph_metric_rows.csv"
DEFAULT_FROZEN_RADGRAPH_SUMMARY = ROOT / "data/splits/v10/v10_radgraph_metrics_summary.json"
DEFAULT_RANDOM_ROWS = ROOT / "experiments/v10_publication/v10_random_history_control_rows.jsonl"
DEFAULT_RANDOM_GENERATION_SUMMARY = ROOT / "data/splits/v10/v10_random_history_generation_summary.json"
DEFAULT_CHEX_CACHE = ROOT / "experiments/v10_publication/v10_random_history_chexbert_text_label_cache.json"
DEFAULT_RANDOM_RADGRAPH = ROOT / "experiments/v10_publication/v10_random_history_radgraph_rows.csv"
DEFAULT_SUMMARY = ROOT / "data/splits/v10/v10_random_history_control_summary.json"
PROTOCOL_COMMIT = "183c5e8"
QUESTION_TYPES = ("findings", "impression")
ASSIGNMENTS = tuple(range(5))
RADGRAPH_METRICS = (
    "f1_radgraph_entity",
    "f1_radgraph_entity_relation",
    "f1_radgraph_complete",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_sources(
    frozen_rows: Sequence[Mapping[str, Any]],
    random_rows: Sequence[Mapping[str, Any]],
    *,
    frozen_rows_hash: str,
    random_rows_hash: str,
    frozen_summary: Mapping[str, Any],
    random_summary: Mapping[str, Any],
) -> None:
    if frozen_summary.get("status") != "confirmation_complete_no_retuning":
        raise RuntimeError("Frozen V10 QA status is invalid")
    if frozen_summary.get("qa_rows_sha256") != frozen_rows_hash:
        raise RuntimeError("Frozen V10 QA row hash differs from its summary")
    if random_summary.get("status") != "posthoc_control_generation_complete_no_retuning":
        raise RuntimeError("Random-history generation is not complete")
    if random_summary["source_hashes"]["output_rows"] != random_rows_hash:
        raise RuntimeError("Random-history rows differ from their generation summary")
    if len(frozen_rows) != 4544 or len(random_rows) != 5680:
        raise RuntimeError("Unexpected frozen or random-history row count")
    keys = [
        (str(row["case_id"]), str(row["question_type"]), int(row["assignment"]))
        for row in random_rows
    ]
    if len(keys) != len(set(keys)) or {key[2] for key in keys} != set(ASSIGNMENTS):
        raise RuntimeError("Random-history assignment coverage is malformed")
    for assignment in ASSIGNMENTS:
        selected = [key for key in keys if key[2] == assignment]
        if len(selected) != 568 * len(QUESTION_TYPES):
            raise RuntimeError(f"Random-history assignment {assignment} is incomplete")


def score_random_radgraph(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    model_type: str,
    batch_size: int,
    cuda: int,
) -> dict[tuple[str, str, int], dict[str, float]]:
    expected_keys = {
        (str(row["case_id"]), str(row["question_type"]), int(row["assignment"]))
        for row in rows
    }
    if output_path.exists():
        cached = read_csv(output_path)
        by_key = {
            (str(row["case_id"]), str(row["question_type"]), int(row["assignment"])): {
                metric: float(row[metric]) for metric in RADGRAPH_METRICS
            }
            for row in cached
        }
        if set(by_key) != expected_keys:
            raise RuntimeError("Cached random-history RadGraph rows are incomplete")
        return by_key

    from radgraph import F1RadGraph

    scorer = F1RadGraph(
        reward_level="all",
        model_type=model_type,
        batch_size=batch_size,
        cuda=cuda,
    )
    _, reward_lists, _, _ = scorer(
        hyps=[str(row.get("answer") or "") for row in rows],
        refs=[str(row.get("reference_answer") or "") for row in rows],
    )
    by_key = {}
    output_rows = []
    for row, entity, entity_relation, complete in zip(rows, *reward_lists, strict=True):
        key = (str(row["case_id"]), str(row["question_type"]), int(row["assignment"]))
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
                "assignment": key[2],
                **values,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "question_type", "assignment", *RADGRAPH_METRICS]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    return by_key


def grouped_linear_values(
    rows: Sequence[Mapping[str, Any]], metric: str, *, assignment_mean: bool
) -> dict[str, float]:
    by_case_question: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_case_question[(str(row["case_id"]), str(row["question_type"]))].append(
            float(row[metric])
        )
    if assignment_mean and any(len(values) != len(ASSIGNMENTS) for values in by_case_question.values()):
        raise RuntimeError("Random-history metric does not contain five values per question")
    by_case: dict[str, list[float]] = defaultdict(list)
    for (case_id, _), values in by_case_question.items():
        by_case[case_id].append(statistics.fmean(values))
    return {case_id: statistics.fmean(values) for case_id, values in by_case.items()}


def paired_linear_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    if set(left) != set(right):
        raise RuntimeError("Linear comparison case coverage differs")
    case_ids = sorted(left)
    differences = np.asarray([left[case_id] - right[case_id] for case_id in case_ids])
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draws[index] = differences[rng.integers(0, len(differences), len(differences))].mean()
    return {
        "case_count": len(case_ids),
        "mean_difference": float(differences.mean()),
        "ci_95_low": float(np.quantile(draws, 0.025)),
        "ci_95_high": float(np.quantile(draws, 0.975)),
    }


def select_scope(rows: Sequence[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    if scope == "all":
        return list(rows)
    return [row for row in rows if row["question_type"] == scope]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the V10 random-history control.")
    parser.add_argument("--frozen-rows", type=Path, default=DEFAULT_FROZEN_ROWS)
    parser.add_argument("--frozen-qa-summary", type=Path, default=DEFAULT_FROZEN_QA_SUMMARY)
    parser.add_argument("--frozen-radgraph", type=Path, default=DEFAULT_FROZEN_RADGRAPH)
    parser.add_argument(
        "--frozen-radgraph-summary", type=Path, default=DEFAULT_FROZEN_RADGRAPH_SUMMARY
    )
    parser.add_argument("--random-rows", type=Path, default=DEFAULT_RANDOM_ROWS)
    parser.add_argument(
        "--random-generation-summary", type=Path, default=DEFAULT_RANDOM_GENERATION_SUMMARY
    )
    parser.add_argument("--chexbert-cache", type=Path, default=DEFAULT_CHEX_CACHE)
    parser.add_argument("--random-radgraph", type=Path, default=DEFAULT_RANDOM_RADGRAPH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chexbert-batch-size", type=int, default=128)
    parser.add_argument("--radgraph-model", default="modern-radgraph-xl")
    parser.add_argument("--radgraph-batch-size", type=int, default=8)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=7132)
    args = parser.parse_args()

    frozen_rows = read_jsonl(args.frozen_rows)
    random_rows = read_jsonl(args.random_rows)
    frozen_summary = read_json(args.frozen_qa_summary)
    random_summary = read_json(args.random_generation_summary)
    validate_sources(
        frozen_rows,
        random_rows,
        frozen_rows_hash=file_sha256(args.frozen_rows),
        random_rows_hash=file_sha256(args.random_rows),
        frozen_summary=frozen_summary,
        random_summary=random_summary,
    )
    selected_frozen = [
        dict(row) for row in frozen_rows if row["system"] in {"g0_target_image", "g2_hierarchical"}
    ]
    combined_for_labels = [*selected_frozen, *map(dict, random_rows)]
    checkpoint = resolve_checkpoint()
    labels_by_hash = label_unique_texts(
        combined_for_labels,
        cache_path=args.chexbert_cache,
        checkpoint_hash=file_sha256(checkpoint),
        device=args.device,
        batch_size=args.chexbert_batch_size,
    )
    for row in combined_for_labels:
        row["reference_labels"] = np.asarray(
            labels_by_hash[text_sha256(str(row.get("reference_answer") or ""))], dtype=np.int8
        )
        row["prediction_labels"] = np.asarray(
            labels_by_hash[text_sha256(str(row.get("answer") or ""))], dtype=np.int8
        )

    frozen_radgraph_summary = read_json(args.frozen_radgraph_summary)
    if (
        frozen_radgraph_summary.get("source_rows_sha256") != file_sha256(args.frozen_rows)
        or frozen_radgraph_summary["per_row_output"]["sha256"]
        != file_sha256(args.frozen_radgraph)
    ):
        raise RuntimeError("Frozen V10 RadGraph metrics differ from their summary")
    frozen_radgraph = {
        (str(row["system"]), str(row["case_id"]), str(row["question_type"])): {
            metric: float(row[metric]) for metric in RADGRAPH_METRICS
        }
        for row in read_csv(args.frozen_radgraph)
    }
    random_radgraph = score_random_radgraph(
        random_rows,
        output_path=args.random_radgraph,
        model_type=args.radgraph_model,
        batch_size=args.radgraph_batch_size,
        cuda=args.cuda,
    )
    frozen_by_key = {
        (str(row["system"]), str(row["case_id"]), str(row["question_type"])): row
        for row in selected_frozen
    }
    random_by_key = {
        (str(row["case_id"]), str(row["question_type"]), int(row["assignment"])): dict(row)
        for row in random_rows
    }
    for key, values in frozen_radgraph.items():
        if key in frozen_by_key:
            frozen_by_key[key].update(values)
    for key, values in random_radgraph.items():
        random_by_key[key].update(values)
    selected_frozen = list(frozen_by_key.values())
    random_rows_enriched = list(random_by_key.values())

    scopes = {}
    for scope_index, scope in enumerate(("all", *QUESTION_TYPES)):
        frozen_scope = select_scope(selected_frozen, scope)
        random_scope = select_scope(random_rows_enriched, scope)
        g0 = [row for row in frozen_scope if row["system"] == "g0_target_image"]
        g2 = [row for row in frozen_scope if row["system"] == "g2_hierarchical"]
        linear_metrics = ("token_f1", *RADGRAPH_METRICS)
        linear_systems = {}
        linear_comparisons = {}
        for metric_index, metric in enumerate(linear_metrics):
            g0_values = grouped_linear_values(g0, metric, assignment_mean=False)
            g2_values = grouped_linear_values(g2, metric, assignment_mean=False)
            random_values = grouped_linear_values(random_scope, metric, assignment_mean=True)
            linear_systems[metric] = {
                "g0_target_image": statistics.fmean(g0_values.values()),
                "g2_selected_history": statistics.fmean(g2_values.values()),
                "gr_random_history_assignment_mean": statistics.fmean(random_values.values()),
            }
            linear_comparisons[metric] = {
                "g2_minus_gr": paired_linear_bootstrap(
                    g2_values,
                    random_values,
                    iterations=args.bootstrap_iterations,
                    seed=args.bootstrap_seed + scope_index * 100 + metric_index,
                ),
                "gr_minus_g0": paired_linear_bootstrap(
                    random_values,
                    g0_values,
                    iterations=args.bootstrap_iterations,
                    seed=args.bootstrap_seed + 20 + scope_index * 100 + metric_index,
                ),
                "g2_minus_g0": paired_linear_bootstrap(
                    g2_values,
                    g0_values,
                    iterations=args.bootstrap_iterations,
                    seed=args.bootstrap_seed + 40 + scope_index * 100 + metric_index,
                ),
            }

        def chex_stats(rows: Sequence[Mapping[str, Any]]):
            return build_case_statistics(
                [str(row["case_id"]) for row in rows],
                np.stack([row["reference_labels"] for row in rows]),
                np.stack([row["prediction_labels"] for row in rows]),
            )

        g0_stats = chex_stats(g0)
        g2_stats = chex_stats(g2)
        random_stats = [
            chex_stats([row for row in random_scope if int(row["assignment"]) == assignment])
            for assignment in ASSIGNMENTS
        ]
        g0_chex = metrics_from_case_statistics(g0_stats)
        g2_chex = metrics_from_case_statistics(g2_stats)
        random_chex = [metrics_from_case_statistics(values) for values in random_stats]
        chexbert_systems = {
            metric: {
                "g0_target_image": float(g0_chex[metric]),
                "g2_selected_history": float(g2_chex[metric]),
                "gr_random_history_assignment_mean": float(
                    np.mean([float(values[metric]) for values in random_chex])
                ),
            }
            for metric in CHEXBERT_METRICS
        }
        chexbert_comparisons = {
            "g2_minus_gr": random_control_case_bootstrap(
                g2_stats,
                random_stats,
                iterations=args.bootstrap_iterations,
                seed=args.bootstrap_seed + 60 + scope_index * 100,
            )
        }
        scopes[scope] = {
            "linear_metrics": linear_systems,
            "linear_comparisons": linear_comparisons,
            "chexbert_metrics": chexbert_systems,
            "chexbert_comparisons": chexbert_comparisons,
        }

    integrity = {
        "answer_contract_valid_rate": statistics.fmean(
            float(row["assembled_schema_valid"]) for row in random_rows
        ),
        "citation_valid_rate": statistics.fmean(
            float(row["citation_valid"]) for row in random_rows
        ),
        "answer_token_ceiling_rate": statistics.fmean(
            float(row["hit_answer_token_ceiling"]) for row in random_rows
        ),
        "mean_input_tokens": statistics.fmean(
            float(row["answer_input_tokens"]) for row in random_rows
        ),
        "mean_output_tokens": statistics.fmean(
            float(row["answer_output_tokens"]) for row in random_rows
        ),
    }
    summary = {
        "study": "V10 post-hoc random-history negative control",
        "status": "posthoc_control_evaluation_complete_no_retuning",
        "protocol_commit": PROTOCOL_COMMIT,
        "counts": {
            "cases": 568,
            "questions_per_case": 2,
            "random_assignments": len(ASSIGNMENTS),
            "random_rows": len(random_rows),
        },
        "scopes": scopes,
        "random_history_integrity": integrity,
        "runtime": {
            "bootstrap_iterations": args.bootstrap_iterations,
            "bootstrap_seed": args.bootstrap_seed,
            "radgraph_model": args.radgraph_model,
            "f1chexbert_version": "0.0.2",
            "f1chexbert_checkpoint_sha256": file_sha256(checkpoint),
        },
        "artifacts": {
            "frozen_rows_sha256": file_sha256(args.frozen_rows),
            "random_rows_sha256": file_sha256(args.random_rows),
            "random_generation_summary_sha256": file_sha256(args.random_generation_summary),
            "random_radgraph_rows_sha256": file_sha256(args.random_radgraph),
            "chexbert_cache_sha256": file_sha256(args.chexbert_cache),
            "script_sha256": file_sha256(Path(__file__)),
        },
        "claim_boundary": (
            "Post-hoc automated negative control; not physician-adjudicated diagnostic accuracy, "
            "clinical utility, safety, or patient benefit."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
