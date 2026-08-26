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

from audit_v6_development_confirmation_separation import file_sha256, read_json  # noqa: E402
from medical_rag.evaluation.graded_retrieval import ndcg_at_k  # noqa: E402
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.relevance import (  # noqa: E402
    active_label_similarity,
    active_label_weights,
    radgraph_fact_similarity,
)


DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v10" / "v10_cluster_disjoint_split.json"
DEFAULT_ROWS = ROOT / "experiments" / "v10_publication" / "v10_confirmation_retrieval_rows.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "splits" / "v10" / "v10_qrel_sensitivity_summary.json"
DEFAULT_BOOTSTRAP = 10_000
VARIANT_WEIGHTS = {
    "combined_0.6_label_0.4_fact": (0.60, 0.40),
    "label_only": (1.00, 0.00),
    "fact_only": (0.00, 1.00),
}
SYSTEMS = (
    "r0_bm25",
    "r1_image_image",
    "r2_image_report",
    "r4_nine_feature",
    "r5_fact_attention",
)
DECLARED_DEVELOPMENT_SYSTEMS = SYSTEMS + ("r3_fixed_multimodal",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit frozen V10 rankings under qrel and spectrum sensitivities."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=7051)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Row {line_number} is not an object.")
                rows.append(value)
    return rows


def report_index_class(case: Any) -> str:
    return str(case.metadata.get("report_index_class", "unknown"))


def qrel_components(query: Any, bank: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    label_gains = np.asarray(
        [active_label_similarity(query.labels, candidate.labels) for candidate in bank],
        dtype=np.float32,
    )
    fact_gains = np.asarray(
        [radgraph_fact_similarity(query.radgraph_facts, candidate.radgraph_facts) for candidate in bank],
        dtype=np.float32,
    )
    return label_gains, fact_gains


def bootstrap_ci(values: Sequence[float], *, iterations: int, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(iterations, array.size))
    samples = array[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def summarize_difference(
    case_values: Mapping[str, Mapping[str, float]],
    *,
    systems: tuple[str, str],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    case_ids = sorted(case_values)
    differences = np.asarray(
        [case_values[case_id][systems[0]] - case_values[case_id][systems[1]] for case_id in case_ids],
        dtype=np.float64,
    )
    ci = bootstrap_ci(differences, iterations=iterations, seed=seed)
    return {
        "n_cases": len(case_ids),
        "difference": float(differences.mean()) if differences.size else float("nan"),
        "ci95_case_bootstrap": list(ci),
        "interpretation": (
            "confirmed_positive_under_this_qrel"
            if ci[0] > 0
            else "numerical_only_under_this_qrel"
            if differences.size and differences.mean() > 0
            else "no_positive_difference_under_this_qrel"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_iterations <= 0:
        raise ValueError("--bootstrap-iterations must be positive.")

    split = read_json(args.split)
    cases = {
        case.study_id: case
        for case in read_openi_paired_cases(args.cases, source_unique_patient=True, radgraph_path=args.radgraph)
    }
    train_ids = [str(case_id) for case_id in split["partitions"]["train"]["case_ids"]]
    test_ids = [str(case_id) for case_id in split["partitions"]["test"]["case_ids"]]
    if len(set(train_ids) & set(test_ids)) != 0:
        raise RuntimeError("V10 Train/Test case overlap detected.")
    bank = [cases[case_id] for case_id in train_ids]
    bank_ids = set(train_ids)

    rows = read_rows(args.rows)
    expected_keys: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["case_id"]), str(row["question_type"]), str(row["system"]))
        if key in expected_keys:
            raise RuntimeError(f"Duplicate frozen ranking row: {key}")
        top_ids = [str(value) for value in row["top_case_ids"]]
        if len(top_ids) != 10 or len(set(top_ids)) != 10:
            raise RuntimeError(f"Ranking row is not a unique Top-10: {key}")
        if not set(top_ids).issubset(bank_ids):
            raise RuntimeError(f"Ranking row contains a non-Train candidate: {key}")
        expected_keys[key] = row

    observed_systems = sorted({key[2] for key in expected_keys})
    missing = [system for system in DECLARED_DEVELOPMENT_SYSTEMS if system not in observed_systems]
    present_test_ids = sorted({key[0] for key in expected_keys})
    question_types = sorted({key[1] for key in expected_keys})
    missing_test_ids = sorted(set(test_ids) - set(present_test_ids))
    unexpected_test_ids = sorted(set(present_test_ids) - set(test_ids))
    if unexpected_test_ids:
        raise RuntimeError(f"Frozen ranking rows contain non-Test cases: {unexpected_test_ids}")
    if missing:
        # R3 is documented as a development system, but was not part of the frozen
        # confirmation runner. Preserve that fact in the output instead of inventing rows.
        unexpected_missing = [system for system in missing if system != "r3_fixed_multimodal"]
        if unexpected_missing:
            raise RuntimeError(f"Unexpected missing systems: {unexpected_missing}")

    per_variant_system_case: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    qrel_structure: dict[str, dict[str, Any]] = {}
    evaluation_test_ids = present_test_ids
    case_classes = {case_id: report_index_class(cases[case_id]) for case_id in evaluation_test_ids}

    for case_id in evaluation_test_ids:
        query = cases[case_id]
        label_gains, fact_gains = qrel_components(query, bank)
        qrel_structure[case_id] = {
            "report_index_class": case_classes[case_id],
            "label_empty": not bool(active_label_weights(query.labels)),
            "fact_empty": not bool(query.radgraph_facts),
            "label_mean": float(label_gains.mean()),
            "fact_mean": float(fact_gains.mean()),
            "combined_mean": float((0.60 * label_gains + 0.40 * fact_gains).mean()),
            "combined_candidates_gain_ge_0.5": int(np.count_nonzero(0.60 * label_gains + 0.40 * fact_gains >= 0.50)),
            "label_candidates_gain_ge_0.5": int(np.count_nonzero(label_gains >= 0.50)),
            "fact_candidates_gain_ge_0.5": int(np.count_nonzero(fact_gains >= 0.50)),
        }
        qrels_by_variant = {
            name: dict(
                zip(
                    train_ids,
                    map(float, label_weight * label_gains + fact_weight * fact_gains),
                    strict=True,
                )
            )
            for name, (label_weight, fact_weight) in VARIANT_WEIGHTS.items()
        }
        for question_type in question_types:
            for system in observed_systems:
                row = expected_keys.get((case_id, question_type, system))
                if row is None:
                    raise RuntimeError(f"Missing ranking row: {(case_id, question_type, system)}")
                ranking = [str(value) for value in row["top_case_ids"]]
                for variant, qrels in qrels_by_variant.items():
                    per_variant_system_case[variant][system][case_id].append(
                        ndcg_at_k(qrels, ranking, 10)
                    )

    metrics: dict[str, Any] = {}
    for variant in VARIANT_WEIGHTS:
        system_metrics: dict[str, Any] = {}
        case_means: dict[str, dict[str, float]] = {}
        for system in observed_systems:
            values_by_case = per_variant_system_case[variant][system]
            case_means[system] = {
                case_id: float(statistics.fmean(values))
                for case_id, values in values_by_case.items()
            }
            all_values = list(case_means[system].values())
            by_class = {
                class_name: [
                    value
                    for case_id, value in case_means[system].items()
                    if case_classes[case_id] == class_name
                ]
                for class_name in ("normal", "abnormal", "indeterminate")
            }
            system_metrics[system] = {
                "n_cases": len(all_values),
                "ndcg@10": float(statistics.fmean(all_values)),
                "by_report_index_class": {
                    class_name: {
                        "n_cases": len(values),
                        "ndcg@10": float(statistics.fmean(values)) if values else None,
                    }
                    for class_name, values in by_class.items()
                },
            }
        comparison = summarize_difference(
            {case_id: {system: case_means[system][case_id] for system in ("r5_fact_attention", "r4_nine_feature")}
             for case_id in evaluation_test_ids},
            systems=("r5_fact_attention", "r4_nine_feature"),
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        subgroup_comparisons = {}
        for class_name in ("normal", "abnormal", "indeterminate"):
            subset = {
                case_id: {system: case_means[system][case_id] for system in ("r5_fact_attention", "r4_nine_feature")}
                for case_id in evaluation_test_ids
                if case_classes[case_id] == class_name
            }
            subgroup_comparisons[class_name] = summarize_difference(
                subset,
                systems=("r5_fact_attention", "r4_nine_feature"),
                iterations=args.bootstrap_iterations,
                seed=args.seed + len(subgroup_comparisons) + 1,
            )
        metrics[variant] = {
            "systems": system_metrics,
            "r5_minus_r4": comparison,
            "r5_minus_r4_by_report_index_class": subgroup_comparisons,
        }

    class_counts = {
        class_name: sum(value == class_name for value in case_classes.values())
        for class_name in ("normal", "abnormal", "indeterminate")
    }
    output = {
        "study": "V10 post-hoc qrel construct and spectrum sensitivity audit",
        "status": "post_hoc_exploratory_complete",
        "frozen_v10_results_unchanged": True,
        "test_case_count": len(test_ids),
        "evaluated_case_count": len(evaluation_test_ids),
        "excluded_case_ids_from_frozen_ranking_rows": missing_test_ids,
        "candidate_bank_count": len(bank),
        "question_types": question_types,
        "observed_systems": observed_systems,
        "development_system_not_in_confirmation_rows": missing,
        "metrics": metrics,
        "report_index_class_counts": class_counts,
        "qrel_variants": {
            name: {"active_label_weight": weights[0], "radgraph_fact_weight": weights[1]}
            for name, weights in VARIANT_WEIGHTS.items()
        },
        "qrel_structure_by_class": {
            class_name: {
                "n_cases": sum(value["report_index_class"] == class_name for value in qrel_structure.values()),
                "label_empty_cases": sum(
                    value["report_index_class"] == class_name and value["label_empty"]
                    for value in qrel_structure.values()
                ),
                "fact_empty_cases": sum(
                    value["report_index_class"] == class_name and value["fact_empty"]
                    for value in qrel_structure.values()
                ),
                "mean_label_gain": float(statistics.fmean(
                    value["label_mean"] for value in qrel_structure.values()
                    if value["report_index_class"] == class_name
                )) if class_counts[class_name] else None,
                "mean_fact_gain": float(statistics.fmean(
                    value["fact_mean"] for value in qrel_structure.values()
                    if value["report_index_class"] == class_name
                )) if class_counts[class_name] else None,
                "mean_combined_gain": float(statistics.fmean(
                    value["combined_mean"] for value in qrel_structure.values()
                    if value["report_index_class"] == class_name
                )) if class_counts[class_name] else None,
                "mean_candidates_with_combined_gain_ge_0.5": float(statistics.fmean(
                    value["combined_candidates_gain_ge_0.5"] for value in qrel_structure.values()
                    if value["report_index_class"] == class_name
                )) if class_counts[class_name] else None,
            }
            for class_name in ("normal", "abnormal", "indeterminate")
        },
        "bootstrap": {"iterations": args.bootstrap_iterations, "seed": args.seed},
        "input_sha256": {
            "cases": file_sha256(args.cases),
            "radgraph": file_sha256(args.radgraph),
            "split": file_sha256(args.split),
            "rows": file_sha256(args.rows),
        },
        "claim_boundary": (
            "This is a post-hoc sensitivity audit of frozen rankings. It does not alter "
            "the V10 combined-qrel result, establish physician-adjudicated relevance, "
            "or establish clinical correctness."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
