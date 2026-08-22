from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.evaluation.case_scoped_benchmark import is_clean_eligible_case
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl
from run_grouped_statistical_analysis import paired_grouped_bootstrap
from run_multimodal_v5_retrieval import derangement_indices
from build_multimodal_v5_cohort import case_ids_from_payload


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def grouped_metric(rows: list[dict], metric: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(float(row[metric]))
    return dict(grouped)


def question_type(qid: str) -> str:
    if qid.endswith("_findings"):
        return "findings"
    if qid.endswith("_impression"):
        return "impression"
    if qid.endswith("_summary"):
        return "summary"
    return "unknown"


def qtype_means(rows: list[dict], metrics: tuple[str, ...]) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        qtype = str(row.get("question_type") or question_type(str(row["qid"])))
        for metric in metrics:
            values[qtype][metric].append(float(row[metric]))
    return {
        qtype: {metric: mean(metric_values) for metric, metric_values in metric_map.items()}
        for qtype, metric_map in values.items()
    }


def paired_by_qtype(
    first_rows: list[dict],
    second_rows: list[dict],
    metric: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    first_by_type: dict[str, list[dict]] = defaultdict(list)
    second_by_type: dict[str, list[dict]] = defaultdict(list)
    for row in first_rows:
        first_by_type[str(row.get("question_type") or question_type(str(row["qid"])))].append(row)
    for row in second_rows:
        second_by_type[str(row.get("question_type") or question_type(str(row["qid"])))].append(row)
    for offset, qtype in enumerate(sorted(first_by_type)):
        output[qtype] = paired_grouped_bootstrap(
            grouped_metric(first_by_type[qtype], metric),
            grouped_metric(second_by_type[qtype], metric),
            iterations=iterations,
            seed=seed + offset,
        )
    return output


def build_cohort_profile(cohort: dict, config: dict, source_cases: list[dict]) -> dict:
    excluded: set[str] = set()
    for relative in config["cohort"]["excluded_source_manifests"]:
        excluded.update(case_ids_from_payload(read_json(ROOT / relative)))
    eligible_all = [case for case in source_cases if is_clean_eligible_case(case)]
    eligible_fresh = [
        case
        for case in eligible_all
        if str(case["case_id"]) not in excluded
    ]
    fields = sorted(source_cases[0]) if source_cases else []
    return {
        "source_case_count": len(source_cases),
        "prior_excluded_case_count": len(excluded),
        "eligible_before_prior_exclusions": len(eligible_all),
        "fresh_eligible_case_count": len(eligible_fresh),
        "selected_case_count": int(cohort["case_count"]),
        "development_case_count": len(cohort["split"]["development"]["case_ids"]),
        "confirmation_case_count": len(cohort["split"]["confirmation"]["case_ids"]),
        "source_fields": fields,
        "patient_identifier_fields_available": [
            field for field in fields if "patient" in field.lower() or "subject" in field.lower()
        ],
        "eligibility_rule": {
            "requires_image": True,
            "minimum_findings_characters": 40,
            "minimum_impression_characters": 8,
            "maximum_indication_placeholder_ratio": 0.5,
            "excludes_empty_or_normal_problems": True,
        },
    }


def build_reference_profile(cohort: dict) -> dict:
    confirmation_ids = set(cohort["split"]["confirmation"]["case_ids"])
    questions = [row for row in cohort["questions"] if str(row["case_id"]) in confirmation_ids]
    by_case: dict[str, dict[str, str]] = defaultdict(dict)
    for row in questions:
        by_case[str(row["case_id"])][str(row["question_type"])] = str(row["reference_answer"])
    duplicated = sum(
        values.get("case_scoped_impression") == values.get("case_scoped_summary")
        for values in by_case.values()
    )
    return {
        "confirmation_question_count": len(questions),
        "question_type_counts": dict(Counter(str(row["question_type"]) for row in questions)),
        "unique_reference_strings": len({str(row["reference_answer"]) for row in questions}),
        "cases_with_identical_impression_and_summary_reference": duplicated,
        "metric_weighting_warning": "Impression content is used twice per case: once as impression reference and once as summary reference.",
    }


def build_shortlist_profile(rows: list[dict]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    for system in sorted({str(row["system"]) for row in rows}):
        system_rows = [row for row in rows if str(row["system"]) == system]
        hits = sum(int(row["target_rank"]) <= 100 for row in system_rows)
        output[system] = {
            "question_count": len(system_rows),
            "target_recall_at_100": hits / len(system_rows) if system_rows else 0.0,
            "target_outside_top_100_count": len(system_rows) - hits,
        }
    return output


def build_derangement_profile(config: dict, confirmation_count: int) -> dict[str, int]:
    count = int(config["random_image_control"]["permutations"])
    seed = int(config["random_image_control"]["seed"])
    permutations = {
        tuple(derangement_indices(confirmation_count, seed + index).tolist())
        for index in range(count)
    }
    return {"configured_permutations": count, "unique_permutations": len(permutations)}


def build_cost_profile(config: dict, retrieval_rows: list[dict], qa_rows: list[dict]) -> dict[str, int]:
    question_count = len(qa_rows)
    permutation_count = int(config["random_image_control"]["permutations"])
    return {
        "confirmation_questions": question_count,
        "logical_bm25_query_calls": question_count * 2,
        "deterministic_retrieval_condition_rows": len(retrieval_rows),
        "correct_image_reranking_rows": question_count * 2,
        "shuffled_image_reranking_rows": question_count * permutation_count,
        "generation_records_report_only_and_multimodal": question_count * 2,
        "semantic_checker_records_report_only_and_multimodal": question_count * 2,
        "candidate_pool_cases": 240,
        "image_reranking_shortlist_size": int(config["reranking"]["shortlist_size"]),
    }


def holm_adjusted_pvalues(pvalues: dict[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted values for one exploratory family."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    count = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (count - index) * float(value)))
        adjusted[name] = running_max
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze frozen V5 outputs without changing the primary result artifact."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/post_submission_v5/v5_supplemental_analysis.json",
    )
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7023)
    args = parser.parse_args()

    cohort = read_json(ROOT / "data/processed/openi_multimodal_v5_cohort.json")
    config = read_json(ROOT / "config/multimodal_v5.json")
    source_cases = load_cases_jsonl(ROOT / "data/processed/openi_cases.jsonl")
    retrieval_rows = read_jsonl(ROOT / "experiments/post_submission_v5/confirmation_retrieval_rows.jsonl")
    report_rows = read_jsonl(ROOT / "experiments/post_submission_v5/qa_report_only/final_optimized_test_rows.jsonl")
    multimodal_rows = read_jsonl(ROOT / "experiments/post_submission_v5/qa_multimodal/final_optimized_test_rows.jsonl")

    qa_metrics: dict[str, dict[str, object]] = {}
    for metric, offset in (("final_token_f1", 0), ("support_rate", 100)):
        qa_metrics[metric] = paired_by_qtype(
            multimodal_rows,
            report_rows,
            metric,
            iterations=args.iterations,
            seed=args.seed + offset,
        )
    for metric, metric_results in qa_metrics.items():
        raw = {
            qtype: float(values["paired_randomization_p"])
            for qtype, values in metric_results.items()
        }
        adjusted = holm_adjusted_pvalues(raw)
        for qtype, value in adjusted.items():
            metric_results[qtype]["holm_adjusted_paired_randomization_p"] = value

    result = {
        "analysis": "v5_frozen_output_supplement",
        "status": "supplemental_descriptive_and_sensitivity_analysis",
        "primary_v5_results_modified": False,
        "inputs": {
            "cohort": "data/processed/openi_multimodal_v5_cohort.json",
            "retrieval_rows": "experiments/post_submission_v5/confirmation_retrieval_rows.jsonl",
            "report_only_qa_rows": "experiments/post_submission_v5/qa_report_only/final_optimized_test_rows.jsonl",
            "multimodal_qa_rows": "experiments/post_submission_v5/qa_multimodal/final_optimized_test_rows.jsonl",
        },
        "cohort_profile": build_cohort_profile(cohort, config, source_cases),
        "reference_profile": build_reference_profile(cohort),
        "retrieval_target_recall_at_100": build_shortlist_profile(retrieval_rows),
        "question_type_means": {
            "report_only": qtype_means(report_rows, ("final_token_f1", "support_rate")),
            "multimodal": qtype_means(multimodal_rows, ("final_token_f1", "support_rate")),
        },
        "question_type_paired_differences": qa_metrics,
        "random_image_control_integrity": build_derangement_profile(
            config, len(cohort["split"]["confirmation"]["case_ids"])
        ),
        "logical_cost_profile": build_cost_profile(config, retrieval_rows, report_rows),
        "interpretation_limits": [
            "Question-type analyses are secondary and exploratory; the frozen primary comparisons are unchanged.",
            "The shortlist diagnostic measures whether the target is available to the image reranker, not independent retrieval quality outside the fixed Top-100 design.",
            "The cohort profile describes selection rules and does not establish patient-level independence because no patient identifier is available in the processed source records.",
            "The duplicated impression and summary reference is disclosed rather than corrected post hoc.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
