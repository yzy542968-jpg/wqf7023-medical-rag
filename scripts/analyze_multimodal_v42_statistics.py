from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_grouped_statistical_analysis import holm_adjust, paired_grouped_bootstrap
from run_multimodal_v4_retrieval import (
    candidate_case_ids,
    load_json,
    verify_committed_selection,
    write_json,
)
from run_multimodal_v42_retrieval import build_v42_rankings

from medical_rag.evaluation.answer_metrics import token_f1
from medical_rag.multimodal.evaluation import routed_extractive_answer
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def question_metric_values(
    question: Mapping[str, Any],
    ranking: Sequence[str],
    cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    target = str(question["case_id"])
    rank = ranking.index(target) + 1
    prediction = routed_extractive_answer(cases[ranking[0]], question)
    return {
        "mrr": 1.0 / rank,
        "hit@1": float(rank <= 1),
        "hit@5": float(rank <= 5),
        "hit@10": float(rank <= 10),
        "token_f1": token_f1(prediction, str(question["reference_answer"])),
    }


def grouped_values(
    questions: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[str]],
    cases: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = {
        metric: defaultdict(list) for metric in ("mrr", "hit@1", "hit@5", "hit@10", "token_f1")
    }
    for question in questions:
        case_id = str(question["case_id"])
        values = question_metric_values(question, rankings[str(question["qid"])], cases)
        for metric, value in values.items():
            grouped[metric][case_id].append(value)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Case-level paired bootstrap for V4.2 confirmation.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "multimodal_v42.json")
    parser.add_argument(
        "--confirmation-summary",
        type=Path,
        default=ROOT / "experiments" / "post_submission_v42" / "confirmation_retrieval_summary.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data" / "processed" / "multimodal_v41_biovil_t_embeddings.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "post_submission_v42" / "confirmation_statistics.json",
    )
    parser.add_argument("--confirmation-commit", default="9bb6bf7")
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7023)
    args = parser.parse_args()

    verify_committed_selection(args.confirmation_commit, args.confirmation_summary)
    config = load_json(args.config)
    summary = load_json(args.confirmation_summary)
    cases = {
        str(row["case_id"]): row
        for row in load_cases_jsonl(ROOT / config["source"]["cases_path"])
    }
    candidate_ids = candidate_case_ids(config)
    cache = np.load(args.cache, allow_pickle=False)
    if cache["case_ids"].tolist() != candidate_ids:
        raise RuntimeError("Cached BioViL-T case IDs differ from the registered candidate pool.")
    if cache["image_embeddings"].shape[1] != int(config["encoder"]["joint_embedding_dimension"]):
        raise RuntimeError("Cached BioViL-T embedding dimension differs from the registered model.")

    benchmark = load_json(ROOT / config["cohorts"]["confirmation"]["benchmark_path"])
    questions = benchmark["questions"]
    report_rankings, _, paired_rankings = build_v42_rankings(
        questions,
        candidate_ids,
        cases,
        cache["image_embeddings"],
        cache["report_embeddings"],
        shortlist_size=int(config["reranking"]["shortlist_size"]),
        text_weight=float(config["reranking"]["text_weight"]),
    )
    report_values = grouped_values(questions, report_rankings, cases)
    paired_values = grouped_values(questions, paired_rankings, cases)

    comparisons = {}
    metric_order = ["mrr", "hit@1", "hit@5", "hit@10", "token_f1"]
    for metric in metric_order:
        comparisons[metric] = paired_grouped_bootstrap(
            paired_values[metric],
            report_values[metric],
            iterations=args.resamples,
            seed=args.seed,
        )
        observed = comparisons[metric]["mean_difference"]
        expected = (
            summary["metrics"]["paired_biovil_t_shortlist_reranker"][metric]
            - summary["metrics"]["report_only_bm25"][metric]
        )
        if abs(observed - expected) > 1e-12:
            raise RuntimeError(f"Reconstructed {metric} difference does not match the locked summary.")

    adjusted = holm_adjust([comparisons[metric]["paired_randomization_p"] for metric in metric_order])
    for metric, adjusted_p in zip(metric_order, adjusted, strict=True):
        comparisons[metric]["holm_adjusted_p"] = adjusted_p
        comparisons[metric]["bootstrap_probability_positive"] = 1.0 - (
            comparisons[metric]["two_sided_bootstrap_p"] / 2.0
        )

    result = {
        "experiment": config["experiment"],
        "split": "confirmation",
        "confirmation_result_commit": args.confirmation_commit,
        "unit": "case_id",
        "case_count": int(config["cohorts"]["confirmation"]["case_count"]),
        "question_count": int(config["cohorts"]["confirmation"]["question_count"]),
        "resamples": args.resamples,
        "seed": args.seed,
        "difference_direction": "paired_biovil_t_shortlist_reranker minus report_only_bm25",
        "primary_metric": "mrr",
        "comparisons": comparisons,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
