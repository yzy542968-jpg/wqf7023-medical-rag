from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_multimodal_v4_retrieval import (
    candidate_case_ids,
    eligible_cases,
    image_lookup,
    load_json,
    sha256,
    verify_committed_selection,
    verify_preregistered_config,
    write_json,
    write_jsonl,
)
from run_multimodal_v41_retrieval import build_or_load_embeddings

from medical_rag.multimodal.evaluation import (
    build_text_query,
    evaluate_rankings_and_answers,
)
from medical_rag.multimodal.fusion import rank_scores, shortlist_score_fusion
from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def verify_encoder_compatibility(v42: dict[str, Any], v41: dict[str, Any]) -> None:
    pairs = {
        "joint_encoder": "joint_encoder",
        "text_model_revision": "text_model_revision",
        "image_weights_md5": "image_weights_md5",
        "joint_embedding_dimension": "joint_embedding_dimension",
        "text_max_length": "text_max_length",
    }
    for v42_key, v41_key in pairs.items():
        if v42["encoder"][v42_key] != v41["retrieval"][v41_key]:
            raise RuntimeError(f"V4.2 encoder field {v42_key} differs from the cached V4.1 encoder.")


def build_v42_rankings(
    questions: list[dict[str, Any]],
    candidate_ids: list[str],
    cases: dict[str, dict[str, Any]],
    image_embeddings: np.ndarray,
    report_embeddings: np.ndarray,
    shortlist_size: int,
    text_weight: float,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, list[str]]]:
    bm25 = BM25Retriever().fit([cases[case_id] for case_id in candidate_ids])
    case_index = {case_id: index for index, case_id in enumerate(candidate_ids)}
    image_score_by_source: dict[str, dict[str, float]] = {}
    image_ranking_by_source: dict[str, list[str]] = {}
    for source_case_id, source_index in case_index.items():
        scores = report_embeddings @ image_embeddings[source_index]
        score_map = {
            candidate_id: float(scores[candidate_index])
            for candidate_id, candidate_index in case_index.items()
        }
        image_score_by_source[source_case_id] = score_map
        image_ranking_by_source[source_case_id] = rank_scores(candidate_ids, scores.tolist())

    text_rankings = {}
    image_rankings = {}
    paired_rankings = {}
    for question in questions:
        qid = str(question["qid"])
        source_case_id = str(question["case_id"])
        rows = bm25.search(
            build_text_query(cases[source_case_id], question),
            top_k=len(candidate_ids),
        )
        text_ranking = [str(row["case_id"]) for row in rows]
        text_scores = [float(row["score"]) for row in rows]
        text_rankings[qid] = text_ranking
        image_rankings[qid] = image_ranking_by_source[source_case_id]
        paired_rankings[qid] = shortlist_score_fusion(
            text_ranking,
            text_scores,
            image_score_by_source[source_case_id],
            shortlist_size=shortlist_size,
            text_weight=text_weight,
        )
    return text_rankings, image_rankings, paired_rankings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preregistered BioViL-T V4.2 shortlist reranking.")
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "multimodal_v42.json")
    parser.add_argument("--v41-config", type=Path, default=ROOT / "config" / "multimodal_v41.json")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "raw" / "openi_official_images")
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data" / "processed" / "multimodal_v41_biovil_t_embeddings.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "post_submission_v42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--prereg-commit", default="5846649")
    parser.add_argument("--development-commit")
    args = parser.parse_args()

    verify_preregistered_config(args.prereg_commit, args.config)
    config = load_json(args.config)
    v41_config = load_json(args.v41_config)
    verify_encoder_compatibility(config, v41_config)
    all_cases = {
        str(row["case_id"]): row
        for row in load_cases_jsonl(ROOT / config["source"]["cases_path"])
    }
    requested = candidate_case_ids(config)
    images = image_lookup(args.image_root)
    eligible, case_images, exclusions = eligible_cases(requested, all_cases, images)

    benchmark = load_json(ROOT / config["cohorts"][args.split]["benchmark_path"])
    eligible_set = set(eligible)
    questions = [row for row in benchmark["questions"] if str(row["case_id"]) in eligible_set]
    split_case_count = len({str(row["case_id"]) for row in questions})
    if split_case_count != int(config["cohorts"][args.split]["case_count"]):
        raise RuntimeError("Eligible split case count differs from the preregistered count.")

    development_path = args.output_dir / "development_retrieval_summary.json"
    if args.split == "confirmation":
        if not args.development_commit:
            raise ValueError("--development-commit is required for confirmation evaluation.")
        verify_committed_selection(args.development_commit, development_path)
        development = load_json(development_path)
        if not development["development_gate"]["passed"]:
            raise RuntimeError("The preregistered V4.2 development gate did not pass.")

    image_embeddings, report_embeddings, runtime = build_or_load_embeddings(
        args.cache,
        eligible,
        all_cases,
        case_images,
        v41_config,
        args.device,
        args.image_batch_size,
        args.text_batch_size,
    )
    text_rankings, image_rankings, paired_rankings = build_v42_rankings(
        questions,
        eligible,
        all_cases,
        image_embeddings,
        report_embeddings,
        shortlist_size=int(config["reranking"]["shortlist_size"]),
        text_weight=float(config["reranking"]["text_weight"]),
    )
    systems = {
        "report_only_bm25": text_rankings,
        "image_only_biovil_t": image_rankings,
        "paired_biovil_t_shortlist_reranker": paired_rankings,
    }
    metrics = {}
    output_rows = []
    for name, rankings in systems.items():
        system_metrics, rows = evaluate_rankings_and_answers(questions, rankings, all_cases)
        metrics[name] = system_metrics
        output_rows.extend({"system": name, **row} for row in rows)

    gate_passed = (
        metrics["paired_biovil_t_shortlist_reranker"]["mrr"]
        > metrics["report_only_bm25"]["mrr"]
    )
    summary = {
        "experiment": config["experiment"],
        "split": args.split,
        "preregistration_commit": args.prereg_commit,
        "development_commit": args.development_commit,
        "config_sha256": sha256(args.config),
        "candidate_case_count": len(eligible),
        "split_case_count": split_case_count,
        "question_count": len(questions),
        "excluded_cases": exclusions,
        "fixed_reranking_policy": config["reranking"],
        "metrics": metrics,
        "development_gate": {
            "passed": gate_passed,
            "paired_mrr_exceeds_report_only": gate_passed,
        },
        "runtime": runtime,
    }
    write_json(args.output_dir / f"{args.split}_retrieval_summary.json", summary)
    write_jsonl(args.output_dir / f"{args.split}_retrieval_rows.jsonl", output_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
