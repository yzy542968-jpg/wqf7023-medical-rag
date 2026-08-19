from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_multimodal_v4_retrieval import eligible_cases, image_lookup, load_json, write_json, write_jsonl
from run_multimodal_v41_retrieval import build_or_load_embeddings

from medical_rag.multimodal.evaluation import build_text_query, evaluate_rankings_and_answers
from medical_rag.multimodal.fusion import rank_scores, shortlist_score_fusion
from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def query_for(case: dict[str, Any], question: dict[str, Any], use_indication: bool) -> str:
    if use_indication:
        return build_text_query(case, question)
    return str(question["question"])


def derangement_indices(size: int, seed: int) -> np.ndarray:
    if size < 2:
        raise ValueError("A random image control needs at least two cases.")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(size)
    for _ in range(size * 2):
        if not np.any(permutation == np.arange(size)):
            return permutation
        permutation = np.roll(rng.permutation(size), 1)
    raise RuntimeError("Could not construct a fixed-point-free image permutation.")


def build_text_rows(
    questions: list[dict[str, Any]],
    candidate_ids: list[str],
    cases: dict[str, dict[str, Any]],
    *,
    use_indication: bool,
) -> dict[str, list[dict[str, Any]]]:
    candidate_cases = [cases[case_id] for case_id in candidate_ids]
    bm25 = BM25Retriever().fit(candidate_cases)
    return {
        str(question["qid"]): bm25.search(
            query_for(cases[str(question["case_id"])], question, use_indication),
            top_k=len(candidate_ids),
        )
        for question in questions
    }


def rankings_from_text_rows(
    questions: list[dict[str, Any]],
    candidate_ids: list[str],
    image_embeddings: np.ndarray,
    report_embeddings: np.ndarray,
    text_rows_by_qid: dict[str, list[dict[str, Any]]],
    *,
    use_image: bool,
    image_source_indices: np.ndarray | None,
    shortlist_size: int,
    text_weight: float,
) -> dict[str, list[str]]:
    candidate_index = {case_id: index for index, case_id in enumerate(candidate_ids)}
    rankings: dict[str, list[str]] = {}
    for question in questions:
        source_id = str(question["case_id"])
        source_index = candidate_index[source_id]
        rows = text_rows_by_qid[str(question["qid"])]
        text_ranking = [str(row["case_id"]) for row in rows]
        if not use_image:
            rankings[str(question["qid"])] = text_ranking
            continue
        image_index = source_index if image_source_indices is None else int(image_source_indices[source_index])
        similarities = report_embeddings @ image_embeddings[image_index]
        image_scores = {
            case_id: float(similarities[index])
            for index, case_id in enumerate(candidate_ids)
        }
        rankings[str(question["qid"])] = shortlist_score_fusion(
            text_ranking,
            [float(row["score"]) for row in rows],
            image_scores,
            shortlist_size=shortlist_size,
            text_weight=text_weight,
        )
    return rankings


def build_rankings(
    questions: list[dict[str, Any]],
    candidate_ids: list[str],
    cases: dict[str, dict[str, Any]],
    image_embeddings: np.ndarray,
    report_embeddings: np.ndarray,
    *,
    use_indication: bool,
    use_image: bool,
    image_source_indices: np.ndarray | None,
    shortlist_size: int,
    text_weight: float,
) -> dict[str, list[str]]:
    text_rows = build_text_rows(
        questions,
        candidate_ids,
        cases,
        use_indication=use_indication,
    )
    return rankings_from_text_rows(
        questions,
        candidate_ids,
        image_embeddings,
        report_embeddings,
        text_rows,
        use_image=use_image,
        image_source_indices=image_source_indices,
        shortlist_size=shortlist_size,
        text_weight=text_weight,
    )


def metrics_for(
    questions: list[dict[str, Any]],
    rankings: dict[str, list[str]],
    cases: dict[str, dict[str, Any]],
) -> dict[str, float]:
    metrics, _ = evaluate_rankings_and_answers(questions, rankings, cases)
    return {key: float(value) for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run specified fresh-cohort V5 multimodal controls.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "multimodal_v5.json")
    parser.add_argument("--v41-config", type=Path, default=ROOT / "config" / "multimodal_v41.json")
    parser.add_argument("--cohort", type=Path, default=ROOT / "data" / "processed" / "openi_multimodal_v5_cohort.json")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "processed" / "openi_cases.jsonl")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "raw" / "openi_official_images")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "processed" / "multimodal_v5_biovil_t_embeddings.npz")
    parser.add_argument("--split", choices=["development", "confirmation"], default="confirmation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "post_submission_v5")
    args = parser.parse_args()

    config = load_json(args.config)
    cohort = load_json(args.cohort)
    cases = {str(row["case_id"]): row for row in load_cases_jsonl(args.cases)}
    candidate_ids = [str(value) for value in cohort["case_ids"]]
    images = image_lookup(args.image_root)
    eligible, case_images, exclusions = eligible_cases(candidate_ids, cases, images)
    if eligible != candidate_ids:
        raise RuntimeError(f"Fresh cohort image eligibility changed: {len(eligible)} / {len(candidate_ids)}")

    benchmark_questions = [
        row for row in cohort["questions"]
        if str(row["case_id"]) in set(cohort["split"][args.split]["case_ids"])
    ]
    image_embeddings, report_embeddings, runtime = build_or_load_embeddings(
        args.cache,
        candidate_ids,
        cases,
        case_images,
        load_json(args.v41_config),
        args.device,
        args.image_batch_size,
        args.text_batch_size,
    )
    shortlist_size = int(config["reranking"]["shortlist_size"])
    text_weight = float(config["reranking"]["text_weight"])
    systems = {
        "question_only_bm25": dict(use_indication=False, use_image=False, image_source_indices=None),
        "indication_question_bm25": dict(use_indication=True, use_image=False, image_source_indices=None),
        "question_only_correct_image": dict(use_indication=False, use_image=True, image_source_indices=None),
        "indication_question_correct_image": dict(use_indication=True, use_image=True, image_source_indices=None),
    }
    text_rows_by_condition = {
        use_indication: build_text_rows(
            benchmark_questions,
            candidate_ids,
            cases,
            use_indication=use_indication,
        )
        for use_indication in (False, True)
    }
    rankings_by_system = {
        name: rankings_from_text_rows(
            benchmark_questions,
            candidate_ids,
            image_embeddings,
            report_embeddings,
            text_rows_by_condition[spec["use_indication"]],
            use_image=spec["use_image"],
            image_source_indices=spec["image_source_indices"],
            shortlist_size=shortlist_size,
            text_weight=text_weight,
        )
        for name, spec in systems.items()
    }
    metrics = {
        name: metrics_for(benchmark_questions, rankings, cases)
        for name, rankings in rankings_by_system.items()
    }
    rows: list[dict[str, Any]] = []
    for name, rankings in rankings_by_system.items():
        _, system_rows = evaluate_rankings_and_answers(benchmark_questions, rankings, cases)
        for row in system_rows:
            ranking = rankings[str(row["qid"])]
            target_rank = ranking.index(str(row["case_id"])) + 1
            rows.append({
                "system": name,
                **row,
                "target_rank": target_rank,
                "mrr": 1.0 / target_rank,
                "hit@1": float(target_rank <= 1),
                "hit@5": float(target_rank <= 5),
                "hit@10": float(target_rank <= 10),
            })

    permutation_metrics = []
    permutation_count = int(config["random_image_control"]["permutations"])
    permutation_seed = int(config["random_image_control"]["seed"])
    for offset in range(permutation_count):
        permutation = derangement_indices(len(candidate_ids), permutation_seed + offset)
        shuffled_rankings = rankings_from_text_rows(
            benchmark_questions,
            candidate_ids,
            image_embeddings,
            report_embeddings,
            text_rows_by_condition[True],
            use_image=True,
            image_source_indices=permutation,
            shortlist_size=shortlist_size,
            text_weight=text_weight,
        )
        permutation_metrics.append(metrics_for(benchmark_questions, shuffled_rankings, cases))

    summary = {
        "experiment": config["experiment"],
        "split": args.split,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "cohort_sha256": hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
        "candidate_case_count": len(candidate_ids),
        "split_case_count": len({str(row["case_id"]) for row in benchmark_questions}),
        "question_count": len(benchmark_questions),
        "excluded_image_cases": exclusions,
        "fixed_policy": config["reranking"],
        "metrics": metrics,
        "random_image_control": {
            "permutations": permutation_count,
            "seed": permutation_seed,
            "metrics": permutation_metrics,
        },
        "runtime": runtime,
        "notes": [
            "Token-F1 is the existing retrieval-conditioned extractive proxy; generated QA is evaluated separately.",
            "The confirmation split was selected before V5 outcomes and is disjoint from prior project cohorts.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"{args.split}_retrieval_summary.json", summary)
    write_jsonl(args.output_dir / f"{args.split}_retrieval_rows.jsonl", rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
