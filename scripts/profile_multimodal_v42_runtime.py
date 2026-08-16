from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_multimodal_v4_retrieval import (
    candidate_case_ids,
    eligible_cases,
    image_lookup,
    load_json,
    verify_committed_selection,
    write_json,
)

from medical_rag.multimodal.biovilt import BioVilTEncoder
from medical_rag.multimodal.evaluation import build_text_query
from medical_rag.multimodal.fusion import shortlist_score_fusion
from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def percentile(probability: float) -> float:
        index = round((len(ordered) - 1) * probability)
        return ordered[index]

    return {
        "mean_ms": statistics.mean(values) * 1000,
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": percentile(0.95) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the locked V4.2 multimodal retrieval path.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "multimodal_v42.json")
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "experiments" / "post_submission_v42" / "confirmation_retrieval_summary.json",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data" / "processed" / "multimodal_v41_biovil_t_embeddings.npz",
    )
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "raw" / "openi_official_images")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "post_submission_v42" / "runtime_profile.json",
    )
    parser.add_argument("--confirmation-commit", default="9bb6bf7")
    parser.add_argument("--image-samples", type=int, default=30)
    args = parser.parse_args()

    verify_committed_selection(args.confirmation_commit, args.summary)
    config = load_json(args.config)
    cases = {
        str(row["case_id"]): row
        for row in load_cases_jsonl(ROOT / config["source"]["cases_path"])
    }
    candidate_ids = candidate_case_ids(config)
    image_files = image_lookup(args.image_root)
    eligible, case_images, exclusions = eligible_cases(candidate_ids, cases, image_files)
    if eligible != candidate_ids or exclusions:
        raise RuntimeError("Runtime cohort differs from the locked 720-case candidate pool.")

    cache = np.load(args.cache, allow_pickle=False)
    if cache["case_ids"].tolist() != candidate_ids:
        raise RuntimeError("Cached embeddings differ from the locked candidate pool.")
    image_embeddings = cache["image_embeddings"]
    report_embeddings = cache["report_embeddings"]
    case_index = {case_id: index for index, case_id in enumerate(candidate_ids)}

    benchmark = load_json(ROOT / config["cohorts"]["confirmation"]["benchmark_path"])
    questions = benchmark["questions"]
    bm25 = BM25Retriever().fit([cases[case_id] for case_id in candidate_ids])
    bm25_latencies = []
    rerank_latencies = []
    for question in questions:
        source_case_id = str(question["case_id"])
        started = time.perf_counter()
        rows = bm25.search(build_text_query(cases[source_case_id], question), top_k=len(candidate_ids))
        bm25_latencies.append(time.perf_counter() - started)

        started = time.perf_counter()
        similarities = report_embeddings @ image_embeddings[case_index[source_case_id]]
        image_scores = {
            candidate_id: float(similarities[index])
            for candidate_id, index in case_index.items()
        }
        shortlist_score_fusion(
            [str(row["case_id"]) for row in rows],
            [float(row["score"]) for row in rows],
            image_scores,
            shortlist_size=int(config["reranking"]["shortlist_size"]),
            text_weight=float(config["reranking"]["text_weight"]),
        )
        rerank_latencies.append(time.perf_counter() - started)

    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    encoder = BioVilTEncoder(
        model_name=config["encoder"]["joint_encoder"],
        text_revision=config["encoder"]["text_model_revision"],
        device="cuda",
        text_max_length=int(config["encoder"]["text_max_length"]),
    )
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    loaded_memory_mib = torch.cuda.memory_allocated() / 1024**2

    sample_case_ids = sorted({str(row["case_id"]) for row in questions})[: args.image_samples]
    sample_paths = [case_images[case_id][0] for case_id in sample_case_ids]
    encoder.encode_images(sample_paths[:1], batch_size=1)
    torch.cuda.synchronize()
    image_latencies = []
    for path in sample_paths:
        started = time.perf_counter()
        encoder.encode_images([path], batch_size=1)
        torch.cuda.synchronize()
        image_latencies.append(time.perf_counter() - started)

    image_profile = latency_summary(image_latencies)
    bm25_profile = latency_summary(bm25_latencies)
    rerank_profile = latency_summary(rerank_latencies)
    result: dict[str, Any] = {
        "experiment": config["experiment"],
        "confirmation_result_commit": args.confirmation_commit,
        "machine": {
            "cuda_device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "sample_counts": {
            "image_encodes": len(image_latencies),
            "confirmation_queries": len(questions),
            "candidate_reports_per_query": len(candidate_ids),
        },
        "cold_model_load_seconds": model_load_seconds,
        "loaded_model_cuda_memory_mib": loaded_memory_mib,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "latency": {
            "single_image_encoding": image_profile,
            "report_only_bm25": bm25_profile,
            "cached_image_similarity_plus_shortlist_rerank": rerank_profile,
            "warm_paired_request_estimated_mean_ms": (
                image_profile["mean_ms"] + bm25_profile["mean_ms"] + rerank_profile["mean_ms"]
            ),
        },
        "logical_calls_per_request": {
            "report_only_bm25": {
                "image_encoder": 0,
                "text_retrieval": 1,
                "image_report_similarity": 0,
                "shortlist_rerank": 0,
            },
            "paired_biovil_t_shortlist_reranker": {
                "image_encoder": 1,
                "text_retrieval": 1,
                "image_report_similarity": 1,
                "shortlist_rerank": 1,
            },
        },
        "locked_confirmation_performance": load_json(args.summary)["metrics"],
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
