from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_v7_confirmation_cohort import committed_json  # noqa: E402
from medical_rag.evaluation.case_scoped_benchmark import (  # noqa: E402
    build_case_chunks,
    build_case_questions,
)
from medical_rag.multimodal.evaluation import build_text_query  # noqa: E402
from medical_rag.multimodal.fusion import aggregate_view_embeddings, minmax_normalize  # noqa: E402
from medical_rag.multimodal.medsiglip import (  # noqa: E402
    DEFAULT_MODEL as MEDSIGLIP_MODEL,
    DEFAULT_REVISION as MEDSIGLIP_REVISION,
    MedSiglipEncoder,
)
from medical_rag.multimodal.openi_images import resolve_official_image  # noqa: E402
from medical_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl  # noqa: E402
from train_v7_adaptive_fusion import (  # noqa: E402
    FEATURE_NAMES,
    FeatureScaler,
    alpha_predictions,
    build_features,
    build_model,
    mean_reciprocal_rank,
)
from run_v7_development_retrieval import (  # noqa: E402
    cache_signature,
    file_sha256,
    image_lookup,
    image_score_map,
    load_embedding_cache,
    portable_path,
    ranked_bm25,
    resolve_images,
    save_embedding_cache,
)


DEFAULT_CONFIG = ROOT / "config" / "v7_confirmation.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_COHORT = ROOT / "data" / "splits" / "v7" / "v7_confirmation_cohort.json"
DEFAULT_CHECKPOINT = ROOT / "experiments" / "post_submission_v7" / "v7_adaptive_fusion_final_checkpoint.pt"
DEFAULT_SCALER = ROOT / "experiments" / "post_submission_v7" / "v7_adaptive_fusion_feature_scaler.json"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_CACHE = ROOT / "data" / "processed" / "v7_confirmation_medsiglip_embeddings.npz"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "post_submission_v7"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def image_order(candidate_ids: Sequence[str], image_scores: np.ndarray) -> list[str]:
    return [
        candidate_id
        for candidate_id, _ in sorted(
            zip(candidate_ids, image_scores.tolist(), strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]


def build_question_rows(
    cases: Mapping[str, Mapping[str, Any]],
    candidate_ids: Sequence[str],
    target_ids: Sequence[str],
    case_chunks: Mapping[str, Sequence[Mapping[str, Any]]],
    case_image_embeddings: Mapping[str, np.ndarray],
    *,
    shortlist_size: int,
) -> list[dict[str, Any]]:
    retriever = BM25Retriever(k1=1.5, b=0.75).fit(
        [cases[case_id] for case_id in candidate_ids]
    )
    candidate_chunk_embeddings = {
        case_id: np.asarray(
            [chunk["embedding"] for chunk in case_chunks[case_id]], dtype=np.float32
        )
        for case_id in candidate_ids
    }
    questions: list[dict[str, Any]] = []
    for target_id in sorted(target_ids):
        questions.extend(build_case_questions(cases[target_id], list(case_chunks[target_id])))
    if len(questions) != 360:
        raise RuntimeError(f"Expected 360 confirmation questions, found {len(questions)}.")
    rows: list[dict[str, Any]] = []
    for question in questions:
        target_id = str(question["case_id"])
        text_ranking, text_scores = ranked_bm25(
            retriever,
            build_text_query(cases[target_id], question),
            candidate_ids,
        )
        shortlist_ids = text_ranking[:shortlist_size]
        shortlist_text = np.asarray(text_scores[:shortlist_size], dtype=np.float64)
        score_map = image_score_map(
            case_image_embeddings[target_id], candidate_chunk_embeddings, candidate_ids
        )
        shortlist_image = np.asarray(
            [score_map[case_id] for case_id in shortlist_ids], dtype=np.float64
        )
        rows.append(
            {
                "qid": str(question["qid"]),
                "case_id": target_id,
                "question_type": str(question["question_type"]),
                "question": str(question["question"]),
                "reference_answer": str(question["reference_answer"]),
                "target_case_id": target_id,
                "target_text_rank": text_ranking.index(target_id) + 1,
                "target_in_shortlist": target_id in shortlist_ids,
                "candidate_case_ids": shortlist_ids,
                "text_scores": [float(value) for value in shortlist_text],
                "image_scores": [float(value) for value in shortlist_image],
                "text_scores_normalized": [float(value) for value in minmax_normalize(shortlist_text)],
                "image_scores_normalized": [float(value) for value in minmax_normalize(shortlist_image)],
            }
        )
    return rows


def shuffled_rows(
    aligned_rows: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, str],
    case_image_embeddings: Mapping[str, np.ndarray],
    case_chunks: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    candidate_chunk_embeddings = {
        case_id: np.asarray(
            [chunk["embedding"] for chunk in case_chunks[case_id]], dtype=np.float32
        )
        for case_id in case_chunks
    }
    output: list[dict[str, Any]] = []
    for row in aligned_rows:
        target_id = str(row["target_case_id"])
        candidate_ids = [str(value) for value in row["candidate_case_ids"]]
        shuffled_image = case_image_embeddings[assignment[target_id]]
        score_map = image_score_map(shuffled_image, candidate_chunk_embeddings, candidate_ids)
        scores = np.asarray([score_map[case_id] for case_id in candidate_ids], dtype=np.float64)
        output_row = dict(row)
        output_row["image_scores"] = [float(value) for value in scores]
        output_row["image_scores_normalized"] = [float(value) for value in minmax_normalize(scores)]
        output.append(output_row)
    return output


def shuffled_assignments(target_ids: Sequence[str], count: int, seed: int, domain: str) -> list[dict[str, str]]:
    if len(target_ids) < 2:
        raise ValueError("Shuffled controls need at least two target IDs.")
    signatures: set[str] = set()
    assignments: list[dict[str, str]] = []
    for control_index in range(count):
        ordered = sorted(
            target_ids,
            key=lambda case_id: hashlib.sha256(
                f"{domain}|{seed}|{control_index}|{case_id}".encode("utf-8")
            ).hexdigest(),
        )
        assignment = {
            source: ordered[(index + 1) % len(ordered)]
            for index, source in enumerate(ordered)
        }
        if any(source == assigned for source, assigned in assignment.items()):
            raise RuntimeError("Shuffled assignment contains a fixed point.")
        signature = stable_signature(assignment)
        if signature in signatures:
            raise RuntimeError("Shuffled assignments are not unique.")
        signatures.add(signature)
        assignments.append(assignment)
    return assignments


def reciprocal_values(rows: Sequence[Mapping[str, Any]], alphas: np.ndarray) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for row, alpha in zip(rows, alphas.tolist(), strict=True):
        candidate_ids = [str(value) for value in row["candidate_case_ids"]]
        text_scores = np.asarray(row["text_scores_normalized"], dtype=np.float64)
        image_scores = np.asarray(row["image_scores_normalized"], dtype=np.float64)
        fused = float(alpha) * text_scores + (1.0 - float(alpha)) * image_scores
        ranking = [
            case_id
            for case_id, _ in sorted(
                zip(candidate_ids, fused.tolist(), strict=True),
                key=lambda item: (-float(item[1]), item[0]),
            )
        ]
        target = str(row["target_case_id"])
        values[str(row["case_id"])].append(
            1.0 / (ranking.index(target) + 1) if target in ranking else 0.0
        )
    return values


def hit_rate(rows: Sequence[Mapping[str, Any]], alphas: np.ndarray, k: int) -> float:
    hits = 0
    for row, alpha in zip(rows, alphas.tolist(), strict=True):
        candidate_ids = [str(value) for value in row["candidate_case_ids"]]
        text_scores = np.asarray(row["text_scores_normalized"], dtype=np.float64)
        image_scores = np.asarray(row["image_scores_normalized"], dtype=np.float64)
        fused = float(alpha) * text_scores + (1.0 - float(alpha)) * image_scores
        ranking = [
            case_id
            for case_id, _ in sorted(
                zip(candidate_ids, fused.tolist(), strict=True),
                key=lambda item: (-float(item[1]), item[0]),
            )
        ]
        hits += int(str(row["target_case_id"]) in ranking[:k])
    return hits / len(rows) if rows else 0.0


def paired_bootstrap(first: Mapping[str, Sequence[float]], second: Mapping[str, Sequence[float]], iterations: int, seed: int) -> dict[str, float]:
    case_ids = sorted(set(first) & set(second))
    differences = {
        case_id: float(np.mean(first[case_id]) - np.mean(second[case_id]))
        for case_id in case_ids
    }
    observed = float(np.mean(list(differences.values())))
    rng = random.Random(seed)
    samples = [
        float(np.mean([differences[rng.choice(case_ids)] for _ in case_ids]))
        for _ in range(iterations)
    ]
    return {
        "mean_difference": observed,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "case_count": len(case_ids),
    }


def system_metrics(rows: Sequence[Mapping[str, Any]], alphas: np.ndarray) -> dict[str, Any]:
    reciprocal = reciprocal_values(rows, alphas)
    return {
        "mrr": float(np.mean([np.mean(values) for values in reciprocal.values()])),
        "hit_at_1": hit_rate(rows, alphas, 1),
        "hit_at_5": hit_rate(rows, alphas, 5),
        "hit_at_10": hit_rate(rows, alphas, 10),
        "target_outside_shortlist_rate": float(
            np.mean([not bool(row["target_in_shortlist"]) for row in rows])
        ),
        "case_count": len(reciprocal),
        "question_count": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V7 frozen adaptive-fusion confirmation retrieval.")
    parser.add_argument("--protocol-commit", default="4821f38")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--scaler", type=Path, default=DEFAULT_SCALER)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--text-batch-size", type=int, default=16)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    config = read_json(args.config)
    if config != committed_json(args.protocol_commit, args.config):
        raise RuntimeError("V7 confirmation config differs from committed protocol config.")
    cohort = read_json(args.cohort)
    if cohort["protocol_commit"] != args.protocol_commit:
        raise RuntimeError("V7 cohort was not built from the frozen confirmation protocol.")
    if cohort["config_sha256"] != file_sha256(args.config):
        raise RuntimeError("V7 cohort config hash does not match current config.")
    cases = {str(case["case_id"]): case for case in load_cases_jsonl(args.cases)}
    candidate_ids = [str(value) for value in cohort["case_ids"]]
    target_ids = [str(value) for value in cohort["target_case_ids"]]
    if len(candidate_ids) != 240 or len(target_ids) != 120:
        raise RuntimeError("V7 confirmation cohort dimensions changed.")

    lookup = image_lookup(args.image_root)
    view_case_ids, view_paths, view_path_strings = resolve_images(candidate_ids, cases, lookup)
    medsiglip = MedSiglipEncoder(
        revision=MEDSIGLIP_REVISION,
        cache_dir=ROOT / ".hf_cache",
        local_files_only=True,
        max_text_tokens=64,
    )
    chunks: list[dict[str, Any]] = []
    for case_id in candidate_ids:
        chunks.extend(
            __import__("medical_rag.multimodal.v6_chunking", fromlist=["build_report_chunks"]).build_report_chunks(
                cases[case_id], medsiglip.processor.tokenizer, max_tokens=64
            )
        )
    chunk_ids = [str(row["chunk_id"]) for row in chunks]
    chunk_case_ids = [str(row["case_id"]) for row in chunks]
    chunk_texts = [str(row["text"]) for row in chunks]
    signature = cache_signature(
        config=args.config,
        manifest=args.cohort,
        cases=args.cases,
        case_ids=candidate_ids,
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        view_paths=view_path_strings,
    )
    cached = load_embedding_cache(args.cache, signature)
    if cached is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        chunk_embeddings = medsiglip.encode_texts(chunk_texts, batch_size=args.text_batch_size)
        view_embeddings = medsiglip.encode_images(view_paths, batch_size=args.image_batch_size)
        case_image_embeddings = {
            case_id: aggregate_view_embeddings(
                [view_embeddings[index] for index, view_case_id in enumerate(view_case_ids) if view_case_id == case_id]
            )
            for case_id in candidate_ids
        }
        runtime = {
            "embedding_build_seconds": time.perf_counter() - started,
            "embedding_build_peak_gpu_memory_allocated_mib": float(
                torch.cuda.max_memory_allocated() / (1024**2)
            ) if torch.cuda.is_available() else 0.0,
        }
        save_embedding_cache(
            args.cache,
            signature=signature,
            chunk_embeddings=chunk_embeddings,
            chunk_case_ids=chunk_case_ids,
            image_embeddings=np.stack([case_image_embeddings[case_id] for case_id in candidate_ids]),
            runtime=runtime,
        )
        cache_used = False
    else:
        chunk_embeddings, cached_chunk_case_ids, image_embeddings, runtime = cached
        if cached_chunk_case_ids != chunk_case_ids:
            raise RuntimeError("V7 cached chunk IDs differ from the frozen cohort.")
        case_image_embeddings = {
            case_id: image_embeddings[index]
            for index, case_id in enumerate(candidate_ids)
        }
        cache_used = True
    del medsiglip
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    case_chunks: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in candidate_ids}
    for row, embedding in zip(chunks, chunk_embeddings, strict=True):
        case_chunks[str(row["case_id"])].append(
            {**row, "embedding": np.asarray(embedding, dtype=np.float32)}
        )
    aligned_rows = build_question_rows(
        cases,
        candidate_ids,
        target_ids,
        case_chunks,
        case_image_embeddings,
        shortlist_size=int(config["retrieval"]["shortlist_size"]),
    )
    features = np.stack([build_features(row, cases) for row in aligned_rows])
    scaler_payload = read_json(args.scaler)
    scaler = FeatureScaler(
        np.asarray(scaler_payload["mean"], dtype=np.float32),
        np.asarray(scaler_payload["scale"], dtype=np.float32),
    )
    if scaler_payload["feature_names"] != FEATURE_NAMES:
        raise RuntimeError("V7 feature schema differs from the frozen scaler.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["feature_names"] != FEATURE_NAMES or checkpoint["model_type"] != "linear_sigmoid":
        raise RuntimeError("V7 checkpoint metadata differs from the frozen model.")
    model = build_model(str(checkpoint["model_type"]), len(FEATURE_NAMES))
    model.load_state_dict(checkpoint["state_dict"])
    adaptive_alphas = alpha_predictions(model, features, scaler)
    fixed_half_alphas = np.full(len(aligned_rows), 0.5, dtype=np.float32)
    global_alphas = np.full(len(aligned_rows), float(config["retrieval"]["global_alpha_star"]), dtype=np.float32)
    text_alphas = np.ones(len(aligned_rows), dtype=np.float32)
    image_alphas = np.zeros(len(aligned_rows), dtype=np.float32)
    systems = {
        "bm25_text_only": text_alphas,
        "medsiglip_image_only_within_bm25_top100": image_alphas,
        "fixed_alpha_0_50": fixed_half_alphas,
        "global_alpha_0_52": global_alphas,
        "adaptive_alpha_q": adaptive_alphas,
    }
    metrics = {name: system_metrics(aligned_rows, alpha_values) for name, alpha_values in systems.items()}
    adaptive_case_values = reciprocal_values(aligned_rows, adaptive_alphas)
    global_case_values = reciprocal_values(aligned_rows, global_alphas)
    h1 = paired_bootstrap(adaptive_case_values, global_case_values, 5000, 7026)
    assignments = shuffled_assignments(
        target_ids,
        count=int(config["shuffled_image_control"]["count"]),
        seed=int(config["shuffled_image_control"]["seed"]),
        domain=str(config["shuffled_image_control"]["order_domain"]),
    )
    shuffled_metrics: list[dict[str, Any]] = []
    shuffled_rows_by_control: list[list[dict[str, Any]]] = []
    for control_index, assignment in enumerate(assignments):
        rows = shuffled_rows(aligned_rows, assignment, case_image_embeddings, case_chunks)
        shuffled_features = np.stack([build_features(row, cases) for row in rows])
        shuffled_alpha = alpha_predictions(model, shuffled_features, scaler)
        control_metrics = system_metrics(rows, shuffled_alpha)
        shuffled_metrics.append(
            {
                "control_index": control_index,
                "assignment_sha256": stable_signature(assignment),
                **control_metrics,
            }
        )
        shuffled_rows_by_control.append(rows)
    aligned_mrr = float(metrics["adaptive_alpha_q"]["mrr"])
    exceedances = sum(float(row["mrr"]) >= aligned_mrr for row in shuffled_metrics)
    h2 = {
        "aligned_adaptive_mrr": aligned_mrr,
        "shuffled_mrr_mean": float(np.mean([row["mrr"] for row in shuffled_metrics])),
        "shuffled_mrr_median": float(np.median([row["mrr"] for row in shuffled_metrics])),
        "shuffled_mrr_min": float(min(row["mrr"] for row in shuffled_metrics)),
        "shuffled_mrr_max": float(max(row["mrr"] for row in shuffled_metrics)),
        "exceedance_count": exceedances,
        "count": len(shuffled_metrics),
        "plus_one_monte_carlo_p": (exceedances + 1) / (len(shuffled_metrics) + 1),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "v7_confirmation_retrieval_rows.jsonl"
    with rows_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row, alpha in zip(aligned_rows, adaptive_alphas.tolist(), strict=True):
            handle.write(json.dumps({**row, "adaptive_alpha_q": float(alpha)}, ensure_ascii=True) + "\n")
    summary = {
        "experiment": "V7 adaptive multimodal fusion confirmation retrieval",
        "status": "formal_confirmation_retrieval_outcomes_frozen",
        "protocol_commit": args.protocol_commit,
        "config_path": portable_path(args.config),
        "config_sha256": file_sha256(args.config),
        "cohort_path": portable_path(args.cohort),
        "cohort_sha256": file_sha256(args.cohort),
        "source_cases_sha256": file_sha256(args.cases),
        "implementation_sha256": file_sha256(Path(__file__)),
        "candidate_case_count": len(candidate_ids),
        "target_case_count": len(target_ids),
        "question_count": len(aligned_rows),
        "chunk_count": len(chunks),
        "view_count": len(view_paths),
        "models": {
            "medsiglip": {"name": MEDSIGLIP_MODEL, "revision": MEDSIGLIP_REVISION, "cache_used": cache_used},
            "adaptive_checkpoint": portable_path(args.checkpoint),
            "adaptive_checkpoint_sha256": file_sha256(args.checkpoint),
            "feature_scaler": portable_path(args.scaler),
            "feature_scaler_sha256": file_sha256(args.scaler),
        },
        "metrics": metrics,
        "h1": h1,
        "h2": h2,
        "shuffled_controls": shuffled_metrics,
        "runtime": runtime,
        "outputs": {
            "rows": portable_path(rows_path),
            "rows_sha256": file_sha256(rows_path),
            "row_count": len(aligned_rows),
        },
        "claim_boundary": "Same-source closed-set paired-report retrieval; no diagnosis, external validation, or clinical adjudication claim.",
    }
    summary_path = args.output_dir / "v7_confirmation_retrieval_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
