from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.multimodal.biovilt import (  # noqa: E402
    DEFAULT_MODEL as BIOVILT_MODEL,
    DEFAULT_TEXT_REVISION as BIOVILT_REVISION,
    IMAGE_WEIGHTS_MD5,
    BioVilTEncoder,
)
from medical_rag.multimodal.evaluation import (  # noqa: E402
    aggregate_case_images,
    build_text_query,
    evaluate_rankings_and_answers,
)
from medical_rag.multimodal.fusion import (  # noqa: E402
    l2_normalize,
    rank_scores,
    shortlist_score_fusion,
)
from medical_rag.multimodal.medsiglip import (  # noqa: E402
    DEFAULT_MODEL as MEDSIGLIP_MODEL,
    DEFAULT_REVISION as MEDSIGLIP_REVISION,
    MedSiglipEncoder,
)
from medical_rag.multimodal.openi_images import resolve_official_image  # noqa: E402
from medical_rag.multimodal.v6_chunking import build_report_chunks  # noqa: E402
from medical_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v6_development.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_COHORT = ROOT / "data" / "processed" / "openi_multimodal_v5_cohort.json"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "post_submission_v6"
DEFAULT_MEDSIGLIP_CACHE = ROOT / "data" / "processed" / "v6_medsiglip_embeddings.npz"
DEFAULT_BIOVILT_CACHE = ROOT / "data" / "processed" / "v6_biovilt_standardized_embeddings.npz"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def stable_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def image_file_lookup(image_root: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in image_root.rglob("*.png"):
        if path.name in lookup:
            duplicates.add(path.name)
        lookup[path.name] = path
    if duplicates:
        raise RuntimeError(f"Duplicate official image names: {sorted(duplicates)[:5]}")
    return lookup


def resolve_case_images(
    candidate_ids: Sequence[str],
    cases: Mapping[str, Mapping[str, Any]],
    image_lookup: Mapping[str, Path],
) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for case_id in candidate_ids:
        paths = []
        for row in cases[case_id].get("images", []):
            path = resolve_official_image(case_id, str(row["filename"]), image_lookup)
            if path is not None:
                paths.append(path)
        if not paths:
            raise RuntimeError(f"Frozen candidate {case_id} has no readable official image.")
        result[case_id] = paths
    return result


def aggregate_chunk_embeddings(
    embeddings: np.ndarray,
    chunk_case_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> np.ndarray:
    grouped: dict[str, list[np.ndarray]] = {case_id: [] for case_id in candidate_ids}
    for embedding, case_id in zip(embeddings, chunk_case_ids, strict=True):
        grouped[case_id].append(np.asarray(embedding, dtype=np.float32))
    missing = [case_id for case_id, values in grouped.items() if not values]
    if missing:
        raise ValueError(f"Cases without report chunks: {missing[:5]}")
    return np.stack(
        [l2_normalize(np.stack(grouped[case_id]).mean(axis=0)) for case_id in candidate_ids]
    )


def maximum_chunk_scores(
    image_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_case_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> dict[str, float]:
    similarities = np.asarray(chunk_embeddings) @ np.asarray(image_embedding)
    scores = {case_id: float("-inf") for case_id in candidate_ids}
    for score, case_id in zip(similarities, chunk_case_ids, strict=True):
        scores[case_id] = max(scores[case_id], float(score))
    if any(not np.isfinite(value) for value in scores.values()):
        raise ValueError("Every candidate must have at least one finite chunk score.")
    return scores


def select_chunk_policy(mean_mrr: float, maximum_mrr: float, tolerance: float) -> dict[str, Any]:
    difference = float(maximum_mrr - mean_mrr)
    selected = "maximum_image_chunk_cosine" if difference >= tolerance else "normalized_mean_chunk_embedding"
    return {
        "selected_chunk_policy": selected,
        "maximum_minus_mean_mrr": difference,
        "tie_tolerance": float(tolerance),
        "rule": (
            "select maximum_image_chunk_cosine only when maximum_minus_mean_mrr "
            ">= tie_tolerance; otherwise select normalized_mean_chunk_embedding"
        ),
    }


def build_bm25_inputs(
    questions: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    cases: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, list[float]]]:
    retriever = BM25Retriever(k1=1.5, b=0.75).fit([cases[case_id] for case_id in candidate_ids])
    rankings: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}
    for question in questions:
        qid = str(question["qid"])
        source_case_id = str(question["case_id"])
        rows = retriever.search(
            build_text_query(cases[source_case_id], question),
            top_k=len(candidate_ids),
        )
        rankings[qid] = [str(row["case_id"]) for row in rows]
        scores[qid] = [float(row["score"]) for row in rows]
    return rankings, scores


def score_maps(
    image_embeddings: np.ndarray,
    report_embeddings: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_case_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    mean_maps: dict[str, dict[str, float]] = {}
    maximum_maps: dict[str, dict[str, float]] = {}
    for source_index, source_case_id in enumerate(candidate_ids):
        mean_scores = report_embeddings @ image_embeddings[source_index]
        mean_maps[source_case_id] = {
            candidate_id: float(mean_scores[index])
            for index, candidate_id in enumerate(candidate_ids)
        }
        maximum_maps[source_case_id] = maximum_chunk_scores(
            image_embeddings[source_index],
            chunk_embeddings,
            chunk_case_ids,
            candidate_ids,
        )
    return mean_maps, maximum_maps


def fused_rankings(
    questions: Sequence[Mapping[str, Any]],
    text_rankings: Mapping[str, Sequence[str]],
    text_scores: Mapping[str, Sequence[float]],
    image_score_maps: Mapping[str, Mapping[str, float]],
    *,
    shortlist_size: int,
    text_weight: float,
) -> dict[str, list[str]]:
    return {
        str(question["qid"]): shortlist_score_fusion(
            text_rankings[str(question["qid"])],
            text_scores[str(question["qid"])],
            image_score_maps[str(question["case_id"])],
            shortlist_size=shortlist_size,
            text_weight=text_weight,
        )
        for question in questions
    }


def cache_signature(
    *,
    encoder: str,
    revision: str,
    config_path: Path,
    cohort_path: Path,
    cases_path: Path,
    candidate_ids: Sequence[str],
    chunk_ids: Sequence[str],
    chunk_texts: Sequence[str],
    view_case_ids: Sequence[str],
    view_paths: Sequence[Path],
) -> str:
    return stable_signature(
        {
            "implementation_sha256": file_sha256(Path(__file__)),
            "chunking_implementation_sha256": file_sha256(
                ROOT / "src" / "medical_rag" / "multimodal" / "v6_chunking.py"
            ),
            "config_sha256": file_sha256(config_path),
            "cohort_sha256": file_sha256(cohort_path),
            "cases_sha256": file_sha256(cases_path),
            "encoder": encoder,
            "revision": revision,
            "candidate_ids": list(candidate_ids),
            "chunk_ids": list(chunk_ids),
            "chunk_texts": list(chunk_texts),
            "view_case_ids": list(view_case_ids),
            "view_paths": [portable_path(path) for path in view_paths],
        }
    )


def load_embedding_cache(path: Path, signature: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["signature"].item()) != signature:
            return None
        runtime = {
            "embedding_build_seconds": float(cached["embedding_build_seconds"].item()),
            "embedding_build_peak_gpu_memory_allocated_mib": float(
                cached["embedding_build_peak_gpu_memory_allocated_mib"].item()
            ),
        }
        return cached["chunk_embeddings"], cached["image_embeddings"], runtime


def save_embedding_cache(
    path: Path,
    signature: str,
    chunk_embeddings: np.ndarray,
    image_embeddings: np.ndarray,
    runtime: Mapping[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        signature=np.asarray(signature),
        chunk_embeddings=np.asarray(chunk_embeddings, dtype=np.float32),
        image_embeddings=np.asarray(image_embeddings, dtype=np.float32),
        embedding_build_seconds=np.asarray(runtime["embedding_build_seconds"]),
        embedding_build_peak_gpu_memory_allocated_mib=np.asarray(
            runtime["embedding_build_peak_gpu_memory_allocated_mib"]
        ),
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen V6 multimodal retrieval development matrix.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--medsiglip-cache", type=Path, default=DEFAULT_MEDSIGLIP_CACHE)
    parser.add_argument("--biovilt-cache", type=Path, default=DEFAULT_BIOVILT_CACHE)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--text-batch-size", type=int, default=32)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    config = read_json(args.config)
    cohort = read_json(args.cohort)
    cases = {str(case["case_id"]): case for case in load_cases_jsonl(args.cases)}
    candidate_ids = [str(value) for value in cohort["case_ids"]]
    development_ids = {str(value) for value in cohort["split"]["development"]["case_ids"]}
    questions = [row for row in cohort["questions"] if str(row["case_id"]) in development_ids]
    if len(candidate_ids) != 240 or len(development_ids) != 120 or len(questions) != 360:
        raise RuntimeError("V6 development source no longer matches the frozen protocol.")

    image_lookup = image_file_lookup(args.image_root)
    case_images = resolve_case_images(candidate_ids, cases, image_lookup)
    view_case_ids = [case_id for case_id in candidate_ids for _ in case_images[case_id]]
    view_paths = [path for case_id in candidate_ids for path in case_images[case_id]]

    medsiglip_encoder = MedSiglipEncoder(
        revision=MEDSIGLIP_REVISION,
        cache_dir=ROOT / ".hf_cache",
        local_files_only=True,
    )
    chunks = [
        chunk
        for case_id in candidate_ids
        for chunk in build_report_chunks(
            cases[case_id],
            medsiglip_encoder.processor.tokenizer,
            max_tokens=int(config["multimodal_retrieval"]["medsiglip_max_text_tokens"]),
        )
    ]
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    chunk_case_ids = [str(chunk["case_id"]) for chunk in chunks]
    chunk_texts = [str(chunk["text"]) for chunk in chunks]

    medsiglip_signature = cache_signature(
        encoder=MEDSIGLIP_MODEL,
        revision=MEDSIGLIP_REVISION,
        config_path=args.config,
        cohort_path=args.cohort,
        cases_path=args.cases,
        candidate_ids=candidate_ids,
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        view_case_ids=view_case_ids,
        view_paths=view_paths,
    )
    medsiglip_cached = load_embedding_cache(args.medsiglip_cache, medsiglip_signature)
    if medsiglip_cached is None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        medsiglip_chunks = medsiglip_encoder.encode_texts(chunk_texts, batch_size=args.text_batch_size)
        medsiglip_views = medsiglip_encoder.encode_images(view_paths, batch_size=args.image_batch_size)
        medsiglip_images = aggregate_case_images(medsiglip_views, view_case_ids, candidate_ids)
        medsiglip_runtime = {
            "embedding_build_seconds": time.perf_counter() - started,
            "embedding_build_peak_gpu_memory_allocated_mib": (
                float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            ),
        }
        save_embedding_cache(
            args.medsiglip_cache,
            medsiglip_signature,
            medsiglip_chunks,
            medsiglip_images,
            medsiglip_runtime,
        )
        medsiglip_cache_used = False
    else:
        medsiglip_chunks, medsiglip_images, medsiglip_runtime = medsiglip_cached
        medsiglip_cache_used = True
    del medsiglip_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    biovilt_signature = cache_signature(
        encoder=BIOVILT_MODEL,
        revision=BIOVILT_REVISION,
        config_path=args.config,
        cohort_path=args.cohort,
        cases_path=args.cases,
        candidate_ids=candidate_ids,
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        view_case_ids=view_case_ids,
        view_paths=view_paths,
    )
    biovilt_cached = load_embedding_cache(args.biovilt_cache, biovilt_signature)
    if biovilt_cached is None:
        biovilt_encoder = BioVilTEncoder(text_max_length=64)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        biovilt_chunks = biovilt_encoder.encode_texts(chunk_texts, batch_size=args.text_batch_size)
        biovilt_views = biovilt_encoder.encode_images(view_paths, batch_size=args.image_batch_size)
        biovilt_images = aggregate_case_images(biovilt_views, view_case_ids, candidate_ids)
        biovilt_runtime = {
            "embedding_build_seconds": time.perf_counter() - started,
            "embedding_build_peak_gpu_memory_allocated_mib": (
                float(torch.cuda.max_memory_allocated() / (1024**2)) if torch.cuda.is_available() else 0.0
            ),
        }
        save_embedding_cache(
            args.biovilt_cache,
            biovilt_signature,
            biovilt_chunks,
            biovilt_images,
            biovilt_runtime,
        )
        biovilt_cache_used = False
        del biovilt_encoder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        biovilt_chunks, biovilt_images, biovilt_runtime = biovilt_cached
        biovilt_cache_used = True

    medsiglip_reports = aggregate_chunk_embeddings(medsiglip_chunks, chunk_case_ids, candidate_ids)
    biovilt_reports = aggregate_chunk_embeddings(biovilt_chunks, chunk_case_ids, candidate_ids)
    medsiglip_mean_maps, medsiglip_max_maps = score_maps(
        medsiglip_images, medsiglip_reports, medsiglip_chunks, chunk_case_ids, candidate_ids
    )
    biovilt_mean_maps, biovilt_max_maps = score_maps(
        biovilt_images, biovilt_reports, biovilt_chunks, chunk_case_ids, candidate_ids
    )

    text_rankings, text_scores = build_bm25_inputs(questions, candidate_ids, cases)
    reranking = config["multimodal_retrieval"]
    common = {
        "shortlist_size": int(reranking["shortlist_size"]),
        "text_weight": float(reranking["text_weight"]),
    }
    systems = {
        "indication_question_bm25": text_rankings,
        "medsiglip_mean_chunk_reranker": fused_rankings(
            questions, text_rankings, text_scores, medsiglip_mean_maps, **common
        ),
        "medsiglip_max_chunk_reranker": fused_rankings(
            questions, text_rankings, text_scores, medsiglip_max_maps, **common
        ),
        "biovilt_mean_chunk_reranker": fused_rankings(
            questions, text_rankings, text_scores, biovilt_mean_maps, **common
        ),
        "biovilt_max_chunk_reranker": fused_rankings(
            questions, text_rankings, text_scores, biovilt_max_maps, **common
        ),
    }
    metrics: dict[str, dict[str, float]] = {}
    all_rows: list[dict[str, Any]] = []
    for system, rankings in systems.items():
        system_metrics, rows = evaluate_rankings_and_answers(questions, rankings, cases)
        metrics[system] = system_metrics
        all_rows.extend({"system": system, **row} for row in rows)

    selection = select_chunk_policy(
        metrics["medsiglip_mean_chunk_reranker"]["mrr"],
        metrics["medsiglip_max_chunk_reranker"]["mrr"],
        float(reranking["chunk_policy_tie_tolerance"]),
    )
    selected_suffix = "max_chunk_reranker" if selection["selected_chunk_policy"] == "maximum_image_chunk_cosine" else "mean_chunk_reranker"
    selected_systems = {
        "medsiglip": f"medsiglip_{selected_suffix}",
        "biovilt": f"biovilt_{selected_suffix}",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "development_multimodal_retrieval_rows.jsonl"
    summary_path = args.output_dir / "development_multimodal_retrieval_summary.json"
    write_jsonl(rows_path, all_rows)
    summary = {
        "experiment": "V6 development multimodal retrieval and chunk-policy selection",
        "protocol": "docs/V6_DEVELOPMENT_PROTOCOL.md",
        "config_sha256": file_sha256(args.config),
        "implementation_sha256": file_sha256(Path(__file__)),
        "cohort_sha256": file_sha256(args.cohort),
        "source_cases_sha256": file_sha256(args.cases),
        "development": {
            "target_case_count": len(development_ids),
            "candidate_case_count": len(candidate_ids),
            "question_count": len(questions),
            "view_count": len(view_paths),
            "chunk_count": len(chunks),
            "max_chunk_token_count": max(int(chunk["token_count"]) for chunk in chunks),
        },
        "models": {
            "medsiglip": {"id": MEDSIGLIP_MODEL, "revision": MEDSIGLIP_REVISION, "dimension": int(medsiglip_chunks.shape[1])},
            "biovilt": {"id": BIOVILT_MODEL, "text_revision": BIOVILT_REVISION, "image_weights_md5": IMAGE_WEIGHTS_MD5, "dimension": int(biovilt_chunks.shape[1])},
        },
        "fixed_reranking_policy": {
            "text_retriever": "BM25(k1=1.5,b=0.75)",
            "query": "clinical_indication_plus_question",
            "shortlist_size": common["shortlist_size"],
            "text_weight": common["text_weight"],
            "image_weight": 1.0 - common["text_weight"],
            "normalization": reranking["score_normalization"],
            "tie_break": reranking["tie_break"],
            "standardized_chunk_boundaries_for_both_encoders": True,
        },
        "metrics": metrics,
        "chunk_policy_selection": selection,
        "selected_standardized_encoder_systems": selected_systems,
        "runtime": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "medsiglip": {"cache_used": medsiglip_cache_used, **medsiglip_runtime},
            "biovilt": {"cache_used": biovilt_cache_used, **biovilt_runtime},
        },
        "caches": {
            "medsiglip": {"path": portable_path(args.medsiglip_cache), "signature": medsiglip_signature, "sha256": file_sha256(args.medsiglip_cache)},
            "biovilt": {"path": portable_path(args.biovilt_cache), "signature": biovilt_signature, "sha256": file_sha256(args.biovilt_cache)},
        },
        "outputs": {
            "summary": portable_path(summary_path),
            "rows": portable_path(rows_path),
            "row_count": len(all_rows),
            "rows_sha256": file_sha256(rows_path),
        },
        "claim_boundary": "Development outcomes select policies only; no V6 confirmation case IDs were generated or inspected.",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
