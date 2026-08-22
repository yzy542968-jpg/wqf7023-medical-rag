from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

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
from medical_rag.multimodal.v6_chunking import build_report_chunks  # noqa: E402
from medical_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v7_adaptive_fusion_development.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_MANIFEST = ROOT / "data" / "splits" / "v7" / "v7_development_manifest.json"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_OUTPUT = ROOT / "experiments" / "post_submission_v7" / "development_retrieval_rows.jsonl"
DEFAULT_CACHE = ROOT / "data" / "processed" / "v7_medsiglip_development_embeddings.npz"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_signature(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def image_lookup(image_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in image_root.rglob("*.png"):
        if path.name in result:
            duplicates.add(path.name)
        result[path.name] = path
    if duplicates:
        raise RuntimeError(f"Duplicate official image names: {sorted(duplicates)[:5]}")
    return result


def resolve_images(
    case_ids: Sequence[str],
    cases: Mapping[str, Mapping[str, Any]],
    lookup: Mapping[str, Path],
) -> tuple[list[str], list[Path], list[str]]:
    view_case_ids: list[str] = []
    view_paths: list[Path] = []
    for case_id in case_ids:
        resolved = []
        for image in cases[case_id].get("images", []):
            path = resolve_official_image(case_id, str(image["filename"]), lookup)
            if path is not None:
                resolved.append(path)
        if not resolved:
            raise RuntimeError(f"Case {case_id} has no readable official image.")
        view_case_ids.extend([case_id] * len(resolved))
        view_paths.extend(resolved)
    return view_case_ids, view_paths, [portable_path(path) for path in view_paths]


def cache_signature(
    *,
    config: Path,
    manifest: Path,
    cases: Path,
    case_ids: Sequence[str],
    chunk_ids: Sequence[str],
    chunk_texts: Sequence[str],
    view_paths: Sequence[str],
) -> str:
    return stable_signature(
        {
            "implementation_sha256": file_sha256(Path(__file__)),
            "chunking_implementation_sha256": file_sha256(
                ROOT / "src" / "medical_rag" / "multimodal" / "v6_chunking.py"
            ),
            "config_sha256": file_sha256(config),
            "manifest_sha256": file_sha256(manifest),
            "cases_sha256": file_sha256(cases),
            "encoder": MEDSIGLIP_MODEL,
            "revision": MEDSIGLIP_REVISION,
            "case_ids": list(case_ids),
            "chunk_ids": list(chunk_ids),
            "chunk_texts": list(chunk_texts),
            "view_paths": list(view_paths),
        }
    )


def save_embedding_cache(
    path: Path,
    *,
    signature: str,
    chunk_embeddings: np.ndarray,
    chunk_case_ids: Sequence[str],
    image_embeddings: np.ndarray,
    runtime: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        signature=np.asarray(signature),
        chunk_embeddings=np.asarray(chunk_embeddings, dtype=np.float32),
        chunk_case_ids=np.asarray(list(chunk_case_ids)),
        image_embeddings=np.asarray(image_embeddings, dtype=np.float32),
        embedding_build_seconds=np.asarray(float(runtime["embedding_build_seconds"])),
        embedding_build_peak_gpu_memory_allocated_mib=np.asarray(
            float(runtime["embedding_build_peak_gpu_memory_allocated_mib"])
        ),
    )


def load_embedding_cache(
    path: Path, signature: str
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, float]] | None:
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
        return (
            np.asarray(cached["chunk_embeddings"], dtype=np.float32),
            [str(value) for value in cached["chunk_case_ids"].tolist()],
            np.asarray(cached["image_embeddings"], dtype=np.float32),
            runtime,
        )


def aggregate_chunk_embeddings(
    chunk_embeddings: np.ndarray,
    chunk_case_ids: Sequence[str],
    case_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {case_id: [] for case_id in case_ids}
    for embedding, case_id in zip(chunk_embeddings, chunk_case_ids, strict=True):
        grouped.setdefault(case_id, []).append(np.asarray(embedding, dtype=np.float32))
    missing = [case_id for case_id, values in grouped.items() if not values]
    if missing:
        raise RuntimeError(f"Cases without report chunks: {missing[:5]}")
    return {
        case_id: np.asarray(values, dtype=np.float32)
        for case_id, values in grouped.items()
    }


def image_score_map(
    image_embedding: np.ndarray,
    candidate_chunks: Mapping[str, np.ndarray],
    candidate_ids: Sequence[str],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for candidate_id in candidate_ids:
        similarities = candidate_chunks[candidate_id] @ image_embedding
        scores[candidate_id] = float(np.max(similarities))
    return scores


def ranked_bm25(
    retriever: BM25Retriever,
    query: str,
    candidate_ids: Sequence[str],
) -> tuple[list[str], list[float]]:
    rows = retriever.search(query, top_k=len(candidate_ids))
    score_by_id = {str(row["case_id"]): float(row["score"]) for row in rows}
    ranking = sorted(candidate_ids, key=lambda case_id: (-score_by_id[case_id], case_id))
    return ranking, [score_by_id[case_id] for case_id in ranking]


def build_rows_for_block(
    block_name: str,
    block: Mapping[str, Any],
    cases: Mapping[str, Mapping[str, Any]],
    case_chunks: Mapping[str, Sequence[Mapping[str, Any]]],
    case_image_embeddings: Mapping[str, np.ndarray],
    *,
    shortlist_size: int,
) -> list[dict[str, Any]]:
    candidate_ids = [str(value) for value in block["case_ids"]]
    target_ids = {str(value) for value in block["target_case_ids"]}
    retriever = BM25Retriever(k1=1.5, b=0.75).fit(
        [cases[case_id] for case_id in candidate_ids]
    )
    questions: list[dict[str, Any]] = []
    for target_id in sorted(target_ids):
        questions.extend(build_case_questions(cases[target_id], list(case_chunks[target_id])))
    if len(questions) != 360:
        raise RuntimeError(f"{block_name} should contain 360 questions, found {len(questions)}.")

    rows: list[dict[str, Any]] = []
    candidate_chunk_embeddings = {
        case_id: np.asarray(
            [chunk["embedding"] for chunk in case_chunks[case_id]], dtype=np.float32
        )
        for case_id in candidate_ids
    }
    for question in questions:
        qid = str(question["qid"])
        target_id = str(question["case_id"])
        text_ranking, text_scores = ranked_bm25(
            retriever,
            build_text_query(cases[target_id], question),
            candidate_ids,
        )
        shortlist_ids = text_ranking[:shortlist_size]
        shortlist_text_scores = np.asarray(text_scores[:shortlist_size], dtype=np.float64)
        image_scores = image_score_map(
            case_image_embeddings[target_id], candidate_chunk_embeddings, candidate_ids
        )
        shortlist_image_scores = np.asarray(
            [image_scores[case_id] for case_id in shortlist_ids], dtype=np.float64
        )
        target_rank = text_ranking.index(target_id) + 1
        rows.append(
            {
                "block": block_name,
                "qid": qid,
                "case_id": target_id,
                "question_type": str(question["question_type"]),
                "question": str(question["question"]),
                "reference_answer": str(question["reference_answer"]),
                "target_case_id": target_id,
                "target_text_rank": target_rank,
                "target_in_shortlist": target_id in shortlist_ids,
                "candidate_case_ids": shortlist_ids,
                "text_scores": [float(value) for value in shortlist_text_scores],
                "image_scores": [float(value) for value in shortlist_image_scores],
                "text_scores_normalized": [
                    float(value) for value in minmax_normalize(shortlist_text_scores)
                ],
                "image_scores_normalized": [
                    float(value) for value in minmax_normalize(shortlist_image_scores)
                ],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the V7 development retrieval matrix.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--text-batch-size", type=int, default=16)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    config = read_json(args.config)
    manifest = read_json(args.manifest)
    if manifest["confirmation_case_ids_instantiated"] is not False:
        raise RuntimeError("V7 development manifest unexpectedly contains confirmation IDs.")
    blocks = manifest["blocks"]
    if set(blocks) != {"train_a", "train_b", "validation"}:
        raise RuntimeError("V7 retrieval requires exactly Train A, Train B, and Validation blocks.")
    cases = {str(case["case_id"]): case for case in load_cases_jsonl(args.cases)}
    all_block_ids = sorted(
        {str(case_id) for block in blocks.values() for case_id in block["case_ids"]}
    )
    if len(all_block_ids) != 720:
        raise RuntimeError("V7 development blocks must contain 720 unique cases.")
    lookup = image_lookup(args.image_root)
    view_case_ids, view_paths, view_path_strings = resolve_images(all_block_ids, cases, lookup)

    medsiglip = MedSiglipEncoder(
        revision=MEDSIGLIP_REVISION,
        cache_dir=ROOT / ".hf_cache",
        local_files_only=True,
        max_text_tokens=64,
    )
    chunks: list[dict[str, Any]] = []
    for case_id in all_block_ids:
        chunks.extend(
            build_report_chunks(
                cases[case_id],
                medsiglip.processor.tokenizer,
                max_tokens=64,
            )
        )
    chunk_ids = [str(row["chunk_id"]) for row in chunks]
    chunk_case_ids = [str(row["case_id"]) for row in chunks]
    chunk_texts = [str(row["text"]) for row in chunks]
    signature = cache_signature(
        config=args.config,
        manifest=args.manifest,
        cases=args.cases,
        case_ids=all_block_ids,
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
        case_image_embeddings: dict[str, np.ndarray] = {}
        for case_id in all_block_ids:
            views = [
                view_embeddings[index]
                for index, view_case_id in enumerate(view_case_ids)
                if view_case_id == case_id
            ]
            case_image_embeddings[case_id] = aggregate_view_embeddings(views)
        runtime = {
            "embedding_build_seconds": time.perf_counter() - started,
            "embedding_build_peak_gpu_memory_allocated_mib": float(
                torch.cuda.max_memory_allocated() / (1024**2)
            )
            if torch.cuda.is_available()
            else 0.0,
        }
        save_embedding_cache(
            args.cache,
            signature=signature,
            chunk_embeddings=chunk_embeddings,
            chunk_case_ids=chunk_case_ids,
            image_embeddings=np.stack([case_image_embeddings[case_id] for case_id in all_block_ids]),
            runtime=runtime,
        )
        cache_used = False
    else:
        chunk_embeddings, cached_chunk_case_ids, image_embeddings, runtime = cached
        if cached_chunk_case_ids != chunk_case_ids:
            raise RuntimeError("Cached chunk case IDs differ from the current V7 manifest.")
        case_image_embeddings = {
            case_id: image_embeddings[index]
            for index, case_id in enumerate(all_block_ids)
        }
        cache_used = True
    del medsiglip
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    chunk_embedding_by_case: dict[str, list[dict[str, Any]]] = {case_id: [] for case_id in all_block_ids}
    for row, embedding in zip(chunks, chunk_embeddings, strict=True):
        chunk_embedding_by_case[str(row["case_id"])].append(
            {**row, "embedding": np.asarray(embedding, dtype=np.float32)}
        )

    output_rows: list[dict[str, Any]] = []
    block_summaries: dict[str, Any] = {}
    shortlist_size = int(config["text_retrieval"]["shortlist_size"])
    for block_name in ("train_a", "train_b", "validation"):
        rows = build_rows_for_block(
            block_name,
            blocks[block_name],
            cases,
            chunk_embedding_by_case,
            case_image_embeddings,
            shortlist_size=shortlist_size,
        )
        output_rows.extend(rows)
        block_summaries[block_name] = {
            "case_count": len(blocks[block_name]["case_ids"]),
            "question_count": len(rows),
            "target_outside_shortlist_count": sum(
                not bool(row["target_in_shortlist"]) for row in rows
            ),
            "target_outside_shortlist_rate": sum(
                not bool(row["target_in_shortlist"]) for row in rows
            )
            / len(rows),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    summary = {
        "experiment": "V7 development retrieval matrix",
        "status": "development_retrieval_inputs_ready_no_confirmation_ids_instantiated",
        "config_path": portable_path(args.config),
        "config_sha256": file_sha256(args.config),
        "manifest_path": portable_path(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "source_cases": portable_path(args.cases),
        "source_cases_sha256": file_sha256(args.cases),
        "implementation_sha256": file_sha256(Path(__file__)),
        "model": {
            "name": MEDSIGLIP_MODEL,
            "revision": MEDSIGLIP_REVISION,
            "cache_used": cache_used,
            "cache_signature": signature,
        },
        "candidate_case_count": len(all_block_ids),
        "question_count": len(output_rows),
        "chunk_count": len(chunks),
        "view_count": len(view_paths),
        "shortlist_size": shortlist_size,
        "runtime": runtime,
        "blocks": block_summaries,
        "outputs": {
            "rows": portable_path(args.output),
            "rows_sha256": file_sha256(args.output),
            "row_count": len(output_rows),
            "cache": portable_path(args.cache),
        },
        "claim_boundary": "Development retrieval inputs only; no confirmation outcome or clinical claim.",
    }
    summary_path = args.output.with_name("development_retrieval_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
