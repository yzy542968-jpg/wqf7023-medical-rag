from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import file_sha256, read_json  # noqa: E402
from medical_rag.evaluation.graded_retrieval import (  # noqa: E402
    binary_recall_at_k,
    ndcg_at_k,
    reciprocal_rank_at_threshold,
)
from medical_rag.multimodal.evaluation import aggregate_case_images  # noqa: E402
from medical_rag.multimodal.fusion import l2_normalize, minmax_normalize, rank_scores  # noqa: E402
from medical_rag.multimodal.medsiglip import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    MedSiglipEncoder,
)
from medical_rag.multimodal.openi_images import resolve_official_image  # noqa: E402
from medical_rag.multimodal.v6_chunking import build_report_chunks  # noqa: E402
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.relevance import build_query_qrels  # noqa: E402
from medical_rag.similar_case.text_baseline import SimilarCaseBM25Retriever  # noqa: E402


DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v9" / "v9_full_source_split.json"
DEFAULT_PROTOCOL = ROOT / "config" / "v9_similar_case_rag_development.json"
DEFAULT_PREPROCESSING = ROOT / "config" / "v9_radgraph_preprocessing.json"
DEFAULT_MATRIX = ROOT / "config" / "v9_medsiglip_development_matrix.json"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_CACHE = ROOT / "data" / "processed" / "v9_medsiglip_development_embeddings.npz"
DEFAULT_ROWS = ROOT / "experiments" / "post_submission_v9" / "v9_medsiglip_validation_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v9" / "v9_medsiglip_validation_summary.json"


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def stable_signature(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def image_lookup(image_root: Path) -> dict[str, Path]:
    lookup: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in image_root.rglob("*.png"):
        if path.name in lookup:
            duplicates.add(path.name)
        lookup[path.name] = path
    if duplicates:
        raise RuntimeError(f"Duplicate official image names: {sorted(duplicates)[:5]}")
    return lookup


def resolve_images(
    case_ids: Sequence[str],
    raw_cases: Mapping[str, Mapping[str, Any]],
    lookup: Mapping[str, Path],
) -> tuple[list[str], list[Path]]:
    view_case_ids: list[str] = []
    paths: list[Path] = []
    for case_id in case_ids:
        case_paths = [
            resolve_official_image(case_id, str(row["filename"]), lookup)
            for row in raw_cases[case_id].get("images", [])
        ]
        case_paths = [path for path in case_paths if path is not None]
        if not case_paths:
            raise RuntimeError(f"V9 case {case_id} has no official readable image path.")
        view_case_ids.extend([case_id] * len(case_paths))
        paths.extend(case_paths)
    return view_case_ids, paths


def aggregate_chunks(
    chunk_embeddings: np.ndarray,
    chunk_case_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> np.ndarray:
    grouped: dict[str, list[np.ndarray]] = {case_id: [] for case_id in candidate_ids}
    for embedding, case_id in zip(chunk_embeddings, chunk_case_ids, strict=True):
        grouped[case_id].append(np.asarray(embedding, dtype=np.float32))
    missing = [case_id for case_id, values in grouped.items() if not values]
    if missing:
        raise RuntimeError(f"Candidate reports lack MedSigLIP chunks: {missing[:5]}")
    return np.stack(
        [l2_normalize(np.stack(grouped[case_id]).mean(axis=0)) for case_id in candidate_ids]
    )


def maximum_chunk_scores(
    query_image: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_case_indices: np.ndarray,
    candidate_count: int,
) -> np.ndarray:
    raw = np.asarray(chunk_embeddings, dtype=np.float32) @ np.asarray(
        query_image, dtype=np.float32
    )
    output = np.full(candidate_count, -np.inf, dtype=np.float32)
    np.maximum.at(output, chunk_case_indices, raw)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Every candidate must have a finite maximum chunk score.")
    return output


def select_report_policy(
    mean_ndcg: float, maximum_ndcg: float, tolerance: float
) -> str:
    return (
        "maximum_image_chunk_cosine"
        if float(maximum_ndcg) - float(mean_ndcg) + 1e-12 >= float(tolerance)
        else "normalized_mean_chunk_embedding"
    )


def select_fusion_weights(
    sweep: Sequence[Mapping[str, Any]], tolerance: float
) -> dict[str, Any]:
    eligible = [
        row
        for row in sweep
        if float(row["weights"]["bm25"]) > 0.0
        and float(row["weights"]["image_image"])
        + float(row["weights"]["image_report"])
        > 0.0
    ]
    if not eligible:
        raise ValueError("Fusion sweep has no eligible multimodal candidate.")
    maximum = max(float(row["metrics"]["ndcg@10"]) for row in eligible)
    contenders = [
        row
        for row in eligible
        if maximum - float(row["metrics"]["ndcg@10"]) <= float(tolerance)
    ]
    return sorted(
        contenders,
        key=lambda row: (
            -float(row["weights"]["bm25"]),
            abs(
                float(row["weights"]["image_image"])
                - float(row["weights"]["image_report"])
            ),
            tuple(float(row["weights"][name]) for name in ("bm25", "image_image", "image_report")),
        ),
    )[0]


def metric_row(
    qrels: Mapping[str, float],
    ranking: Sequence[str],
    *,
    threshold: float,
) -> dict[str, float]:
    return {
        "ndcg@1": ndcg_at_k(qrels, ranking, 1),
        "ndcg@5": ndcg_at_k(qrels, ranking, 5),
        "ndcg@10": ndcg_at_k(qrels, ranking, 10),
        "recall@1": binary_recall_at_k(qrels, ranking, 1, threshold=threshold),
        "recall@5": binary_recall_at_k(qrels, ranking, 5, threshold=threshold),
        "recall@10": binary_recall_at_k(qrels, ranking, 10, threshold=threshold),
        "mrr": reciprocal_rank_at_threshold(qrels, ranking, threshold=threshold),
    }


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    names = ("ndcg@1", "ndcg@5", "ndcg@10", "recall@1", "recall@5", "recall@10", "mrr")
    return {name: statistics.fmean(float(row[name]) for row in rows) for name in names}


def embedding_signature(
    *,
    matrix_path: Path,
    split_path: Path,
    cases_path: Path,
    candidate_ids: Sequence[str],
    validation_ids: Sequence[str],
    chunk_ids: Sequence[str],
    chunk_texts: Sequence[str],
    view_case_ids: Sequence[str],
    view_paths: Sequence[Path],
) -> str:
    return stable_signature(
        {
            "runner_sha256": file_sha256(Path(__file__)),
            "encoder_sha256": file_sha256(
                ROOT / "src" / "medical_rag" / "multimodal" / "medsiglip.py"
            ),
            "chunker_sha256": file_sha256(
                ROOT / "src" / "medical_rag" / "multimodal" / "v6_chunking.py"
            ),
            "matrix_sha256": file_sha256(matrix_path),
            "split_sha256": file_sha256(split_path),
            "cases_sha256": file_sha256(cases_path),
            "model": DEFAULT_MODEL,
            "revision": DEFAULT_REVISION,
            "candidate_ids": list(candidate_ids),
            "validation_ids": list(validation_ids),
            "chunk_ids": list(chunk_ids),
            "chunk_texts": list(chunk_texts),
            "view_case_ids": list(view_case_ids),
            "view_paths": [portable_path(path) for path in view_paths],
        }
    )


def load_cache(path: Path, signature: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cache:
        if str(cache["signature"].item()) != signature:
            return None
        return {name: cache[name] for name in cache.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen V9 MedSigLIP validation development.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--preprocessing", type=Path, default=DEFAULT_PREPROCESSING)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--text-batch-size", type=int, default=32)
    args = parser.parse_args()

    started = time.perf_counter()
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    preprocessing = read_json(args.preprocessing)
    matrix = read_json(args.matrix)
    split = read_json(args.split)
    protocol = read_json(args.protocol)
    if file_sha256(args.cases) != preprocessing["source"]["sha256"]:
        raise RuntimeError("V9 source changed from preprocessing protocol.")
    if file_sha256(args.split) != preprocessing["split"]["sha256"]:
        raise RuntimeError("V9 split changed from preprocessing protocol.")
    if matrix["model"]["name"] != DEFAULT_MODEL or matrix["model"]["revision"] != DEFAULT_REVISION:
        raise RuntimeError("MedSigLIP implementation defaults differ from the frozen matrix.")

    raw_rows = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_cases = {str(row["case_id"]): row for row in raw_rows}
    formal_cases = read_openi_paired_cases(
        args.cases, source_unique_patient=True, radgraph_path=args.radgraph
    )
    cases = {case.study_id: case for case in formal_cases}
    candidate_ids = sorted(
        case_id
        for case_id in split["partitions"]["train"]["case_ids"]
        if cases[case_id].metadata["radgraph_annotation_available"] is True
    )
    validation_ids = sorted(
        case_id
        for case_id in split["partitions"]["validation"]["case_ids"]
        if cases[case_id].metadata["radgraph_annotation_available"] is True
    )
    if len(candidate_ids) != matrix["candidate_bank_count"] or len(validation_ids) != matrix["validation_qrel_case_count"]:
        raise RuntimeError("V9 MedSigLIP development frame counts changed.")

    lookup = image_lookup(args.image_root)
    embedding_case_ids = candidate_ids + validation_ids
    view_case_ids, view_paths = resolve_images(embedding_case_ids, raw_cases, lookup)

    encoder = MedSiglipEncoder(
        revision=DEFAULT_REVISION,
        cache_dir=ROOT / ".hf_cache",
        local_files_only=True,
    )
    chunks = [
        chunk
        for case_id in candidate_ids
        for chunk in build_report_chunks(
            raw_cases[case_id],
            encoder.processor.tokenizer,
            max_tokens=int(matrix["model"]["max_text_tokens"]),
        )
    ]
    chunk_ids = [str(chunk["chunk_id"]) for chunk in chunks]
    chunk_case_ids = [str(chunk["case_id"]) for chunk in chunks]
    chunk_texts = [str(chunk["text"]) for chunk in chunks]
    signature = embedding_signature(
        matrix_path=args.matrix,
        split_path=args.split,
        cases_path=args.cases,
        candidate_ids=candidate_ids,
        validation_ids=validation_ids,
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        view_case_ids=view_case_ids,
        view_paths=view_paths,
    )
    cache = load_cache(args.cache, signature)
    if cache is None:
        import torch

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        embedding_started = time.perf_counter()
        view_embeddings = encoder.encode_images(view_paths, batch_size=args.image_batch_size)
        case_image_embeddings = aggregate_case_images(
            view_embeddings, view_case_ids, embedding_case_ids
        )
        chunk_embeddings = encoder.encode_texts(chunk_texts, batch_size=args.text_batch_size)
        report_mean_embeddings = aggregate_chunks(
            chunk_embeddings, chunk_case_ids, candidate_ids
        )
        embedding_seconds = time.perf_counter() - embedding_started
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
        chunk_case_index = {case_id: index for index, case_id in enumerate(candidate_ids)}
        chunk_case_indices = np.asarray(
            [chunk_case_index[case_id] for case_id in chunk_case_ids], dtype=np.int32
        )
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.cache,
            signature=np.asarray(signature),
            candidate_ids=np.asarray(candidate_ids),
            validation_ids=np.asarray(validation_ids),
            candidate_image_embeddings=case_image_embeddings[: len(candidate_ids)].astype(np.float32),
            validation_image_embeddings=case_image_embeddings[len(candidate_ids) :].astype(np.float32),
            chunk_embeddings=chunk_embeddings.astype(np.float32),
            chunk_case_indices=chunk_case_indices,
            report_mean_embeddings=report_mean_embeddings.astype(np.float32),
            embedding_seconds=np.asarray(embedding_seconds),
            peak_gpu_memory_mib=np.asarray(peak_mib),
            image_count=np.asarray(len(view_paths)),
            chunk_count=np.asarray(len(chunks)),
        )
        cache = load_cache(args.cache, signature)
        if cache is None:
            raise RuntimeError("Fresh V9 MedSigLIP cache could not be reloaded.")

    bank_images = np.asarray(cache["candidate_image_embeddings"], dtype=np.float32)
    validation_images = np.asarray(cache["validation_image_embeddings"], dtype=np.float32)
    chunk_embeddings = np.asarray(cache["chunk_embeddings"], dtype=np.float32)
    chunk_case_indices = np.asarray(cache["chunk_case_indices"], dtype=np.int32)
    report_means = np.asarray(cache["report_mean_embeddings"], dtype=np.float32)
    if bank_images.shape[0] != len(candidate_ids) or validation_images.shape[0] != len(validation_ids):
        raise RuntimeError("MedSigLIP cache case dimensions changed.")

    retriever = SimilarCaseBM25Retriever().fit(
        [cases[case_id] for case_id in candidate_ids], require_patient_ids=True
    )
    question_suite = protocol["question_suite"]
    qrel_arrays: dict[str, np.ndarray] = {}
    score_state: dict[str, dict[str, Any]] = {}
    component_rows: list[dict[str, Any]] = []
    threshold = float(matrix["binary_relevance_threshold"])

    for query_index, case_id in enumerate(validation_ids):
        query = cases[case_id]
        qrels = build_query_qrels(query, [cases[value] for value in candidate_ids])
        qrel_arrays[case_id] = np.asarray([qrels[value] for value in candidate_ids], dtype=np.float32)
        image_image = bank_images @ validation_images[query_index]
        image_report_mean = report_means @ validation_images[query_index]
        image_report_max = maximum_chunk_scores(
            validation_images[query_index],
            chunk_embeddings,
            chunk_case_indices,
            len(candidate_ids),
        )
        state = {
            "image_image": image_image,
            "image_report_mean": image_report_mean,
            "image_report_max": image_report_max,
            "bm25": {},
        }
        score_state[case_id] = state
        for question_type, question in question_suite.items():
            ranked = retriever.search(query, question, top_k=len(candidate_ids))
            score_map = {row.study_id: row.score for row in ranked}
            bm25 = np.asarray([score_map[value] for value in candidate_ids], dtype=np.float64)
            state["bm25"][question_type] = bm25
            systems = {
                "bm25_text": bm25,
                "medsiglip_image_image": image_image,
                "medsiglip_image_report_mean": image_report_mean,
                "medsiglip_image_report_max": image_report_max,
            }
            for system, scores in systems.items():
                ranking = rank_scores(candidate_ids, scores.tolist())
                component_rows.append(
                    {
                        "case_id": case_id,
                        "qid": f"{case_id}:{question_type}",
                        "question_type": question_type,
                        "system": system,
                        **metric_row(qrels, ranking, threshold=threshold),
                    }
                )
        if (query_index + 1) % 25 == 0 or query_index + 1 == len(validation_ids):
            print(f"component_validation_cases={query_index + 1}/{len(validation_ids)}", flush=True)

    component_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in component_rows:
        component_by_system[row["system"]].append(row)
    component_metrics = {
        system: aggregate_metrics(rows) for system, rows in sorted(component_by_system.items())
    }
    report_selection = select_report_policy(
        component_metrics["medsiglip_image_report_mean"]["ndcg@10"],
        component_metrics["medsiglip_image_report_max"]["ndcg@10"],
        float(matrix["report_policy_selection"]["maximum_minus_mean_required"]),
    )
    report_key = (
        "image_report_max"
        if report_selection == "maximum_image_chunk_cosine"
        else "image_report_mean"
    )

    fusion_sweep: list[dict[str, Any]] = []
    fusion_rows: list[dict[str, Any]] = []
    for weights in matrix["fusion_grid"]:
        weight_map = dict(zip(matrix["fusion_weight_order"], map(float, weights), strict=True))
        condition = (
            f"fusion_b{weight_map['bm25']:.2f}_ii{weight_map['image_image']:.2f}_"
            f"ir{weight_map['image_report']:.2f}"
        )
        condition_rows: list[dict[str, Any]] = []
        for case_id in validation_ids:
            state = score_state[case_id]
            qrels = {
                candidate_id: float(gain)
                for candidate_id, gain in zip(candidate_ids, qrel_arrays[case_id], strict=True)
            }
            normalized_ii = minmax_normalize(state["image_image"])
            normalized_ir = minmax_normalize(state[report_key])
            for question_type in question_suite:
                normalized_bm25 = minmax_normalize(state["bm25"][question_type])
                fused = (
                    weight_map["bm25"] * normalized_bm25
                    + weight_map["image_image"] * normalized_ii
                    + weight_map["image_report"] * normalized_ir
                )
                ranking = rank_scores(candidate_ids, fused.tolist())
                row = {
                    "case_id": case_id,
                    "qid": f"{case_id}:{question_type}",
                    "question_type": question_type,
                    "system": condition,
                    **metric_row(qrels, ranking, threshold=threshold),
                }
                condition_rows.append(row)
                fusion_rows.append(row)
        fusion_sweep.append(
            {
                "condition": condition,
                "weights": weight_map,
                "metrics": aggregate_metrics(condition_rows),
            }
        )
        print(f"fusion_condition_complete={condition}", flush=True)

    selected_fusion = select_fusion_weights(
        fusion_sweep, float(matrix["fusion_selection"]["tie_tolerance"])
    )
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    with args.rows_output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in component_rows + fusion_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    summary = {
        "study": "V9 MedSigLIP retrieval development validation",
        "status": "development_validation_complete_test_not_encoded_or_executed",
        "candidate_bank_count": len(candidate_ids),
        "validation_case_count": len(validation_ids),
        "validation_question_count": len(validation_ids) * len(question_suite),
        "model": {
            "name": DEFAULT_MODEL,
            "revision": DEFAULT_REVISION,
            "foundation_parameters_updated": False,
        },
        "embedding_cache": {
            "path": portable_path(args.cache),
            "sha256": file_sha256(args.cache),
            "signature": signature,
            "image_count": int(cache["image_count"].item()),
            "report_chunk_count": int(cache["chunk_count"].item()),
            "build_seconds": float(cache["embedding_seconds"].item()),
            "peak_gpu_memory_mib": float(cache["peak_gpu_memory_mib"].item()),
            "committed_to_public_repository": False,
        },
        "component_metrics": component_metrics,
        "report_policy_selection": {
            "selected": report_selection,
            "maximum_minus_mean_ndcg_at_10": (
                component_metrics["medsiglip_image_report_max"]["ndcg@10"]
                - component_metrics["medsiglip_image_report_mean"]["ndcg@10"]
            ),
            "required_difference": matrix["report_policy_selection"][
                "maximum_minus_mean_required"
            ],
        },
        "fusion_sweep": fusion_sweep,
        "selected_fixed_multimodal": selected_fusion,
        "rows_output": {
            "path": portable_path(args.rows_output),
            "sha256": file_sha256(args.rows_output),
            "committed_to_public_repository": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "test_queries_encoded": 0,
        "v9_test_outcomes_inspected": False,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
