from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_v6_development_multimodal_retrieval import (  # noqa: E402
    aggregate_chunk_embeddings,
    build_bm25_inputs,
    fused_rankings,
    image_file_lookup,
    maximum_chunk_scores,
    read_json,
    resolve_case_images,
)
from run_v6_development_text_retrieval import (  # noqa: E402
    dense_rankings,
    detailed_instruction,
)

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
from medical_rag.multimodal.medsiglip import (  # noqa: E402
    DEFAULT_MODEL as MEDSIGLIP_MODEL,
    DEFAULT_REVISION as MEDSIGLIP_REVISION,
    MedSiglipEncoder,
)
from medical_rag.multimodal.v6_chunking import build_report_chunks  # noqa: E402
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v6_confirmation.json"
DEFAULT_COHORT = ROOT / "data" / "splits" / "v6" / "v6_confirmation_cohort.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "post_submission_v6"
DEFAULT_QWEN_CACHE = ROOT / "data" / "processed" / "v6_confirmation_qwen3_embeddings.npz"
DEFAULT_MEDSIGLIP_CACHE = ROOT / "data" / "processed" / "v6_confirmation_medsiglip_embeddings.npz"
DEFAULT_BIOVILT_CACHE = ROOT / "data" / "processed" / "v6_confirmation_biovilt_embeddings.npz"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def stable_signature(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def committed_json(commit: str, path: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return json.loads(result.stdout.decode("utf-8"))


def shuffled_image_assignments(
    target_ids: Sequence[str], *, count: int, seed: int, domain: str
) -> list[dict[str, str]]:
    if len(target_ids) < 2:
        raise ValueError("Shuffled-image controls need at least two target cases.")
    assignments = []
    signatures: set[str] = set()
    for control_index in range(count):
        ordered = sorted(
            target_ids,
            key=lambda case_id: (
                hashlib.sha256(
                    f"{domain}|{seed}|{control_index}|{case_id}".encode("utf-8")
                ).hexdigest(),
                case_id,
            ),
        )
        assignment = {
            case_id: ordered[(index + 1) % len(ordered)]
            for index, case_id in enumerate(ordered)
        }
        if any(source == assigned for source, assigned in assignment.items()):
            raise RuntimeError("A shuffled-image control contains a fixed point.")
        signature = stable_signature(assignment)
        if signature in signatures:
            raise RuntimeError("Shuffled-image controls contain a duplicate assignment.")
        signatures.add(signature)
        assignments.append(assignment)
    return assignments


def embedding_cache_signature(
    *,
    encoder: str,
    revision: str,
    config: Path,
    cohort: Path,
    cases: Path,
    candidate_ids: Sequence[str],
    chunk_ids: Sequence[str] | None = None,
    chunk_texts: Sequence[str] | None = None,
    view_paths: Sequence[Path] | None = None,
    queries: Sequence[str] | None = None,
) -> str:
    return stable_signature(
        {
            "implementation_sha256": file_sha256(Path(__file__)),
            "config_sha256": file_sha256(config),
            "cohort_sha256": file_sha256(cohort),
            "cases_sha256": file_sha256(cases),
            "encoder": encoder,
            "revision": revision,
            "candidate_ids": list(candidate_ids),
            "chunk_ids": list(chunk_ids or []),
            "chunk_texts": list(chunk_texts or []),
            "view_paths": [portable_path(path) for path in (view_paths or [])],
            "queries": list(queries or []),
        }
    )


def load_cache(path: Path, signature: str, names: Sequence[str]) -> tuple[dict[str, np.ndarray], dict[str, float]] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cached:
        if str(cached["signature"].item()) != signature:
            return None
        arrays = {name: cached[name] for name in names}
        runtime = {
            "embedding_build_seconds": float(cached["embedding_build_seconds"].item()),
            "embedding_build_peak_gpu_memory_allocated_mib": float(
                cached["embedding_build_peak_gpu_memory_allocated_mib"].item()
            ),
        }
    return arrays, runtime


def save_cache(
    path: Path,
    signature: str,
    arrays: dict[str, np.ndarray],
    runtime: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        signature=np.asarray(signature),
        **{name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()},
        embedding_build_seconds=np.asarray(runtime["embedding_build_seconds"]),
        embedding_build_peak_gpu_memory_allocated_mib=np.asarray(
            runtime["embedding_build_peak_gpu_memory_allocated_mib"]
        ),
    )


def image_score_maps_max(
    image_embeddings: np.ndarray,
    chunk_embeddings: np.ndarray,
    chunk_case_ids: Sequence[str],
    candidate_ids: Sequence[str],
    source_ids: Sequence[str],
) -> dict[str, dict[str, float]]:
    image_index = {case_id: index for index, case_id in enumerate(candidate_ids)}
    return {
        source_id: maximum_chunk_scores(
            image_embeddings[image_index[source_id]],
            chunk_embeddings,
            chunk_case_ids,
            candidate_ids,
        )
        for source_id in source_ids
    }


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen V6 confirmation retrieval experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol-commit", default="eee7405")
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--cohort-commit", default="43fe1a0")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qwen-cache", type=Path, default=DEFAULT_QWEN_CACHE)
    parser.add_argument("--medsiglip-cache", type=Path, default=DEFAULT_MEDSIGLIP_CACHE)
    parser.add_argument("--biovilt-cache", type=Path, default=DEFAULT_BIOVILT_CACHE)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--text-batch-size", type=int, default=32)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    config = read_json(args.config)
    cohort = read_json(args.cohort)
    if config != committed_json(args.protocol_commit, args.config):
        raise RuntimeError("Confirmation config differs from its frozen protocol commit.")
    if cohort != committed_json(args.cohort_commit, args.cohort):
        raise RuntimeError("Confirmation cohort differs from its frozen cohort commit.")
    summary_path = args.output_dir / "confirmation_retrieval_summary.json"
    if summary_path.exists():
        raise RuntimeError("Formal confirmation retrieval summary already exists; outcome-driven rerun is prohibited.")

    cases = {str(case["case_id"]): case for case in load_cases_jsonl(args.cases)}
    candidate_ids = [str(value) for value in cohort["case_ids"]]
    target_ids = [str(value) for value in cohort["target_case_ids"]]
    questions = list(cohort["questions"])
    if len(candidate_ids) != 240 or len(target_ids) != 120 or len(questions) != 360:
        raise RuntimeError("Frozen V6 confirmation cohort counts changed.")
    if set(str(row["case_id"]) for row in questions) != set(target_ids):
        raise RuntimeError("Confirmation questions differ from target IDs.")

    case_images = resolve_case_images(
        candidate_ids, cases, image_file_lookup(args.image_root)
    )
    view_case_ids = [case_id for case_id in candidate_ids for _ in case_images[case_id]]
    view_paths = [path for case_id in candidate_ids for path in case_images[case_id]]
    queries = [build_text_query(cases[str(row["case_id"])], row) for row in questions]
    question_ids = [str(row["qid"]) for row in questions]

    qwen_config = config["text_retrieval"]["secondary_dense"]
    qwen_signature = embedding_cache_signature(
        encoder=str(qwen_config["model"]),
        revision=str(qwen_config["revision"]),
        config=args.config,
        cohort=args.cohort,
        cases=args.cases,
        candidate_ids=candidate_ids,
        queries=queries,
    )
    qwen_cached = load_cache(args.qwen_cache, qwen_signature, ["documents", "queries"])
    if qwen_cached is None:
        from sentence_transformers import SentenceTransformer

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = SentenceTransformer(
            str(qwen_config["model"]),
            revision=str(qwen_config["revision"]),
            device="cuda",
            cache_folder=str(ROOT / ".hf_cache"),
            model_kwargs={"dtype": torch.float16},
            processor_kwargs={"padding_side": "left"},
        )
        instructed_queries = [
            detailed_instruction(str(qwen_config["instruction"]), query)
            for query in queries
        ]
        started = time.perf_counter()
        documents = np.asarray(
            model.encode(
                [str(cases[case_id].get("report_text", "")) for case_id in candidate_ids],
                batch_size=8,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )
        query_embeddings = np.asarray(
            model.encode(
                instructed_queries,
                batch_size=8,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )
        qwen_runtime = {
            "embedding_build_seconds": time.perf_counter() - started,
            "embedding_build_peak_gpu_memory_allocated_mib": float(
                torch.cuda.max_memory_allocated() / (1024**2)
            ),
        }
        save_cache(
            args.qwen_cache,
            qwen_signature,
            {"documents": documents, "queries": query_embeddings},
            qwen_runtime,
        )
        qwen_cache_used = False
        del model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        arrays, qwen_runtime = qwen_cached
        documents, query_embeddings = arrays["documents"], arrays["queries"]
        qwen_cache_used = True

    medsiglip = MedSiglipEncoder(
        revision=MEDSIGLIP_REVISION,
        cache_dir=ROOT / ".hf_cache",
        local_files_only=True,
    )
    chunks = [
        chunk
        for case_id in candidate_ids
        for chunk in build_report_chunks(
            cases[case_id],
            medsiglip.processor.tokenizer,
            max_tokens=int(config["multimodal_retrieval"]["primary_encoder"]["max_text_tokens"]),
        )
    ]
    chunk_ids = [str(row["chunk_id"]) for row in chunks]
    chunk_case_ids = [str(row["case_id"]) for row in chunks]
    chunk_texts = [str(row["text"]) for row in chunks]
    medsiglip_signature = embedding_cache_signature(
        encoder=MEDSIGLIP_MODEL,
        revision=MEDSIGLIP_REVISION,
        config=args.config,
        cohort=args.cohort,
        cases=args.cases,
        candidate_ids=candidate_ids,
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        view_paths=view_paths,
    )
    medsiglip_cached = load_cache(
        args.medsiglip_cache, medsiglip_signature, ["chunks", "images"]
    )
    if medsiglip_cached is None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        medsiglip_chunks = medsiglip.encode_texts(chunk_texts, batch_size=args.text_batch_size)
        medsiglip_views = medsiglip.encode_images(view_paths, batch_size=args.image_batch_size)
        medsiglip_images = aggregate_case_images(medsiglip_views, view_case_ids, candidate_ids)
        medsiglip_runtime = {
            "embedding_build_seconds": time.perf_counter() - started,
            "embedding_build_peak_gpu_memory_allocated_mib": float(
                torch.cuda.max_memory_allocated() / (1024**2)
            ),
        }
        save_cache(
            args.medsiglip_cache,
            medsiglip_signature,
            {"chunks": medsiglip_chunks, "images": medsiglip_images},
            medsiglip_runtime,
        )
        medsiglip_cache_used = False
    else:
        arrays, medsiglip_runtime = medsiglip_cached
        medsiglip_chunks, medsiglip_images = arrays["chunks"], arrays["images"]
        medsiglip_cache_used = True
    del medsiglip
    gc.collect()
    torch.cuda.empty_cache()

    biovilt_signature = embedding_cache_signature(
        encoder=BIOVILT_MODEL,
        revision=BIOVILT_REVISION,
        config=args.config,
        cohort=args.cohort,
        cases=args.cases,
        candidate_ids=candidate_ids,
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        view_paths=view_paths,
    )
    biovilt_cached = load_cache(
        args.biovilt_cache, biovilt_signature, ["chunks", "images"]
    )
    if biovilt_cached is None:
        biovilt = BioVilTEncoder(text_max_length=64)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        biovilt_chunks = biovilt.encode_texts(chunk_texts, batch_size=args.text_batch_size)
        biovilt_views = biovilt.encode_images(view_paths, batch_size=args.image_batch_size)
        biovilt_images = aggregate_case_images(biovilt_views, view_case_ids, candidate_ids)
        biovilt_runtime = {
            "embedding_build_seconds": time.perf_counter() - started,
            "embedding_build_peak_gpu_memory_allocated_mib": float(
                torch.cuda.max_memory_allocated() / (1024**2)
            ),
        }
        save_cache(
            args.biovilt_cache,
            biovilt_signature,
            {"chunks": biovilt_chunks, "images": biovilt_images},
            biovilt_runtime,
        )
        biovilt_cache_used = False
        del biovilt
        gc.collect()
        torch.cuda.empty_cache()
    else:
        arrays, biovilt_runtime = biovilt_cached
        biovilt_chunks, biovilt_images = arrays["chunks"], arrays["images"]
        biovilt_cache_used = True

    text_rankings, text_scores = build_bm25_inputs(questions, candidate_ids, cases)
    qwen_rankings = dense_rankings(
        question_ids, candidate_ids, query_embeddings, documents
    )
    medsiglip_maps = image_score_maps_max(
        medsiglip_images,
        medsiglip_chunks,
        chunk_case_ids,
        candidate_ids,
        target_ids,
    )
    biovilt_maps = image_score_maps_max(
        biovilt_images,
        biovilt_chunks,
        chunk_case_ids,
        candidate_ids,
        target_ids,
    )
    common = {
        "shortlist_size": int(config["multimodal_retrieval"]["shortlist_size"]),
        "text_weight": float(config["multimodal_retrieval"]["text_weight"]),
    }
    medsiglip_rankings = fused_rankings(
        questions, text_rankings, text_scores, medsiglip_maps, **common
    )
    biovilt_rankings = fused_rankings(
        questions, text_rankings, text_scores, biovilt_maps, **common
    )
    systems = {
        "bm25": text_rankings,
        "qwen3_embedding": qwen_rankings,
        "biovilt_max_chunk_reranker": biovilt_rankings,
        "medsiglip_max_chunk_reranker": medsiglip_rankings,
    }
    metrics: dict[str, dict[str, float]] = {}
    output_rows: list[dict[str, Any]] = []
    for system, rankings in systems.items():
        system_metrics, rows = evaluate_rankings_and_answers(questions, rankings, cases)
        metrics[system] = system_metrics
        output_rows.extend({"system": system, **row} for row in rows)

    shuffled_config = config["shuffled_image_control"]
    assignments = shuffled_image_assignments(
        target_ids,
        count=int(shuffled_config["count"]),
        seed=int(shuffled_config["seed"]),
        domain=str(shuffled_config["order_domain"]),
    )
    image_index = {case_id: index for index, case_id in enumerate(candidate_ids)}
    shuffled_metrics = []
    for control_index, assignment in enumerate(assignments):
        assigned_images = np.stack(
            [medsiglip_images[image_index[assignment[source_id]]] for source_id in target_ids]
        )
        score_maps = {
            source_id: maximum_chunk_scores(
                assigned_images[index],
                medsiglip_chunks,
                chunk_case_ids,
                candidate_ids,
            )
            for index, source_id in enumerate(target_ids)
        }
        rankings = fused_rankings(
            questions, text_rankings, text_scores, score_maps, **common
        )
        control_metrics, _ = evaluate_rankings_and_answers(questions, rankings, cases)
        shuffled_metrics.append(
            {
                "control_index": control_index,
                "assignment_sha256": stable_signature(assignment),
                **control_metrics,
            }
        )

    correct_mrr = float(metrics["medsiglip_max_chunk_reranker"]["mrr"])
    exceedances = sum(float(row["mrr"]) >= correct_mrr for row in shuffled_metrics)
    rows_path = args.output_dir / "confirmation_retrieval_rows.jsonl"
    write_jsonl(rows_path, output_rows)
    summary = {
        "experiment": "V6 model-modernized confirmation retrieval",
        "status": "formal_confirmation_outcomes_frozen",
        "protocol_commit": args.protocol_commit,
        "cohort_commit": args.cohort_commit,
        "config_sha256": file_sha256(args.config),
        "cohort_sha256": file_sha256(args.cohort),
        "implementation_sha256": file_sha256(Path(__file__)),
        "candidate_case_count": len(candidate_ids),
        "target_case_count": len(target_ids),
        "question_count": len(questions),
        "view_count": len(view_paths),
        "chunk_count": len(chunks),
        "max_chunk_tokens": max(int(row["token_count"]) for row in chunks),
        "metrics": metrics,
        "random_image_control": {
            "count": len(shuffled_metrics),
            "fixed_point_count": 0,
            "unique_assignment_count": len({row["assignment_sha256"] for row in shuffled_metrics}),
            "metrics": shuffled_metrics,
            "mrr_exceedance_count": exceedances,
            "plus_one_monte_carlo_p_mrr": (exceedances + 1) / (len(shuffled_metrics) + 1),
        },
        "runtime": {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "qwen3": {"cache_used": qwen_cache_used, **qwen_runtime},
            "medsiglip": {"cache_used": medsiglip_cache_used, **medsiglip_runtime},
            "biovilt": {"cache_used": biovilt_cache_used, **biovilt_runtime},
        },
        "models": {
            "qwen3": qwen_config,
            "medsiglip": config["multimodal_retrieval"]["primary_encoder"],
            "biovilt": config["multimodal_retrieval"]["historical_encoder"],
        },
        "outputs": {
            "rows": portable_path(rows_path),
            "rows_sha256": file_sha256(rows_path),
            "row_count": len(output_rows),
            "summary": portable_path(summary_path),
        },
        "claim_boundary": (
            "Same-source closed-set paired-report retrieval; not diagnosis, external "
            "validation, patient-level independence, or clinical adjudication."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
