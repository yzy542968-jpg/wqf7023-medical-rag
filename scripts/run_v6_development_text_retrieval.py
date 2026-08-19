from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.multimodal.evaluation import (  # noqa: E402
    build_text_query,
    evaluate_rankings_and_answers,
)
from medical_rag.multimodal.fusion import rank_scores  # noqa: E402
from medical_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v6_development.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_COHORT = ROOT / "data" / "processed" / "openi_multimodal_v5_cohort.json"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "post_submission_v6"
DEFAULT_CACHE = ROOT / "data" / "processed" / "v6_qwen3_text_embeddings.npz"


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


def detailed_instruction(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery:{query}"


def select_text_retriever(
    bm25_mrr: float,
    qwen3_mrr: float,
    tie_tolerance: float,
) -> dict[str, Any]:
    difference = float(qwen3_mrr - bm25_mrr)
    selected = "qwen3_embedding" if difference >= tie_tolerance else "bm25"
    return {
        "selected_text_retriever": selected,
        "qwen3_minus_bm25_mrr": difference,
        "tie_tolerance": float(tie_tolerance),
        "rule": (
            "select qwen3_embedding only when qwen3_minus_bm25_mrr "
            ">= tie_tolerance; otherwise select bm25"
        ),
    }


def build_queries(
    questions: Sequence[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    *,
    use_indication: bool,
) -> list[str]:
    if use_indication:
        return [
            build_text_query(cases[str(row["case_id"])], row)
            for row in questions
        ]
    return [str(row["question"]) for row in questions]


def bm25_rankings(
    candidate_cases: list[dict[str, Any]],
    questions: Sequence[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    *,
    use_indication: bool,
) -> dict[str, list[str]]:
    retriever = BM25Retriever(k1=1.5, b=0.75).fit(candidate_cases)
    queries = build_queries(questions, cases, use_indication=use_indication)
    return {
        str(row["qid"]): [
            str(result["case_id"])
            for result in retriever.search(query, top_k=len(candidate_cases))
        ]
        for row, query in zip(questions, queries, strict=True)
    }


def dense_rankings(
    question_ids: Sequence[str],
    candidate_ids: Sequence[str],
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
) -> dict[str, list[str]]:
    if len(question_ids) != len(query_embeddings):
        raise ValueError("Question IDs and query embeddings must have equal length.")
    if len(candidate_ids) != len(document_embeddings):
        raise ValueError("Candidate IDs and document embeddings must have equal length.")
    similarities = np.asarray(query_embeddings) @ np.asarray(document_embeddings).T
    return {
        str(qid): rank_scores(candidate_ids, similarities[index].tolist())
        for index, qid in enumerate(question_ids)
    }


def cache_signature(
    *,
    config_path: Path,
    cohort_path: Path,
    cases_path: Path,
    model_id: str,
    model_revision: str,
    candidate_ids: Sequence[str],
    indication_queries: Sequence[str],
    question_only_queries: Sequence[str],
) -> str:
    payload = {
        "implementation_sha256": file_sha256(Path(__file__)),
        "config_sha256": file_sha256(config_path),
        "cohort_sha256": file_sha256(cohort_path),
        "cases_sha256": file_sha256(cases_path),
        "model_id": model_id,
        "model_revision": model_revision,
        "candidate_ids": list(candidate_ids),
        "indication_queries": list(indication_queries),
        "question_only_queries": list(question_only_queries),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen V6 BM25 versus Qwen3-Embedding development comparison."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    config = read_json(args.config)
    cohort = read_json(args.cohort)
    source_cases = load_cases_jsonl(args.cases)
    cases = {str(case["case_id"]): case for case in source_cases}

    candidate_ids = [str(value) for value in cohort["case_ids"]]
    candidate_cases = [cases[case_id] for case_id in candidate_ids]
    development_ids = {
        str(value) for value in cohort["split"]["development"]["case_ids"]
    }
    questions = [
        row for row in cohort["questions"] if str(row["case_id"]) in development_ids
    ]
    if len(candidate_ids) != 240 or len(development_ids) != 120 or len(questions) != 360:
        raise RuntimeError("V6 development source no longer matches the frozen protocol.")

    indication_queries = build_queries(questions, cases, use_indication=True)
    question_only_queries = build_queries(questions, cases, use_indication=False)
    document_texts = [str(cases[case_id].get("report_text", "")) for case_id in candidate_ids]

    model_id = str(config["text_retrieval"]["candidates"][1])
    from huggingface_hub import model_info

    model_revision = str(model_info(model_id).sha)
    signature = cache_signature(
        config_path=args.config,
        cohort_path=args.cohort,
        cases_path=args.cases,
        model_id=model_id,
        model_revision=model_revision,
        candidate_ids=candidate_ids,
        indication_queries=indication_queries,
        question_only_queries=question_only_queries,
    )

    embedding_runtime_seconds = 0.0
    embedding_build_runtime_seconds = 0.0
    embedding_build_peak_gpu_memory_mib = 0.0
    cache_used = False
    if args.cache.is_file():
        cached = np.load(args.cache, allow_pickle=False)
        cached_signature = str(cached["signature"].item())
        if cached_signature == signature:
            document_embeddings = cached["document_embeddings"]
            indication_embeddings = cached["indication_embeddings"]
            question_only_embeddings = cached["question_only_embeddings"]
            embedding_build_runtime_seconds = float(
                cached["embedding_build_runtime_seconds"].item()
            )
            embedding_build_peak_gpu_memory_mib = float(
                cached["embedding_build_peak_gpu_memory_mib"].item()
            )
            cache_used = True
        else:
            cached.close()

    if not cache_used:
        from sentence_transformers import SentenceTransformer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = SentenceTransformer(
            model_id,
            revision=model_revision,
            device="cuda" if torch.cuda.is_available() else "cpu",
            cache_folder=str(ROOT / ".hf_cache"),
            model_kwargs={"dtype": torch.float16} if torch.cuda.is_available() else {},
            processor_kwargs={"padding_side": "left"},
        )
        task = str(config["text_retrieval"]["dense_query_instruction"])
        instructed_indication_queries = [
            detailed_instruction(task, query) for query in indication_queries
        ]
        instructed_question_only_queries = [
            detailed_instruction(task, query) for query in question_only_queries
        ]

        started = time.perf_counter()
        document_embeddings = model.encode(
            document_texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        indication_embeddings = model.encode(
            instructed_indication_queries,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        question_only_embeddings = model.encode(
            instructed_question_only_queries,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        document_embeddings = np.asarray(document_embeddings, dtype=np.float32)
        indication_embeddings = np.asarray(indication_embeddings, dtype=np.float32)
        question_only_embeddings = np.asarray(
            question_only_embeddings, dtype=np.float32
        )
        embedding_runtime_seconds = time.perf_counter() - started
        embedding_build_runtime_seconds = embedding_runtime_seconds
        embedding_build_peak_gpu_memory_mib = (
            float(torch.cuda.max_memory_allocated() / (1024**2))
            if torch.cuda.is_available()
            else 0.0
        )
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.cache,
            signature=np.asarray(signature),
            document_embeddings=document_embeddings,
            indication_embeddings=indication_embeddings,
            question_only_embeddings=question_only_embeddings,
            embedding_build_runtime_seconds=np.asarray(
                embedding_build_runtime_seconds
            ),
            embedding_build_peak_gpu_memory_mib=np.asarray(
                embedding_build_peak_gpu_memory_mib
            ),
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    question_ids = [str(row["qid"]) for row in questions]
    rankings = {
        "indication_question_bm25": bm25_rankings(
            candidate_cases, questions, cases, use_indication=True
        ),
        "question_only_bm25": bm25_rankings(
            candidate_cases, questions, cases, use_indication=False
        ),
        "indication_question_qwen3_embedding": dense_rankings(
            question_ids,
            candidate_ids,
            indication_embeddings,
            document_embeddings,
        ),
        "question_only_qwen3_embedding": dense_rankings(
            question_ids,
            candidate_ids,
            question_only_embeddings,
            document_embeddings,
        ),
    }

    metrics: dict[str, dict[str, float]] = {}
    all_rows: list[dict[str, Any]] = []
    for system, system_rankings in rankings.items():
        system_metrics, rows = evaluate_rankings_and_answers(
            questions, system_rankings, cases
        )
        metrics[system] = system_metrics
        all_rows.extend({"system": system, **row} for row in rows)

    selection = select_text_retriever(
        metrics["indication_question_bm25"]["mrr"],
        metrics["indication_question_qwen3_embedding"]["mrr"],
        float(config["text_retrieval"]["tie_tolerance"]),
    )
    summary = {
        "experiment": "V6 development text retriever selection",
        "protocol": "docs/V6_DEVELOPMENT_PROTOCOL.md",
        "config_sha256": file_sha256(args.config),
        "implementation_sha256": file_sha256(Path(__file__)),
        "cohort_sha256": file_sha256(args.cohort),
        "source_cases_sha256": file_sha256(args.cases),
        "model": {
            "id": model_id,
            "revision": model_revision,
            "embedding_dimension": int(document_embeddings.shape[1]),
            "dtype_on_disk": str(document_embeddings.dtype),
            "query_instruction": config["text_retrieval"]["dense_query_instruction"],
            "instruction_format": "Instruct: {task}\\nQuery:{query}",
            "document_instruction": None,
        },
        "development": {
            "target_case_count": len(development_ids),
            "candidate_case_count": len(candidate_ids),
            "question_count": len(questions),
        },
        "metrics": metrics,
        "selection": selection,
        "runtime": {
            "embedding_cache_used": cache_used,
            "current_run_embedding_seconds": embedding_runtime_seconds,
            "embedding_build_runtime_seconds": embedding_build_runtime_seconds,
            "embedding_build_peak_gpu_memory_allocated_mib": (
                embedding_build_peak_gpu_memory_mib
            ),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "cache": {
            "path": portable_path(args.cache),
            "signature": signature,
            "sha256": file_sha256(args.cache),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "development_text_retrieval_summary.json"
    rows_path = args.output_dir / "development_text_retrieval_rows.jsonl"
    write_jsonl(rows_path, all_rows)
    summary["outputs"] = {
        "summary": portable_path(summary_path),
        "rows": portable_path(rows_path),
        "row_count": len(all_rows),
        "rows_sha256": file_sha256(rows_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
