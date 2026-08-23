from __future__ import annotations

import argparse
import csv
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
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import file_sha256, read_json  # noqa: E402
from medical_rag.evaluation.graded_retrieval import ndcg_at_k  # noqa: E402
from medical_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.relevance import active_label_weights  # noqa: E402
from train_v9_learned_reranker import (  # noqa: E402
    MLPScorer,
    exact_leave_one_out_bm25_scores,
    feature_matrix,
    relevance_array,
)


DEFAULT_CONFIG = ROOT / "config" / "v9_supplemental_validity.json"
DEFAULT_CONFIRMATION_CONFIG = ROOT / "config" / "v9_retrieval_confirmation.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v9" / "v9_full_source_split.json"
DEFAULT_DEV_EMBEDDINGS = ROOT / "data" / "processed" / "v9_medsiglip_development_embeddings.npz"
DEFAULT_TEST_EMBEDDINGS = ROOT / "data" / "processed" / "v9_medsiglip_test_embeddings.npz"
DEFAULT_DENSE_CACHE = ROOT / "data" / "processed" / "v9_qwen3_supplemental_embeddings.npz"
DEFAULT_CHECKPOINT = (
    ROOT
    / "experiments"
    / "post_submission_v9"
    / "reranker_checkpoints"
    / "v9_mlp_best.pt"
)
DEFAULT_ROWS = ROOT / "experiments" / "post_submission_v9" / "v9_dense_robustness_rows.csv"
DEFAULT_SUMMARY = (
    ROOT / "data" / "splits" / "v9" / "v9_dense_text_robustness_summary.json"
)


def instructed_query(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery:{query}"


def local_model_snapshot(model_id: str, revision: str, cache_root: Path) -> Path:
    snapshot = (
        cache_root
        / f"models--{model_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    required = ("config.json", "model.safetensors", "tokenizer.json", "modules.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Pinned local model snapshot is incomplete: {snapshot}; missing={missing}"
        )
    return snapshot


def stable_order(scores: np.ndarray) -> np.ndarray:
    return np.lexsort((np.arange(len(scores)), -np.asarray(scores)))


def top_k_jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def cache_signature(
    *,
    config_path: Path,
    cases_path: Path,
    candidate_ids: Sequence[str],
    document_texts: Sequence[str],
    query_keys: Sequence[str],
    query_texts: Sequence[str],
    model_id: str,
    revision: str,
) -> str:
    payload = {
        "implementation_sha256": file_sha256(Path(__file__)),
        "config_sha256": file_sha256(config_path),
        "cases_sha256": file_sha256(cases_path),
        "candidate_ids": list(candidate_ids),
        "document_text_sha256": hashlib.sha256(
            "\n\u241e\n".join(document_texts).encode("utf-8")
        ).hexdigest(),
        "query_keys": list(query_keys),
        "query_text_sha256": hashlib.sha256(
            "\n\u241e\n".join(query_texts).encode("utf-8")
        ).hexdigest(),
        "model": model_id,
        "revision": revision,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def aggregate_robustness(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_variant: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    by_role_variant: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    by_case_role: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[int(row["variant_index"])].append(row)
        by_role_variant[(str(row["question_type"]), int(row["variant_index"]))].append(row)
        by_case_role[(str(row["case_id"]), str(row["question_type"]))].append(row)
    variant_metrics = {
        str(index): {
            "questions": {
                str(row["question_type"]): str(row["question"])
                for row in values
            },
            "ndcg@10": statistics.fmean(float(row["ndcg@10"]) for row in values),
        }
        for index, values in sorted(by_variant.items())
    }
    role_variant_metrics: dict[str, dict[str, Any]] = defaultdict(dict)
    for (role, index), values in sorted(by_role_variant.items()):
        role_variant_metrics[role][str(index)] = {
            "question": str(values[0]["question"]),
            "ndcg@10": statistics.fmean(float(row["ndcg@10"]) for row in values),
        }
    top1_values: list[float] = []
    top10_values: list[float] = []
    ndcg_stds: list[float] = []
    for values in by_case_role.values():
        ordered = sorted(values, key=lambda row: int(row["variant_index"]))
        canonical = ordered[0]
        for variant in ordered[1:]:
            top1_values.append(float(variant["top1_case_id"] == canonical["top1_case_id"]))
            top10_values.append(
                top_k_jaccard(
                    str(canonical["top10_case_ids"]).split(";"),
                    str(variant["top10_case_ids"]).split(";"),
                )
            )
        ndcg_stds.append(
            statistics.pstdev(float(row["ndcg@10"]) for row in ordered)
        )
    return {
        "ndcg@10_by_variant": variant_metrics,
        "ndcg@10_by_question_type_and_variant": dict(role_variant_metrics),
        "canonical_ndcg@10": variant_metrics["0"]["ndcg@10"],
        "top1_consistency_with_canonical": statistics.fmean(top1_values),
        "top10_jaccard_with_canonical": statistics.fmean(top10_values),
        "mean_within_case_role_ndcg_standard_deviation": statistics.fmean(ndcg_stds),
        "case_role_count": len(by_case_role),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the prespecified V9 dense baseline and paraphrase audit."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--confirmation-config", type=Path, default=DEFAULT_CONFIRMATION_CONFIG
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--development-embeddings", type=Path, default=DEFAULT_DEV_EMBEDDINGS
    )
    parser.add_argument("--test-embeddings", type=Path, default=DEFAULT_TEST_EMBEDDINGS)
    parser.add_argument("--dense-cache", type=Path, default=DEFAULT_DENSE_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    config = read_json(args.config)
    confirmation = read_json(args.confirmation_config)
    split = read_json(args.split)
    dense_config = config["dense_text_baseline"]
    paraphrases = config["paraphrase_robustness"]["roles"]
    if file_sha256(args.checkpoint) != confirmation["systems"]["r4"]["checkpoint_sha256"]:
        raise RuntimeError("R4 checkpoint differs from the frozen confirmation state.")

    cases_list = read_openi_paired_cases(
        args.cases, source_unique_patient=True, radgraph_path=args.radgraph
    )
    cases = {case.study_id: case for case in cases_list}
    with np.load(args.development_embeddings, allow_pickle=False) as cache:
        candidate_ids = [str(value) for value in cache["candidate_ids"].tolist()]
        bank_images = np.asarray(cache["candidate_image_embeddings"], dtype=np.float32)
        report_means = np.asarray(cache["report_mean_embeddings"], dtype=np.float32)
    with np.load(args.test_embeddings, allow_pickle=False) as cache:
        cached_test_ids = [str(value) for value in cache["test_ids"].tolist()]
        test_images = np.asarray(cache["test_image_embeddings"], dtype=np.float32)
    test_ids = sorted(str(value) for value in split["partitions"]["test"]["case_ids"])
    if cached_test_ids != test_ids:
        raise RuntimeError("Test embedding order differs from the frozen test split.")

    bank = [cases[case_id] for case_id in candidate_ids]
    document_texts = [case.report_text for case in bank]
    query_specs: list[dict[str, Any]] = []
    for case_id in test_ids:
        query_case = cases[case_id]
        for question_type, questions in paraphrases.items():
            for variant_index, question in enumerate(questions):
                query_specs.append(
                    {
                        "key": f"{case_id}:{question_type}:v{variant_index}",
                        "case_id": case_id,
                        "question_type": question_type,
                        "variant_index": variant_index,
                        "question": question,
                        "query_text": query_case.query_text(question),
                    }
                )
    query_keys = [str(row["key"]) for row in query_specs]
    query_texts = [str(row["query_text"]) for row in query_specs]
    model_id = str(dense_config["model"])
    revision = str(dense_config["revision"])
    signature = cache_signature(
        config_path=args.config,
        cases_path=args.cases,
        candidate_ids=candidate_ids,
        document_texts=document_texts,
        query_keys=query_keys,
        query_texts=query_texts,
        model_id=model_id,
        revision=revision,
    )

    cache_used = False
    build_seconds = 0.0
    peak_mib = 0.0
    if args.dense_cache.is_file():
        with np.load(args.dense_cache, allow_pickle=False) as cache:
            if str(cache["signature"].item()) == signature:
                document_embeddings = np.asarray(
                    cache["document_embeddings"], dtype=np.float32
                )
                query_embeddings = np.asarray(cache["query_embeddings"], dtype=np.float32)
                build_seconds = float(cache["build_seconds"].item())
                peak_mib = float(cache["peak_gpu_memory_mib"].item())
                cache_used = True
    if not cache_used:
        from sentence_transformers import SentenceTransformer

        os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
        snapshot = local_model_snapshot(model_id, revision, ROOT / ".hf_cache")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = SentenceTransformer(
            str(snapshot),
            device="cuda" if torch.cuda.is_available() else "cpu",
            cache_folder=str(ROOT / ".hf_cache"),
            local_files_only=True,
            model_kwargs={"dtype": torch.float16} if torch.cuda.is_available() else {},
            processor_kwargs={"padding_side": "left"},
        )
        instructed = [
            instructed_query(str(dense_config["query_instruction"]), query)
            for query in query_texts
        ]
        build_started = time.perf_counter()
        document_embeddings = np.asarray(
            model.encode(
                document_texts,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )
        query_embeddings = np.asarray(
            model.encode(
                instructed,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            ),
            dtype=np.float32,
        )
        build_seconds = time.perf_counter() - build_started
        peak_mib = (
            float(torch.cuda.max_memory_allocated() / (1024**2))
            if torch.cuda.is_available()
            else 0.0
        )
        args.dense_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dense_cache,
            signature=np.asarray(signature),
            document_embeddings=document_embeddings,
            query_embeddings=query_embeddings,
            build_seconds=np.asarray(build_seconds),
            peak_gpu_memory_mib=np.asarray(peak_mib),
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    bm25 = BM25Retriever().fit(
        [{"case_id": case.study_id, "report_text": case.report_text} for case in bank]
    )
    term_cache: dict[str, tuple[np.ndarray, int]] = {}
    prepared_labels = [active_label_weights(case.labels) for case in bank]
    prepared_facts = [case.radgraph_facts for case in bank]
    model = MLPScorer()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()
    test_index = {case_id: index for index, case_id in enumerate(test_ids)}
    dense_index = {key: index for index, key in enumerate(query_keys)}
    audit_rows: list[dict[str, Any]] = []
    current_case_id = ""
    gains: np.ndarray | None = None
    qrels: dict[str, float] = {}
    for spec_index, spec in enumerate(query_specs, start=1):
        case_id = str(spec["case_id"])
        if case_id != current_case_id:
            current_case_id = case_id
            query = cases[case_id]
            gains = relevance_array(
                query,
                bank,
                None,
                prepared_labels=prepared_labels,
                prepared_facts=prepared_facts,
            )
            qrels = dict(zip(candidate_ids, map(float, gains), strict=True))
        image_embedding = test_images[test_index[case_id]]
        image_image = bank_images @ image_embedding
        image_report = report_means @ image_embedding
        text = exact_leave_one_out_bm25_scores(
            bm25,
            str(spec["query_text"]),
            excluded_index=None,
            term_cache=term_cache,
        )
        features = feature_matrix(
            text,
            image_image,
            image_report,
            question_type=str(spec["question_type"]),
            excluded_index=None,
        )
        with torch.inference_mode():
            learned = model(torch.from_numpy(features)).numpy()
        dense = document_embeddings @ query_embeddings[dense_index[str(spec["key"])]]
        channels = {
            "r0_bm25": text,
            "qwen3_dense_text": dense,
            "r4_learned_mlp": learned,
        }
        for system, scores in channels.items():
            top = stable_order(scores)[:10]
            ranking = [candidate_ids[int(index)] for index in top]
            audit_rows.append(
                {
                    "system": system,
                    "case_id": case_id,
                    "question_type": str(spec["question_type"]),
                    "variant_index": int(spec["variant_index"]),
                    "question": str(spec["question"]),
                    "ndcg@10": ndcg_at_k(qrels, ranking, 10),
                    "top1_case_id": ranking[0],
                    "top10_case_ids": ";".join(ranking),
                }
            )
        if spec_index % 1000 == 0 or spec_index == len(query_specs):
            print(f"robustness_queries={spec_index}/{len(query_specs)}", flush=True)

    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    with args.rows_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit_rows)
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        by_system[str(row["system"])].append(row)
    systems = {
        system: aggregate_robustness(rows) for system, rows in sorted(by_system.items())
    }
    dense_difference = (
        systems["qwen3_dense_text"]["canonical_ndcg@10"]
        - systems["r0_bm25"]["canonical_ndcg@10"]
    )
    output = {
        "study": "V9 modern dense baseline and fixed-paraphrase robustness",
        "status": "post_hoc_exploratory_complete_no_model_selection",
        "test_case_count": len(test_ids),
        "candidate_bank_count": len(candidate_ids),
        "query_variant_count": len(query_specs),
        "config_sha256": file_sha256(args.config),
        "source_cases_sha256": file_sha256(args.cases),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "model": {
            "id": model_id,
            "revision": revision,
            "embedding_dimension": int(document_embeddings.shape[1]),
            "query_instruction": dense_config["query_instruction"],
            "instruction_format": "Instruct: {task}\\nQuery:{query}",
            "normalized_embeddings": True,
        },
        "systems": systems,
        "canonical_dense_minus_bm25_ndcg@10": dense_difference,
        "runtime": {
            "cache_used": cache_used,
            "embedding_build_seconds": build_seconds,
            "peak_gpu_memory_mib": peak_mib,
            "total_current_run_seconds": time.perf_counter() - started,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "embedding_cache": {
            "path": args.dense_cache.relative_to(ROOT).as_posix(),
            "signature": signature,
            "sha256": file_sha256(args.dense_cache),
            "committed_to_public_repository": False,
        },
        "rows_output": {
            "path": args.rows_output.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(args.rows_output),
            "committed_to_public_repository": False,
        },
        "claim_boundary": (
            "The questions are fixed researcher-written paraphrases. This audit "
            "measures phrasing sensitivity, not physician-authored external validity."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
