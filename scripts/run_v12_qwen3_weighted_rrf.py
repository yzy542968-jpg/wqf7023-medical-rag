"""Run an isolated V12 Qwen3 dense-channel and weighted-RRF pilot.

This script never reads the V10/V11 Test partition.  It recomputes Qwen3
document/query embeddings for the current V10 Train/Validation frame, builds
candidate pools from BM25, MedCPT, MedSigLIP, and Qwen3, selects one weighted
RRF recipe on the existing Train internal-early-stop role, and evaluates the
frozen recipe on Validation.  A fresh LambdaMART model is also trained on the
selected candidate recipe so candidate generation and learned reranking are
not conflated.

All outputs are development evidence only.  The qrel-v2 and fact-only scores
are report-derived proxies, not physician-adjudicated clinical correctness.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from lightgbm import LGBMRanker, early_stopping
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v10_reranker import augment_r4_features  # noqa: E402
from medical_rag.similar_case.v10_runtime import QUESTIONS, FrozenR5Runtime, r4_feature_matrix  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from medical_rag.similar_case.v11_qrel import prepare_qrel_case, qrel_v2_profile_prepared  # noqa: E402
from medical_rag.similar_case.relevance import active_label_similarity, radgraph_fact_similarity  # noqa: E402
from run_v12_retrieval_pilot import exact_leave_one_out_bm25_scores  # noqa: E402

warnings.filterwarnings("ignore", message="Found 'eval_at' in params")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightgbm")


MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MODEL_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
QWEN_INSTRUCTION = (
    "Given a radiology question and clinical indication, retrieve a chest X-ray "
    "report containing clinically relevant evidence for the question."
)
RRF_K = 60
RRF_SOURCE_TOP_K = 100
RRF_OUTPUT_K = 200
QWEN_MAX_SEQ_LENGTH = 512
QWEN_CACHE_SCHEMA = "v12-qwen3-last-token-512-v1"
MEDCPT_QUERY_CACHE_SCHEMA = "v12-medcpt-query-cls-64-v1"

# The order is part of the exploratory protocol.  The selected recipe is
# chosen on the Train internal role only, never by looking at Validation.
WEIGHT_RECIPES: dict[str, tuple[float, float, float, float]] = {
    "three_channel_baseline": (1.0, 1.0, 1.0, 0.0),
    "four_channel_equal": (1.0, 1.0, 1.0, 1.0),
    "text_heavy": (2.0, 1.0, 1.0, 1.0),
    "visual_heavy": (1.0, 1.0, 2.0, 1.0),
    "qwen_heavy": (1.0, 1.0, 1.0, 2.0),
    "text_qwen_heavy": (2.0, 1.0, 1.0, 2.0),
    "visual_qwen_heavy": (1.0, 1.0, 2.0, 2.0),
    "qwen_only_diagnostic": (0.0, 0.0, 0.0, 1.0),
}

RANKER_CONFIGS: dict[str, dict[str, float | int]] = {
    "default": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 40, "reg_lambda": 1.0},
    "small_regularized": {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 7, "min_child_samples": 40, "reg_lambda": 2.0},
    "deeper": {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 40, "reg_lambda": 1.0},
    "higher_min_child": {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 80, "reg_lambda": 3.0},
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def ndcg(ranked: Sequence[str], qrels: dict[str, float], k: int = 10) -> float:
    def dcg(values: Sequence[float]) -> float:
        return sum((2.0**value - 1.0) / np.log2(index + 2.0) for index, value in enumerate(values[:k]))

    ideal = dcg(sorted(qrels.values(), reverse=True))
    if ideal <= 0.0:
        return 0.0
    return float(dcg([qrels.get(case_id, 0.0) for case_id in ranked]) / ideal)


def spectrum(raw_case: dict[str, Any]) -> str:
    value = str(raw_case.get("problems", "")).strip().lower()
    if value == "normal":
        return "normal"
    if value in {"", "no indexing"}:
        return "indeterminate"
    return "abnormal"


def stable_rank(scores: np.ndarray) -> list[int]:
    return [int(index) for index in np.lexsort((np.arange(len(scores)), -np.asarray(scores, dtype=np.float64)))]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def local_model_snapshot(cache_root: Path) -> Path:
    snapshot = cache_root / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots" / MODEL_REVISION
    required = ("config.json", "model.safetensors", "tokenizer.json", "modules.json")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete local Qwen3 snapshot {snapshot}; missing={missing}")
    return snapshot


def encode_qwen3(
    document_texts: Sequence[str],
    query_texts: Sequence[str],
    *,
    cache_path: Path,
    cache_signature: str,
    cache_root: Path,
    batch_size: int,
    device: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cache:
            cached_signature = str(cache["signature"].item())
            if cached_signature == cache_signature:
                metadata = {
                    "cache_used": True,
                    "build_seconds": float(cache["build_seconds"].item()),
                    "peak_gpu_memory_mib": float(cache["peak_gpu_memory_mib"].item()),
                }
                return (
                    np.asarray(cache["document_embeddings"], dtype=np.float32),
                    np.asarray(cache["query_embeddings"], dtype=np.float32),
                    metadata,
                )
            print(
                f"qwen3_cache_signature_mismatch cached={cached_signature} expected={cache_signature}",
                flush=True,
            )

    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    snapshot = local_model_snapshot(cache_root)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if selected_device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        local_files_only=True,
        padding_side="left",
    )
    model = AutoModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
        dtype=torch.float16 if selected_device == "cuda" else torch.float32,
    ).to(selected_device)
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = False
    print(f"qwen3_model_loaded device={selected_device}", flush=True)
    instructed_queries = [f"Instruct: {QWEN_INSTRUCTION}\nQuery:{text}" for text in query_texts]
    started = time.perf_counter()
    safe_batch_size = min(int(batch_size), 8)

    def encode_in_chunks(values: Sequence[str], label: str) -> np.ndarray:
        chunks: list[np.ndarray] = []
        chunk_size = max(safe_batch_size * 8, 32)
        for start in range(0, len(values), chunk_size):
            values_chunk = list(values[start : start + chunk_size])
            for batch_start in range(0, len(values_chunk), safe_batch_size):
                batch = values_chunk[batch_start : batch_start + safe_batch_size]
                encoded = tokenizer(
                    batch,
                    max_length=QWEN_MAX_SEQ_LENGTH,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(selected_device) for key, value in encoded.items()}
                with torch.inference_mode():
                    output = model(**encoded)
                    mask = encoded["attention_mask"]
                    last_indices = mask.sum(dim=1) - 1
                    pooled = output.last_hidden_state[
                        torch.arange(output.last_hidden_state.shape[0], device=selected_device),
                        last_indices,
                    ]
                    pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
                chunks.append(pooled.cpu().numpy().astype(np.float32, copy=False))
                del output, encoded, pooled
            if start == 0 or (start + chunk_size) % (chunk_size * 10) == 0 or start + chunk_size >= len(values):
                print(f"qwen3_{label}={min(start + chunk_size, len(values))}/{len(values)}", flush=True)
        return np.concatenate(chunks, axis=0)

    document_embeddings = np.asarray(
        encode_in_chunks(document_texts, "documents"),
        dtype=np.float32,
    )
    query_embeddings = np.asarray(
        encode_in_chunks(instructed_queries, "queries"),
        dtype=np.float32,
    )
    build_seconds = time.perf_counter() - started
    peak_mib = float(torch.cuda.max_memory_allocated() / (1024**2)) if selected_device == "cuda" else 0.0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        signature=np.asarray(cache_signature),
        document_embeddings=document_embeddings,
        query_embeddings=query_embeddings,
        build_seconds=np.asarray(build_seconds),
        peak_gpu_memory_mib=np.asarray(peak_mib),
    )
    del model, tokenizer
    gc.collect()
    if selected_device == "cuda":
        torch.cuda.empty_cache()
    return document_embeddings, query_embeddings, {
        "cache_used": False,
        "build_seconds": build_seconds,
        "peak_gpu_memory_mib": peak_mib,
    }


def weighted_rrf(
    ranked_channels: Sequence[Sequence[int]],
    weights: Sequence[float],
    candidate_ids: Sequence[str],
    *,
    source_top_k: int = RRF_SOURCE_TOP_K,
    output_k: int = RRF_OUTPUT_K,
) -> list[str]:
    scores: dict[int, float] = {}
    for channel, weight in zip(ranked_channels, weights, strict=True):
        if weight <= 0.0:
            continue
        for rank, index in enumerate(channel[:source_top_k], start=1):
            scores[int(index)] = scores.get(int(index), 0.0) + float(weight) / (RRF_K + rank)
    ordered = sorted(scores, key=lambda index: (-scores[index], str(candidate_ids[index])))
    return [str(candidate_ids[index]) for index in ordered[:output_k]]


def build_query_state(
    runtime: FrozenR5Runtime,
    case_id: str,
    question_type: str,
    query_image: np.ndarray,
    query_medcpt: np.ndarray,
    query_qwen: np.ndarray,
    qwen_document_embeddings: np.ndarray,
    bank_medcpt: np.ndarray,
    raw_cases: dict[str, dict[str, Any]],
    qrels: np.ndarray,
    formal: dict[str, Any],
    *,
    leave_one_out: bool,
    term_cache: dict[str, tuple[np.ndarray, int]],
) -> dict[str, Any]:
    query = formal[case_id]
    query_text = "\n".join(part for part in (query.indication, QUESTIONS[question_type]) if part)
    bm25 = np.asarray(runtime.bm25.score_all(query_text), dtype=np.float32)
    excluded_index = runtime.candidate_ids.index(case_id) if case_id in runtime.candidate_ids else None
    if leave_one_out:
        bm25 = exact_leave_one_out_bm25_scores(
            runtime.bm25,
            query_text,
            excluded_index=excluded_index,
            term_cache=term_cache,
        )
    image = np.asarray(runtime.candidate_images @ query_image, dtype=np.float32)
    report = np.asarray(runtime.candidate_reports @ query_image, dtype=np.float32)
    if excluded_index is not None:
        image[excluded_index] = -np.inf
        report[excluded_index] = -np.inf
    medcpt = np.asarray(bank_medcpt @ query_medcpt, dtype=np.float32)
    qwen = np.asarray(qwen_document_embeddings @ query_qwen, dtype=np.float32)
    if excluded_index is not None:
        medcpt[excluded_index] = -np.inf
        qwen[excluded_index] = -np.inf
    ranks = {
        "bm25": stable_rank(bm25),
        "medcpt": stable_rank(medcpt),
        "medsiglip": stable_rank(image),
        "qwen3": stable_rank(qwen),
    }
    fact_features = runtime.fact_index.query_features(query_text)
    r5 = augment_r4_features(
        r4_feature_matrix(bm25, image, report, question_type=question_type),
        fact_features,
    )
    recipes = {
        name: weighted_rrf(
            [ranks["bm25"], ranks["medcpt"], ranks["medsiglip"], ranks["qwen3"]],
            weights,
            runtime.candidate_ids,
        )
        for name, weights in WEIGHT_RECIPES.items()
    }
    feature_case_ids = sorted(
        set(case_id for ranking in recipes.values() for case_id in ranking),
        key=lambda candidate_id: runtime.candidate_ids.index(candidate_id),
    )
    feature_indices = [runtime.candidate_ids.index(case_id) for case_id in feature_case_ids]
    return {
        "case_id": case_id,
        "question_type": question_type,
        "spectrum": spectrum(raw_cases[case_id]),
        "feature_case_ids": feature_case_ids,
        "features_by_index": r5[feature_indices],
        "qrels": qrels,
        "recipes": recipes,
    }


def bootstrap_difference(rows: Sequence[dict[str, Any]], system: str, baseline: str) -> dict[str, float | int | bool]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), {}).setdefault(str(row["system"]), []).append(float(row["ndcg@10"]))
    differences = np.asarray(
        [
            mean(grouped[case_id][system]) - mean(grouped[case_id][baseline])
            for case_id in sorted(grouped)
            if grouped[case_id].get(system) and grouped[case_id].get(baseline)
        ],
        dtype=np.float64,
    )
    if not len(differences):
        raise RuntimeError(f"No paired rows for bootstrap comparison {system} vs {baseline}")
    rng = np.random.default_rng(2026)
    sampled = differences[rng.integers(0, len(differences), size=(10000, len(differences)))].mean(axis=1)
    low = float(np.quantile(sampled, 0.025))
    high = float(np.quantile(sampled, 0.975))
    return {
        "difference": float(differences.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "case_count": len(differences),
    }


def matrix_and_labels(
    states: Sequence[dict[str, Any]],
    candidate_ids: Sequence[str],
    recipe_name: str,
    qrel_variant: str = "qrel_v2",
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    matrices: list[np.ndarray] = []
    labels: list[float] = []
    groups: list[int] = []
    for state in states:
        candidate_ids_for_query = state["recipes"][recipe_name]
        feature_index_by_id = {
            case_id: index for index, case_id in enumerate(state["feature_case_ids"])
        }
        indices = [feature_index_by_id[case_id] for case_id in candidate_ids_for_query]
        candidate_index_by_id = {case_id: index for index, case_id in enumerate(candidate_ids)}
        matrices.append(state["features_by_index"][indices])
        qrels = state["qrels_by_variant"][qrel_variant]
        labels.extend(float(qrels[candidate_index_by_id[case_id]]) for case_id in candidate_ids_for_query)
        groups.append(len(indices))
    return np.concatenate(matrices, axis=0), np.asarray(labels, dtype=np.float32), groups


def train_ranker(
    fit_states: Sequence[dict[str, Any]],
    internal_states: Sequence[dict[str, Any]],
    candidate_ids: Sequence[str],
    recipe_name: str,
    config: dict[str, float | int],
    qrel_variant: str = "qrel_v2",
) -> tuple[LGBMRanker, dict[str, Any]]:
    x_fit, y_fit, g_fit = matrix_and_labels(fit_states, candidate_ids, recipe_name, qrel_variant)
    x_internal, y_internal, g_internal = matrix_and_labels(internal_states, candidate_ids, recipe_name, qrel_variant)
    y_fit_rank = np.clip(np.rint(y_fit * 10.0), 0, 10).astype(np.int32)
    y_internal_rank = np.clip(np.rint(y_internal * 10.0), 0, 10).astype(np.int32)
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[10],
        n_estimators=int(config["n_estimators"]),
        learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["num_leaves"]),
        min_child_samples=int(config["min_child_samples"]),
        reg_lambda=float(config["reg_lambda"]),
        random_state=2026,
        verbosity=-1,
    )
    ranker.fit(
        x_fit,
        y_fit_rank,
        group=g_fit,
        eval_set=[(x_internal, y_internal_rank)],
        eval_group=[g_internal],
        callbacks=[early_stopping(25, verbose=False)],
    )
    return ranker, {
        "best_iteration": int(ranker.best_iteration_ or 0),
        "fit_query_count": len(fit_states),
        "internal_query_count": len(internal_states),
        "qrel_variant": qrel_variant,
        "config": config,
    }


def learned_ranking(state: dict[str, Any], ranker: LGBMRanker, recipe_name: str, candidate_ids: Sequence[str]) -> list[str]:
    selected = state["recipes"][recipe_name]
    index_by_id = {case_id: index for index, case_id in enumerate(state["feature_case_ids"])}
    indices = [index_by_id[case_id] for case_id in selected]
    scores = ranker.predict(state["features_by_index"][indices])
    return [case_id for _, case_id in sorted(zip(scores, selected), key=lambda pair: (-float(pair[0]), pair[1]))]


def evaluate_states(
    states: Sequence[dict[str, Any]],
    candidate_ids: Sequence[str],
    recipe_name: str,
    ranker: LGBMRanker | None = None,
    qrel_variant: str = "qrel_v2",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in states:
        ranking = learned_ranking(state, ranker, recipe_name, candidate_ids) if ranker is not None else state["recipes"][recipe_name]
        candidate_index_by_id = {case_id: index for index, case_id in enumerate(candidate_ids)}
        qrels = {
            case_id: float(state["qrels_by_variant"][qrel_variant][index])
            for case_id, index in candidate_index_by_id.items()
        }
        row = {
            "case_id": state["case_id"],
            "question_type": state["question_type"],
            "spectrum": state["spectrum"],
            "system": (
                recipe_name
                if ranker is None
                else f"{recipe_name}_lambdamart"
                if qrel_variant == "qrel_v2"
                else f"{recipe_name}_lambdamart_{qrel_variant}"
            ),
            "ndcg@10": ndcg(ranking, qrels, 10),
            "target_in_top200": float(state["case_id"] in ranking[:200]),
            "top10_case_ids": ranking[:10],
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--radgraph", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_medsiglip_runtime_embeddings.npz")
    parser.add_argument("--medcpt", type=Path, default=ROOT / "data/processed/openi_medcpt_full.npz")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "experiments/v10_publication/reranker_checkpoints")
    parser.add_argument("--cache-root", type=Path, default=ROOT / ".hf_cache")
    parser.add_argument("--qwen-cache", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_qwen3_v10_embeddings.npz")
    parser.add_argument("--medcpt-query-cache", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_medcpt_query_embeddings.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_qwen3_weighted_rrf.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    started = time.perf_counter()

    raw_rows = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_cases = {str(row["case_id"]): row for row in raw_rows}
    formal = {case.study_id: case for case in read_openi_paired_cases(args.cases, source_unique_patient=True, radgraph_path=args.radgraph)}
    radgraph = read_radgraph_case_records(args.radgraph)
    split = read_json(args.split)
    train_ids = [str(case_id) for case_id in split["partitions"]["train"]["case_ids"]]
    validation_ids = [str(case_id) for case_id in split["partitions"]["validation"]["case_ids"]]
    eligible = {case_id for case_id in train_ids + validation_ids if case_id in formal and radgraph[case_id].status == "ok"}
    train_ids = [case_id for case_id in train_ids if case_id in eligible]
    validation_ids = [case_id for case_id in validation_ids if case_id in eligible]
    all_ids = train_ids + validation_ids
    facts_by_case = {case_id: tuple(radgraph[case_id].facts) for case_id in all_ids}
    prepared_by_case = {case_id: prepare_qrel_case(raw_cases[case_id], facts_by_case) for case_id in all_ids}
    print(f"v12_frame_loaded train={len(train_ids)} validation={len(validation_ids)}", flush=True)

    print("v12_embedding_files_open", flush=True)
    with np.load(args.embeddings, allow_pickle=False) as encoded:
        print("v12_medsiglip_archive_opened", flush=True)
        image_ids = [str(value) for value in encoded["case_ids"].tolist()]
        print("v12_medsiglip_case_ids_loaded", flush=True)
        report_ids = [str(value) for value in encoded["report_ids"].tolist()]
        print("v12_medsiglip_report_ids_loaded", flush=True)
        image_matrix = np.asarray(encoded["case_image_embeddings"], dtype=np.float32)
        print("v12_medsiglip_case_embeddings_loaded", flush=True)
        report_matrix = np.asarray(encoded["report_embeddings"], dtype=np.float32)
        print("v12_medsiglip_report_embeddings_loaded", flush=True)
    image_by_id = {case_id: image_matrix[index] for index, case_id in enumerate(image_ids)}
    report_by_id = {case_id: report_matrix[index] for index, case_id in enumerate(report_ids)}
    with np.load(args.medcpt, allow_pickle=False) as encoded:
        medcpt_ids = [str(value) for value in encoded["case_ids"].tolist()]
        medcpt_matrix = np.asarray(encoded["embeddings"], dtype=np.float32)
    medcpt_by_id = {
        case_id: medcpt_matrix[index]
        for index, case_id in enumerate(medcpt_ids)
    }
    bank_medcpt = np.stack([medcpt_by_id[case_id] for case_id in train_ids])

    query_keys = [f"{case_id}:{question_type}" for case_id in all_ids for question_type in QUESTIONS]
    query_texts = [
        "\n".join(part for part in (formal[case_id].indication, QUESTIONS[question_type]) if part)
        for case_id in all_ids
        for question_type in QUESTIONS
    ]
    document_texts = [formal[case_id].report_text for case_id in train_ids]
    print("v12_embedding_files_loaded", flush=True)
    signature_payload = {
        "cache_schema": QWEN_CACHE_SCHEMA,
        "cases_sha256": file_sha256(args.cases),
        "candidate_ids": train_ids,
        "query_keys": query_keys,
        "query_text_sha256": sha256_text("\n\u241e\n".join(query_texts)),
        "document_text_sha256": sha256_text("\n\u241e\n".join(document_texts)),
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
            "instruction": QWEN_INSTRUCTION,
            "max_seq_length": QWEN_MAX_SEQ_LENGTH,
            "pooling": "last_non_padding_token",
        }
    qwen_signature = sha256_text(json.dumps(signature_payload, sort_keys=True, ensure_ascii=True))
    print("qwen_cache_open", flush=True)
    qwen_documents, qwen_queries, qwen_runtime = encode_qwen3(
        document_texts,
        query_texts,
        cache_path=args.qwen_cache,
        cache_signature=qwen_signature,
        cache_root=args.cache_root,
        batch_size=args.batch_size,
        device=args.device,
    )
    print("qwen3_encoding_complete", flush=True)
    # The existing MedCPT bank contains article/report vectors, not query
    # vectors.  Read a separately built, signature-checked query cache so the
    # main process does not load a second Transformer beside the runtime.
    medcpt_query_payload = {
        "cache_schema": MEDCPT_QUERY_CACHE_SCHEMA,
        "cases_sha256": file_sha256(args.cases),
        "split_sha256": file_sha256(args.split),
        "candidate_ids": train_ids,
        "query_keys": query_keys,
        "query_text_sha256": sha256_text("\n\u241e\n".join(query_texts)),
        "model": "ncbi/MedCPT-Query-Encoder",
        "max_length": 64,
        "normalized": True,
    }
    medcpt_query_signature = sha256_text(json.dumps(medcpt_query_payload, sort_keys=True, ensure_ascii=True))
    if not args.medcpt_query_cache.is_file():
        raise RuntimeError(
            "Missing V12 MedCPT query cache. Run scripts/build_v12_medcpt_query_embeddings.py first."
        )
    with np.load(args.medcpt_query_cache, allow_pickle=False) as query_cache:
        cached_signature = str(query_cache["signature"].item())
        if cached_signature != medcpt_query_signature:
            raise RuntimeError(
                "V12 MedCPT query cache signature mismatch; rebuild scripts/build_v12_medcpt_query_embeddings.py."
            )
        cached_keys = [str(value) for value in query_cache["query_keys"].tolist()]
        if cached_keys != query_keys:
            raise RuntimeError("V12 MedCPT query cache key order differs from the current query order.")
        medcpt_queries = np.asarray(query_cache["query_embeddings"], dtype=np.float32)
    print("medcpt_query_cache_verified", flush=True)

    checkpoint_states = [torch.load(args.checkpoints / f"r5_seed_{seed}.pt", map_location="cpu", weights_only=True) for seed in (7041, 7042, 7043, 7044, 7045)]
    r4_state = torch.load(args.checkpoints / "r4.pt", map_location="cpu", weights_only=True)
    runtime = FrozenR5Runtime.build(
        candidate_ids=train_ids,
        cases=formal,
        raw_cases=raw_cases,
        facts_by_case=facts_by_case,
        image_by_id=image_by_id,
        report_by_id=report_by_id,
        checkpoint_states=checkpoint_states,
        r4_checkpoint_state=r4_state,
    )
    roles = read_json(ROOT / "data/splits/v10/v10_reranker_roles.json")
    fit_ids = [str(case_id) for case_id in roles["roles"]["pairwise_fit"]["case_ids"] if case_id in train_ids]
    internal_ids = [str(case_id) for case_id in roles["roles"]["internal_early_stop"]["case_ids"] if case_id in train_ids]
    query_index = {
        (case_id, question_type): index
        for index, (case_id, question_type) in enumerate(
            (item for item in ((case_id, question_type) for case_id in all_ids for question_type in QUESTIONS))
        )
    }
    # The explicit tuple indexing above is intentionally deterministic and
    # keeps both embedding matrices tied to query_keys.
    term_cache: dict[str, tuple[np.ndarray, int]] = {}
    # qrel-v2 is identical for the three question types of one case.  Compute
    # it once per query case and share the compact array across its states;
    # recomputing the same pairwise set operations three times caused large
    # transient allocations in the first implementation.
    qrels_by_case: dict[str, dict[str, np.ndarray]] = {}
    for position, query_case_id in enumerate(all_ids, start=1):
        qrel_v2_values = np.asarray(
            [
                float(
                    qrel_v2_profile_prepared(
                        prepared_by_case[query_case_id],
                        prepared_by_case[candidate_id],
                    )["qrel_v2"]
                )
                for candidate_id in train_ids
            ],
            dtype=np.float32,
        )
        label_values = np.asarray(
            [
                float(active_label_similarity(formal[query_case_id].labels, formal[candidate_id].labels))
                for candidate_id in train_ids
            ],
            dtype=np.float32,
        )
        fact_values = np.asarray(
            [
                float(radgraph_fact_similarity(formal[query_case_id].radgraph_facts, formal[candidate_id].radgraph_facts))
                for candidate_id in train_ids
            ],
            dtype=np.float32,
        )
        qrels_by_case[query_case_id] = {
            "qrel_v2": qrel_v2_values,
            "label_only": label_values,
            "fact_only": fact_values,
        }
        if position % 250 == 0:
            print(f"qrel_cases={position}/{len(all_ids)}", flush=True)

    all_states: list[dict[str, Any]] = []
    for position, case_id in enumerate(all_ids, start=1):
        for question_type in QUESTIONS:
            state = build_query_state(
                runtime,
                case_id,
                question_type,
                image_by_id[case_id],
                medcpt_queries[query_index[(case_id, question_type)]],
                qwen_queries[query_index[(case_id, question_type)]],
                qwen_documents,
                bank_medcpt,
                raw_cases,
                qrels_by_case[case_id]["qrel_v2"],
                formal,
                leave_one_out=case_id in train_ids,
                term_cache=term_cache,
            )
            state["qrels_by_variant"] = qrels_by_case[case_id]
            all_states.append(state)
        if position % 100 == 0:
            print(f"qwen_states={position}/{len(all_ids)}", flush=True)

    fit_set = set(fit_ids)
    internal_set = set(internal_ids)
    validation_set = set(validation_ids)
    state_groups = {
        "fit": [state for state in all_states if state["case_id"] in fit_set],
        "internal": [state for state in all_states if state["case_id"] in internal_set],
        "validation": [state for state in all_states if state["case_id"] in validation_set],
    }
    internal_metrics: dict[str, float] = {}
    for recipe_name in WEIGHT_RECIPES:
        rows = evaluate_states(state_groups["internal"], train_ids, recipe_name)
        internal_metrics[recipe_name] = mean([float(row["ndcg@10"]) for row in rows])
    selected_recipe = max(WEIGHT_RECIPES, key=lambda name: (internal_metrics[name], -list(WEIGHT_RECIPES).index(name)))
    validation_rows: list[dict[str, Any]] = []
    for recipe_name in WEIGHT_RECIPES:
        validation_rows.extend(evaluate_states(state_groups["validation"], train_ids, recipe_name))
    ranker_candidates: dict[str, dict[str, Any]] = {}
    trained_rankers: dict[str, LGBMRanker] = {}
    for ranker_name, ranker_config in RANKER_CONFIGS.items():
        candidate_ranker, candidate_meta = train_ranker(
            state_groups["fit"],
            state_groups["internal"],
            train_ids,
            selected_recipe,
            ranker_config,
        )
        internal_rows = evaluate_states(
            state_groups["internal"],
            train_ids,
            selected_recipe,
            candidate_ranker,
        )
        ranker_candidates[ranker_name] = {
            "internal_qrel_v2_ndcg10": mean([float(row["ndcg@10"]) for row in internal_rows]),
            **candidate_meta,
        }
        trained_rankers[ranker_name] = candidate_ranker
    selected_ranker_name = max(
        RANKER_CONFIGS,
        key=lambda name: (
            float(ranker_candidates[name]["internal_qrel_v2_ndcg10"]),
            -list(RANKER_CONFIGS).index(name),
        ),
    )
    selected_ranker = trained_rankers[selected_ranker_name]
    ranker_meta = {
        "selection_rule": "maximum Train internal-early-stop qrel-v2 nDCG@10; deterministic declaration-order tie-break",
        "selected_model": selected_ranker_name,
        "candidates": ranker_candidates,
        **ranker_candidates[selected_ranker_name],
    }
    selected_lambdamart_rows = evaluate_states(state_groups["validation"], train_ids, selected_recipe, selected_ranker)
    validation_rows.extend(selected_lambdamart_rows)

    # Train proxy-specific rankers as a sensitivity analysis.  These models
    # are selected only on the corresponding Train internal proxy and are not
    # mixed into the primary qrel-v2 ranker selection.
    proxy_ranker_results: dict[str, dict[str, Any]] = {}
    for qrel_variant in ("label_only", "fact_only"):
        proxy_candidates: dict[str, dict[str, Any]] = {}
        proxy_models: dict[str, LGBMRanker] = {}
        for ranker_name, ranker_config in RANKER_CONFIGS.items():
            proxy_ranker, proxy_meta = train_ranker(
                state_groups["fit"],
                state_groups["internal"],
                train_ids,
                selected_recipe,
                ranker_config,
                qrel_variant=qrel_variant,
            )
            internal_rows = evaluate_states(
                state_groups["internal"],
                train_ids,
                selected_recipe,
                proxy_ranker,
                qrel_variant=qrel_variant,
            )
            proxy_candidates[ranker_name] = {
                "internal_ndcg10": mean([float(row["ndcg@10"]) for row in internal_rows]),
                **proxy_meta,
            }
            proxy_models[ranker_name] = proxy_ranker
        selected_proxy_name = max(
            RANKER_CONFIGS,
            key=lambda name: (
                float(proxy_candidates[name]["internal_ndcg10"]),
                -list(RANKER_CONFIGS).index(name),
            ),
        )
        selected_proxy_model = proxy_models[selected_proxy_name]
        proxy_validation_metrics: dict[str, float] = {}
        for evaluation_variant in ("qrel_v2", "label_only", "fact_only"):
            proxy_rows = evaluate_states(
                state_groups["validation"],
                train_ids,
                selected_recipe,
                selected_proxy_model,
                qrel_variant=evaluation_variant,
            )
            proxy_validation_metrics[evaluation_variant] = mean(
                [float(row["ndcg@10"]) for row in proxy_rows]
            )
        proxy_model_path = args.output.with_name(f"v12_qwen3_lambdamart_{qrel_variant}.txt")
        proxy_model_path.parent.mkdir(parents=True, exist_ok=True)
        selected_proxy_model.booster_.save_model(str(proxy_model_path))
        proxy_ranker_results[qrel_variant] = {
            "selection_rule": "maximum Train internal nDCG for the corresponding proxy; deterministic declaration-order tie-break",
            "selected_model": selected_proxy_name,
            "candidates": proxy_candidates,
            "validation_ndcg10_by_evaluation_proxy": proxy_validation_metrics,
            "model_path": str(proxy_model_path.relative_to(ROOT)),
            "model_sha256": file_sha256(proxy_model_path),
        }

    summary: dict[str, Any] = {
        "study": "V12 Qwen3 dense channel and weighted RRF pilot",
        "status": "validation_only_development",
        "no_test_evaluation": True,
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "radgraph_sha256": file_sha256(args.radgraph),
            "split_sha256": file_sha256(args.split),
            "medsiglip_embeddings_sha256": file_sha256(args.embeddings),
            "medcpt_sha256": file_sha256(args.medcpt),
            "train_case_count": len(train_ids),
            "validation_case_count": len(validation_ids),
            "fit_case_count": len(fit_ids),
            "internal_case_count": len(internal_ids),
        },
        "qwen3": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "instruction": QWEN_INSTRUCTION,
            "document": "full_historical_report",
            "query": "indication_newline_question",
            "embedding_dimension": int(qwen_documents.shape[1]),
            "cache_signature": qwen_signature,
            "cache_sha256": file_sha256(args.qwen_cache),
            "runtime": qwen_runtime,
        },
        "rrf": {
            "k": RRF_K,
            "source_top_k": RRF_SOURCE_TOP_K,
            "output_k": RRF_OUTPUT_K,
            "channels": ["bm25", "medcpt", "medsiglip", "qwen3"],
            "weight_recipes": WEIGHT_RECIPES,
            "selection_role": "maximum Train internal-early-stop qrel-v2 nDCG@10; deterministic declaration-order tie-break",
            "selected_recipe": selected_recipe,
            "internal_qrel_v2_ndcg10": internal_metrics,
        },
        "metrics": {},
        "bootstrap_vs_baselines": {},
        "ranker": {
            "type": "LightGBM LambdaMART",
            "version": "4.7.0",
            "features": 17,
            "candidate_pool": f"{selected_recipe} Top-200",
            "qrel": "qrel-v2 full report-derived proxy",
            **ranker_meta,
        },
        "proxy_specific_rankers": proxy_ranker_results,
        "rows_path": str(args.output.with_name("v12_qwen3_weighted_rrf_rows.jsonl").resolve().relative_to(ROOT)),
        "claim_boundary": "Exploratory Validation-only development evidence. No score is confirmation evidence or physician-adjudicated clinical correctness.",
    }
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in validation_rows:
        grouped_rows.setdefault(str(row["system"]), []).append(row)
    for system, rows in sorted(grouped_rows.items()):
        summary["metrics"][system] = {
            "qrel_v2_ndcg10": mean([float(row["ndcg@10"]) for row in rows]),
            "normal_ndcg10": mean([float(row["ndcg@10"]) for row in rows if row["spectrum"] == "normal"]),
            "abnormal_ndcg10": mean([float(row["ndcg@10"]) for row in rows if row["spectrum"] == "abnormal"]),
            "indeterminate_ndcg10": mean([float(row["ndcg@10"]) for row in rows if row["spectrum"] == "indeterminate"]),
        }
    baseline = "three_channel_baseline"
    for system in sorted(grouped_rows):
        if system == baseline:
            continue
        summary["bootstrap_vs_baselines"][system] = bootstrap_difference(
            [row for row in validation_rows if row["system"] in {system, baseline}],
            system,
            baseline,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_path = args.output.with_name("v12_qwen3_weighted_rrf_rows.jsonl")
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in validation_rows), encoding="utf-8")
    ranker_path = args.output.with_name("v12_qwen3_lambdamart.txt")
    selected_ranker.booster_.save_model(str(ranker_path))
    summary["ranker"]["model_path"] = str(ranker_path.relative_to(ROOT))
    summary["ranker"]["model_sha256"] = file_sha256(ranker_path)
    summary["rows_sha256"] = file_sha256(rows_path)
    summary["runtime_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"selected_recipe": selected_recipe, "metrics": summary["metrics"], "bootstrap_vs_baselines": summary["bootstrap_vs_baselines"]}, indent=2))


if __name__ == "__main__":
    main()
