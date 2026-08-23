from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import file_sha256, read_json  # noqa: E402
from medical_rag.evaluation.graded_retrieval import (  # noqa: E402
    binary_recall_at_k,
    ndcg_at_k,
    reciprocal_rank_at_threshold,
)
from medical_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from medical_rag.retrieval.tfidf_retriever import _tokens  # noqa: E402
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.relevance import (  # noqa: E402
    active_label_weights,
)


DEFAULT_CONFIG = ROOT / "config" / "v9_learned_reranker_development.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_ROLES = ROOT / "data" / "splits" / "v9" / "v9_reranker_role_manifest.json"
DEFAULT_EMBEDDINGS = ROOT / "data" / "processed" / "v9_medsiglip_development_embeddings.npz"
DEFAULT_MEDSIGLIP_SUMMARY = ROOT / "data" / "splits" / "v9" / "v9_medsiglip_validation_summary.json"
DEFAULT_PROTOCOL = ROOT / "config" / "v9_similar_case_rag_development.json"
DEFAULT_ROWS = ROOT / "experiments" / "post_submission_v9" / "v9_learned_reranker_validation_rows.jsonl"
DEFAULT_CHECKPOINT_DIR = ROOT / "experiments" / "post_submission_v9" / "reranker_checkpoints"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v9" / "v9_learned_reranker_validation_summary.json"


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def exact_leave_one_out_bm25_scores(
    retriever: BM25Retriever,
    query: str,
    *,
    excluded_index: int | None,
    term_cache: dict[str, tuple[np.ndarray, int]] | None = None,
) -> np.ndarray:
    """Score with exact leave-one-out corpus statistics when an index is excluded."""

    count = len(retriever.cases)
    if count == 0:
        raise RuntimeError("BM25 retriever is not fitted.")
    if excluded_index is not None and not 0 <= excluded_index < count:
        raise IndexError(excluded_index)
    effective_count = count - (1 if excluded_index is not None else 0)
    total_length = sum(retriever.doc_lengths)
    if excluded_index is not None:
        total_length -= retriever.doc_lengths[excluded_index]
    average_length = total_length / effective_count if effective_count else 0.0
    scores = np.zeros(count, dtype=np.float64)
    query_terms = _tokens(query)
    frequencies = Counter(query_terms)
    for term, query_frequency in frequencies.items():
        cached = term_cache.get(term) if term_cache is not None else None
        if cached is None:
            term_frequency = np.fromiter(
                (counts.get(term, 0) for counts in retriever.doc_term_counts),
                dtype=np.float64,
                count=count,
            )
            document_frequency = int(np.count_nonzero(term_frequency))
            if term_cache is not None:
                term_cache[term] = (term_frequency, document_frequency)
        else:
            term_frequency, document_frequency = cached
        if excluded_index is not None and term in retriever.doc_term_counts[excluded_index]:
            document_frequency -= 1
        idf = math.log(
            1 + (effective_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        denominator = term_frequency + retriever.k1 * (
            1 - retriever.b
            + retriever.b * np.asarray(retriever.doc_lengths, dtype=np.float64) / average_length
        )
        contribution = np.divide(
            term_frequency * (retriever.k1 + 1),
            denominator,
            out=np.zeros_like(term_frequency),
            where=denominator > 0,
        )
        scores += query_frequency * idf * contribution
    if excluded_index is not None:
        scores[excluded_index] = -np.inf
    return scores


def normalized_scores_and_reciprocal_ranks(
    scores: np.ndarray, *, excluded_index: int | None
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(values)
    if excluded_index is not None:
        valid[excluded_index] = False
    normalized = np.zeros(len(values), dtype=np.float32)
    finite = values[valid]
    if finite.size and float(finite.max() - finite.min()) > 1e-12:
        normalized[valid] = ((finite - finite.min()) / (finite.max() - finite.min())).astype(
            np.float32
        )
    order = sorted(
        np.flatnonzero(valid).tolist(), key=lambda index: (-float(values[index]), index)
    )
    reciprocal = np.zeros(len(values), dtype=np.float32)
    for rank, index in enumerate(order, start=1):
        reciprocal[index] = 1.0 / rank
    return normalized, reciprocal


def relevance_array(
    query: Any,
    bank: Sequence[Any],
    excluded_index: int | None,
    *,
    prepared_labels: Sequence[Mapping[str, float]] | None = None,
    prepared_facts: Sequence[frozenset[str]] | None = None,
) -> np.ndarray:
    query_labels = active_label_weights(query.labels)
    query_facts = query.radgraph_facts
    gains = np.empty(len(bank), dtype=np.float32)
    for index, candidate in enumerate(bank):
        if excluded_index is not None and index == excluded_index:
            gains[index] = np.nan
            continue
        candidate_labels = (
            prepared_labels[index]
            if prepared_labels is not None
            else active_label_weights(candidate.labels)
        )
        label_keys = set(query_labels) | set(candidate_labels)
        if not label_keys:
            label = 1.0
        else:
            numerator = sum(
                min(query_labels.get(key, 0.0), candidate_labels.get(key, 0.0))
                for key in label_keys
            )
            denominator = sum(
                max(query_labels.get(key, 0.0), candidate_labels.get(key, 0.0))
                for key in label_keys
            )
            label = numerator / denominator if denominator else 0.0
        candidate_facts = (
            prepared_facts[index] if prepared_facts is not None else candidate.radgraph_facts
        )
        if not query_facts and not candidate_facts:
            facts = 1.0
        elif not query_facts or not candidate_facts:
            facts = 0.0
        else:
            overlap = len(query_facts & candidate_facts)
            facts = (
                2.0 * overlap / (len(query_facts) + len(candidate_facts))
                if overlap
                else 0.0
            )
        gains[index] = 0.60 * label + 0.40 * facts
    return gains


def feature_matrix(
    bm25: np.ndarray,
    image_image: np.ndarray,
    image_report: np.ndarray,
    *,
    question_type: str,
    excluded_index: int | None,
) -> np.ndarray:
    components = [
        normalized_scores_and_reciprocal_ranks(values, excluded_index=excluded_index)
        for values in (bm25, image_image, image_report)
    ]
    question = {
        "findings": (1.0, 0.0, 0.0),
        "impression": (0.0, 1.0, 0.0),
        "acute": (0.0, 0.0, 1.0),
    }[question_type]
    indicators = np.tile(np.asarray(question, dtype=np.float32), (len(bm25), 1))
    return np.column_stack(
        [components[0][0], components[1][0], components[2][0],
         components[0][1], components[1][1], components[2][1], indicators]
    ).astype(np.float32)


def stable_top(values: np.ndarray, count: int, *, largest: bool, valid: np.ndarray) -> list[int]:
    indices = np.flatnonzero(valid).tolist()
    return sorted(
        indices,
        key=lambda index: ((-1 if largest else 1) * float(values[index]), index),
    )[:count]


def sample_pairs(
    features: np.ndarray,
    gains: np.ndarray,
    component_scores: Sequence[np.ndarray],
    *,
    config: Mapping[str, Any],
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    valid = np.isfinite(gains)
    pool: set[int] = set()
    per_component = (
        int(config["top_per_bm25"]),
        int(config["top_per_image_image"]),
        int(config["top_per_image_report"]),
    )
    for scores, count in zip(component_scores, per_component, strict=True):
        pool.update(stable_top(scores, count, largest=True, valid=valid))
    pool.update(stable_top(gains, int(config["top_per_relevance"]), largest=True, valid=valid))
    pool.update(stable_top(gains, int(config["bottom_per_relevance"]), largest=False, valid=valid))
    ordered_high = sorted(pool, key=lambda index: (-float(gains[index]), index))[
        : int(config["high_candidates"])
    ]
    ordered_low = sorted(pool, key=lambda index: (float(gains[index]), index))[
        : int(config["low_candidates"])
    ]
    high_rows: list[np.ndarray] = []
    low_rows: list[np.ndarray] = []
    weights: list[float] = []
    minimum = float(config["minimum_gain_difference"])
    for high in ordered_high:
        for low in ordered_low:
            difference = float(gains[high] - gains[low])
            if difference + 1e-12 < minimum:
                continue
            high_rows.append(features[high])
            low_rows.append(features[low])
            weights.append(difference)
    return high_rows, low_rows, weights


class LinearScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Linear(9, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


class MLPScorer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(-1)


def numeric_ndcg10(gains: np.ndarray, ranking: np.ndarray) -> float:
    valid_gains = gains[np.isfinite(gains)]
    ideal = np.sort(valid_gains)[::-1][:10]
    observed = gains[ranking[:10]]
    discounts = np.log2(np.arange(2, len(ideal) + 2, dtype=np.float64))
    ideal_dcg = float(np.sum((np.power(2.0, ideal) - 1.0) / discounts))
    if ideal_dcg == 0.0:
        return 0.0
    observed_dcg = float(np.sum((np.power(2.0, observed) - 1.0) / discounts[: len(observed)]))
    return observed_dcg / ideal_dcg


def evaluate_internal(model: nn.Module, states: Sequence[dict[str, Any]]) -> float:
    model.eval()
    values: list[float] = []
    with torch.inference_mode():
        for state in states:
            for features in state["features"]:
                scores = model(torch.from_numpy(features)).numpy()
                if state["excluded_index"] is not None:
                    scores[state["excluded_index"]] = -np.inf
                ranking = np.lexsort((np.arange(len(scores)), -scores))
                values.append(numeric_ndcg10(state["gains"], ranking))
    return statistics.fmean(values)


def train_architecture(
    name: str,
    high: np.ndarray,
    low: np.ndarray,
    weights: np.ndarray,
    internal_states: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    checkpoint_dir: Path,
    seed: int,
) -> dict[str, Any]:
    set_determinism(seed + (0 if name == "linear" else 1))
    model: nn.Module = LinearScorer() if name == "linear" else MLPScorer()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    high_tensor = torch.from_numpy(high)
    low_tensor = torch.from_numpy(low)
    weight_tensor = torch.from_numpy(weights)
    generator = torch.Generator().manual_seed(seed + (10 if name == "linear" else 11))
    best_metric = -math.inf
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"v9_{name}_best.pt"
    for epoch in range(1, int(config["maximum_epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(high_tensor), generator=generator)
        losses: list[float] = []
        for start in range(0, len(permutation), int(config["batch_size"])):
            indices = permutation[start : start + int(config["batch_size"])]
            optimizer.zero_grad(set_to_none=True)
            margin = model(high_tensor[indices]) - model(low_tensor[indices])
            loss = (F.softplus(-margin) * weight_tensor[indices]).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        metric = evaluate_internal(model, internal_states)
        history.append(
            {"epoch": float(epoch), "loss": statistics.fmean(losses), "internal_ndcg@10": metric}
        )
        print(f"architecture={name} epoch={epoch} loss={history[-1]['loss']:.6f} internal_ndcg10={metric:.6f}", flush=True)
        if metric >= best_metric + float(config["early_stopping_minimum_improvement"]):
            best_metric = metric
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= int(config["early_stopping_patience"]):
            break
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return {
        "name": name,
        "model": model,
        "best_epoch": best_epoch,
        "best_internal_ndcg@10": best_metric,
        "history": history,
        "checkpoint": checkpoint,
        "checkpoint_sha256": file_sha256(checkpoint),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen V9 learned paired reranker.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--medsiglip-summary", type=Path, default=DEFAULT_MEDSIGLIP_SUMMARY)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(args.config)
    roles = read_json(args.roles)
    protocol = read_json(args.protocol)
    medsiglip = read_json(args.medsiglip_summary)
    set_determinism(int(config["seed"]))

    cases_list = read_openi_paired_cases(
        args.cases, source_unique_patient=True, radgraph_path=args.radgraph
    )
    cases = {case.study_id: case for case in cases_list}
    with np.load(args.embeddings, allow_pickle=False) as cache:
        candidate_ids = [str(value) for value in cache["candidate_ids"].tolist()]
        validation_ids = [str(value) for value in cache["validation_ids"].tolist()]
        bank_images = np.asarray(cache["candidate_image_embeddings"], dtype=np.float32)
        validation_images = np.asarray(cache["validation_image_embeddings"], dtype=np.float32)
        report_means = np.asarray(cache["report_mean_embeddings"], dtype=np.float32)
        embedding_signature = str(cache["signature"].item())
    role_case_ids = {
        str(case_id)
        for block in roles["roles"].values()
        for case_id in block["case_ids"]
    }
    if set(candidate_ids) != role_case_ids:
        raise RuntimeError("Reranker role bank differs from embedding bank.")
    bank = [cases[case_id] for case_id in candidate_ids]
    raw_bm25 = BM25Retriever().fit(
        [
            {
                "case_id": case.study_id,
                "report_text": case.report_text,
                "findings": case.findings,
                "impression": case.impression,
                "images": list(case.image_paths),
            }
            for case in bank
        ]
    )
    bm25_term_cache: dict[str, tuple[np.ndarray, int]] = {}
    prepared_labels = [active_label_weights(case.labels) for case in bank]
    prepared_facts = [case.radgraph_facts for case in bank]
    index_by_id = {case_id: index for index, case_id in enumerate(candidate_ids)}
    question_suite = protocol["question_suite"]

    fit_ids = roles["roles"]["fit"]["case_ids"]
    internal_ids = roles["roles"]["internal_early_stop"]["case_ids"]
    high_rows: list[np.ndarray] = []
    low_rows: list[np.ndarray] = []
    pair_weights: list[float] = []
    skipped_fit_queries = 0
    for query_number, case_id in enumerate(fit_ids, start=1):
        query_index = index_by_id[case_id]
        query = cases[case_id]
        image_image = bank_images @ bank_images[query_index]
        image_report = report_means @ bank_images[query_index]
        image_image[query_index] = -np.inf
        image_report[query_index] = -np.inf
        gains = relevance_array(
            query,
            bank,
            query_index,
            prepared_labels=prepared_labels,
            prepared_facts=prepared_facts,
        )
        query_had_pair = False
        for question_type, question in question_suite.items():
            query_text = query.query_text(question)
            bm25 = exact_leave_one_out_bm25_scores(
                raw_bm25,
                query_text,
                excluded_index=query_index,
                term_cache=bm25_term_cache,
            )
            features = feature_matrix(
                bm25,
                image_image,
                image_report,
                question_type=question_type,
                excluded_index=query_index,
            )
            sampled = sample_pairs(
                features,
                gains,
                (bm25, image_image, image_report),
                config=config["pair_sampling"],
            )
            high_rows.extend(sampled[0])
            low_rows.extend(sampled[1])
            pair_weights.extend(sampled[2])
            query_had_pair = query_had_pair or bool(sampled[2])
        if not query_had_pair:
            skipped_fit_queries += 1
        if query_number % 100 == 0 or query_number == len(fit_ids):
            print(f"fit_feature_cases={query_number}/{len(fit_ids)} pairs={len(pair_weights)}", flush=True)

    high_array = np.asarray(high_rows, dtype=np.float32)
    low_array = np.asarray(low_rows, dtype=np.float32)
    weight_array = np.asarray(pair_weights, dtype=np.float32)
    if not len(weight_array):
        raise RuntimeError("Frozen V9 pair construction produced no training pairs.")

    internal_states: list[dict[str, Any]] = []
    for query_number, case_id in enumerate(internal_ids, start=1):
        query_index = index_by_id[case_id]
        query = cases[case_id]
        image_image = bank_images @ bank_images[query_index]
        image_report = report_means @ bank_images[query_index]
        image_image[query_index] = -np.inf
        image_report[query_index] = -np.inf
        gains = relevance_array(
            query,
            bank,
            query_index,
            prepared_labels=prepared_labels,
            prepared_facts=prepared_facts,
        )
        matrices = []
        for question_type, question in question_suite.items():
            bm25 = exact_leave_one_out_bm25_scores(
                raw_bm25,
                query.query_text(question),
                excluded_index=query_index,
                term_cache=bm25_term_cache,
            )
            matrices.append(
                feature_matrix(
                    bm25,
                    image_image,
                    image_report,
                    question_type=question_type,
                    excluded_index=query_index,
                )
            )
        internal_states.append(
            {"case_id": case_id, "excluded_index": query_index, "gains": gains, "features": matrices}
        )
        if query_number % 100 == 0 or query_number == len(internal_ids):
            print(f"internal_feature_cases={query_number}/{len(internal_ids)}", flush=True)

    trained = [
        train_architecture(
            name,
            high_array,
            low_array,
            weight_array,
            internal_states,
            config["training"],
            args.checkpoint_dir,
            int(config["seed"]),
        )
        for name in ("linear", "mlp")
    ]

    threshold = 0.5
    validation_rows: list[dict[str, Any]] = []
    by_architecture: dict[str, list[dict[str, Any]]] = {item["name"]: [] for item in trained}
    for query_index, case_id in enumerate(validation_ids):
        query = cases[case_id]
        qrels = {
            candidate.study_id: float(gain)
            for candidate, gain in zip(
                bank,
                relevance_array(
                    query,
                    bank,
                    None,
                    prepared_labels=prepared_labels,
                    prepared_facts=prepared_facts,
                ),
                strict=True,
            )
        }
        image_image = bank_images @ validation_images[query_index]
        image_report = report_means @ validation_images[query_index]
        for question_type, question in question_suite.items():
            bm25 = exact_leave_one_out_bm25_scores(
                raw_bm25,
                query.query_text(question),
                excluded_index=None,
                term_cache=bm25_term_cache,
            )
            features = feature_matrix(
                bm25,
                image_image,
                image_report,
                question_type=question_type,
                excluded_index=None,
            )
            for trained_model in trained:
                model = trained_model["model"]
                model.eval()
                with torch.inference_mode():
                    scores = model(torch.from_numpy(features)).numpy()
                ranking_indices = np.lexsort((np.arange(len(scores)), -scores))
                ranking = [candidate_ids[index] for index in ranking_indices]
                metrics = {
                    "ndcg@1": ndcg_at_k(qrels, ranking, 1),
                    "ndcg@5": ndcg_at_k(qrels, ranking, 5),
                    "ndcg@10": ndcg_at_k(qrels, ranking, 10),
                    "recall@1": binary_recall_at_k(qrels, ranking, 1, threshold=threshold),
                    "recall@5": binary_recall_at_k(qrels, ranking, 5, threshold=threshold),
                    "recall@10": binary_recall_at_k(qrels, ranking, 10, threshold=threshold),
                    "mrr": reciprocal_rank_at_threshold(qrels, ranking, threshold=threshold),
                }
                row = {
                    "case_id": case_id,
                    "qid": f"{case_id}:{question_type}",
                    "question_type": question_type,
                    "architecture": trained_model["name"],
                    **metrics,
                }
                validation_rows.append(row)
                by_architecture[trained_model["name"]].append(row)
        if (query_index + 1) % 50 == 0 or query_index + 1 == len(validation_ids):
            print(f"validation_cases={query_index + 1}/{len(validation_ids)}", flush=True)

    metric_names = ("ndcg@1", "ndcg@5", "ndcg@10", "recall@1", "recall@5", "recall@10", "mrr")
    validation_metrics = {
        name: {
            metric: statistics.fmean(float(row[metric]) for row in rows)
            for metric in metric_names
        }
        for name, rows in by_architecture.items()
    }
    difference = validation_metrics["mlp"]["ndcg@10"] - validation_metrics["linear"]["ndcg@10"]
    selected_name = (
        "linear"
        if abs(difference) < float(config["architecture_tie_tolerance"])
        else ("mlp" if difference > 0 else "linear")
    )
    selected_metric = validation_metrics[selected_name]["ndcg@10"]
    fixed_metric = float(medsiglip["selected_fixed_multimodal"]["metrics"]["ndcg@10"])
    component_metric = max(
        float(row["ndcg@10"]) for row in medsiglip["component_metrics"].values()
    )
    promotion = selected_metric - fixed_metric >= float(
        config["promotion_margin_over_fixed_multimodal"]
    )
    strong = selected_metric - component_metric >= float(
        config["strong_diagnostic_margin_over_best_component"]
    )

    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    with args.rows_output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in validation_rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    summary = {
        "study": "V9 learned paired reranker development",
        "status": "development_training_and_validation_complete_test_not_executed",
        "roles_path": portable_path(args.roles),
        "roles_sha256": file_sha256(args.roles),
        "embedding_cache": {
            "path": portable_path(args.embeddings),
            "sha256": file_sha256(args.embeddings),
            "signature": embedding_signature,
            "committed_to_public_repository": False,
        },
        "training_pairs": len(weight_array),
        "fit_queries_without_valid_pairs": skipped_fit_queries,
        "architectures": {
            item["name"]: {
                "parameter_count": sum(parameter.numel() for parameter in item["model"].parameters()),
                "best_epoch": item["best_epoch"],
                "best_internal_ndcg@10": item["best_internal_ndcg@10"],
                "history": item["history"],
                "checkpoint_path": portable_path(item["checkpoint"]),
                "checkpoint_sha256": item["checkpoint_sha256"],
                "checkpoint_committed_to_public_repository": False,
                "validation_metrics": validation_metrics[item["name"]],
            }
            for item in trained
        },
        "architecture_selection": {
            "selected": selected_name,
            "mlp_minus_linear_ndcg@10": difference,
            "tie_tolerance": config["architecture_tie_tolerance"],
        },
        "promotion": {
            "selected_ndcg@10": selected_metric,
            "fixed_multimodal_ndcg@10": fixed_metric,
            "delta_over_fixed_multimodal": selected_metric - fixed_metric,
            "required_margin": config["promotion_margin_over_fixed_multimodal"],
            "promoted_over_fixed_multimodal": promotion,
            "best_single_component_ndcg@10": component_metric,
            "delta_over_best_single_component": selected_metric - component_metric,
            "strong_diagnostic_passed": strong,
        },
        "rows_output": {
            "path": portable_path(args.rows_output),
            "sha256": file_sha256(args.rows_output),
            "committed_to_public_repository": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "foundation_parameters_updated": False,
        "trainable_component": "small_pairwise_ranking_scorer",
        "test_queries_executed": 0,
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
