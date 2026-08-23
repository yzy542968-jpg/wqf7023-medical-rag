from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


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
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.relevance import (  # noqa: E402
    active_label_similarity,
    active_label_weights,
    radgraph_fact_similarity,
)
from train_v9_learned_reranker import (  # noqa: E402
    MLPScorer,
    exact_leave_one_out_bm25_scores,
    feature_matrix,
)


DEFAULT_CONFIG = ROOT / "config" / "v9_supplemental_validity.json"
DEFAULT_CONFIRMATION_CONFIG = ROOT / "config" / "v9_retrieval_confirmation.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v9" / "v9_full_source_split.json"
DEFAULT_PROTOCOL = ROOT / "config" / "v9_similar_case_rag_development.json"
DEFAULT_DEV_EMBEDDINGS = ROOT / "data" / "processed" / "v9_medsiglip_development_embeddings.npz"
DEFAULT_TEST_EMBEDDINGS = ROOT / "data" / "processed" / "v9_medsiglip_test_embeddings.npz"
DEFAULT_CHECKPOINT = (
    ROOT
    / "experiments"
    / "post_submission_v9"
    / "reranker_checkpoints"
    / "v9_mlp_best.pt"
)
DEFAULT_OUTPUT = ROOT / "data" / "splits" / "v9" / "v9_qrel_sensitivity_summary.json"


def qrel_components(query: Any, bank: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(
        [active_label_similarity(query.labels, candidate.labels) for candidate in bank],
        dtype=np.float32,
    )
    facts = np.asarray(
        [
            radgraph_fact_similarity(query.radgraph_facts, candidate.radgraph_facts)
            for candidate in bank
        ],
        dtype=np.float32,
    )
    return labels, facts


def qrel_array(
    labels: np.ndarray, facts: np.ndarray, *, label_weight: float, fact_weight: float
) -> np.ndarray:
    if not np.isclose(label_weight + fact_weight, 1.0):
        raise ValueError("Qrel weights must sum to one.")
    return (label_weight * labels + fact_weight * facts).astype(np.float32)


def evaluate_ranking(
    candidate_ids: Sequence[str], gains: np.ndarray, ranking_indices: np.ndarray
) -> dict[str, float]:
    qrels = dict(zip(candidate_ids, map(float, gains), strict=True))
    ranking = [candidate_ids[int(index)] for index in ranking_indices]
    return {
        "ndcg@10": ndcg_at_k(qrels, ranking, 10),
        "mrr_at_gain_0.5": reciprocal_rank_at_threshold(qrels, ranking, threshold=0.5),
        "recall@10_at_gain_0.5": binary_recall_at_k(
            qrels, ranking, 10, threshold=0.5
        ),
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    metrics = ("ndcg@10", "mrr_at_gain_0.5", "recall@10_at_gain_0.5")
    return {
        metric: statistics.fmean(float(row[metric]) for row in rows)
        for metric in metrics
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen V9 rankings under prespecified qrel variants."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--confirmation-config", type=Path, default=DEFAULT_CONFIRMATION_CONFIG
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--development-embeddings", type=Path, default=DEFAULT_DEV_EMBEDDINGS
    )
    parser.add_argument("--test-embeddings", type=Path, default=DEFAULT_TEST_EMBEDDINGS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    confirmation = read_json(args.confirmation_config)
    split = read_json(args.split)
    protocol = read_json(args.protocol)
    expected_checkpoint = confirmation["systems"]["r4"]["checkpoint_sha256"]
    if file_sha256(args.checkpoint) != expected_checkpoint:
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
    if set(candidate_ids) & set(test_ids):
        raise RuntimeError("Test study entered the historical candidate bank.")

    bank = [cases[case_id] for case_id in candidate_ids]
    bm25 = BM25Retriever().fit(
        [{"case_id": case.study_id, "report_text": case.report_text} for case in bank]
    )
    term_cache: dict[str, tuple[np.ndarray, int]] = {}
    model = MLPScorer()
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()
    fixed = confirmation["systems"]["r3"]["weights"]
    variants = config["qrel_sensitivity"]["variants"]
    by_variant_system: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)

    for query_index, case_id in enumerate(test_ids):
        query = cases[case_id]
        label_gains, fact_gains = qrel_components(query, bank)
        image_image = bank_images @ test_images[query_index]
        image_report = report_means @ test_images[query_index]
        for question_type, question in protocol["question_suite"].items():
            text = exact_leave_one_out_bm25_scores(
                bm25,
                query.query_text(question),
                excluded_index=None,
                term_cache=term_cache,
            )
            features = feature_matrix(
                text,
                image_image,
                image_report,
                question_type=question_type,
                excluded_index=None,
            )
            with torch.inference_mode():
                learned = model(torch.from_numpy(features)).numpy()
            channels = {
                "r0_bm25": text,
                "r1_image_image": image_image,
                "r2_image_report": image_report,
                "r3_fixed_multimodal": (
                    fixed["bm25"] * features[:, 0]
                    + fixed["image_image"] * features[:, 1]
                    + fixed["image_report"] * features[:, 2]
                ),
                "r4_learned_mlp": learned,
            }
            rankings = {
                system: np.lexsort((np.arange(len(scores)), -scores))
                for system, scores in channels.items()
            }
            for variant, weights in variants.items():
                gains = qrel_array(
                    label_gains,
                    fact_gains,
                    label_weight=float(weights[0]),
                    fact_weight=float(weights[1]),
                )
                for system, ranking in rankings.items():
                    by_variant_system[(variant, system)].append(
                        evaluate_ranking(candidate_ids, gains, ranking)
                    )
        if (query_index + 1) % 100 == 0 or query_index + 1 == len(test_ids):
            print(f"qrel_cases={query_index + 1}/{len(test_ids)}", flush=True)

    metrics: dict[str, Any] = {}
    for variant in variants:
        system_metrics = {
            system: aggregate(by_variant_system[(variant, system)])
            for system in config["qrel_sensitivity"]["systems"]
        }
        ordering = sorted(
            system_metrics,
            key=lambda system: (-system_metrics[system]["ndcg@10"], system),
        )
        metrics[variant] = {
            "systems": system_metrics,
            "ndcg@10_ordering": ordering,
            "r4_minus_r1_ndcg@10": (
                system_metrics["r4_learned_mlp"]["ndcg@10"]
                - system_metrics["r1_image_image"]["ndcg@10"]
            ),
        }

    output = {
        "study": "V9 qrel construct sensitivity",
        "status": "post_hoc_exploratory_complete",
        "test_case_count": len(test_ids),
        "question_count": len(test_ids) * len(protocol["question_suite"]),
        "candidate_bank_count": len(candidate_ids),
        "config_sha256": file_sha256(args.config),
        "confirmation_config_sha256": file_sha256(args.confirmation_config),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "metrics": metrics,
        "claim_boundary": (
            "This audit tests sensitivity to the relevance construct. It does not "
            "replace the frozen combined-qrel confirmation result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
