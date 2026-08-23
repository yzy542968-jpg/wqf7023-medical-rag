from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.evaluation.v10_confirmation import (  # noqa: E402
    case_grouped_bootstrap_difference,
    deterministic_derangement,
    graded_ndcg,
    hit_at_k,
    plus_one_monte_carlo_p,
    reciprocal_rank_at_threshold,
)
from medical_rag.similar_case.v10_calibration import (  # noqa: E402
    calibration_metrics,
    predict_from_payload,
    risk_coverage_curve,
)
from medical_rag.similar_case.v10_evidence import evidence_diagnostics, select_case_evidence  # noqa: E402
from medical_rag.similar_case.v10_loader import load_v10_runtime_assets, read_json  # noqa: E402
from medical_rag.similar_case.v10_runtime import QUESTIONS, component_agreement  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from train_v9_learned_reranker import relevance_array  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v10_confirmation.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v10" / "v10_cluster_disjoint_split.json"
DEFAULT_EMBEDDINGS = ROOT / "data" / "processed" / "v10_medsiglip_embeddings.npz"
DEFAULT_R5 = ROOT / "experiments" / "v10_publication" / "reranker_checkpoints"
DEFAULT_ATTENTION = ROOT / "experiments" / "v10_publication" / "multiview_checkpoints"
DEFAULT_CALIBRATOR = ROOT / "artifacts" / "v10" / "retrieval_calibrator.json"
DEFAULT_ROWS = ROOT / "experiments" / "v10_publication" / "v10_confirmation_retrieval_rows.jsonl"
DEFAULT_SHUFFLED = ROOT / "experiments" / "v10_publication" / "v10_confirmation_shuffled_summary.json"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_confirmation_retrieval_summary.json"


def verify_frozen_inputs(config: Mapping[str, Any], paths: Mapping[str, Path]) -> None:
    if config.get("status") != "confirmation_frozen_before_test_execution":
        raise RuntimeError("V10 confirmation config is not frozen")
    if not config.get("test_execution_authorized", False):
        raise RuntimeError("V10 confirmation config does not authorize Test execution")
    if config.get("test_outcomes_inspected", True):
        raise RuntimeError("V10 confirmation config records inspected Test outcomes")
    for name, path in paths.items():
        expected = str(config["frozen_input_sha256"][name])
        observed = file_sha256(path)
        if observed != expected:
            raise RuntimeError(f"frozen input changed for {name}: {observed} != {expected}")


def retrieval_metrics(gains: np.ndarray, ranking: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "ndcg@10": graded_ndcg(gains, ranking, k=10),
        "mrr_relevance_0.5": reciprocal_rank_at_threshold(
            gains, ranking, threshold=threshold
        ),
        "hit@1_relevance_0.5": hit_at_k(gains, ranking, k=1, threshold=threshold),
        "hit@5_relevance_0.5": hit_at_k(gains, ranking, k=5, threshold=threshold),
        "hit@10_relevance_0.5": hit_at_k(gains, ranking, k=10, threshold=threshold),
    }


def summarize_systems(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["system"])].append(row)
    return {
        system: {
            "row_count": len(values),
            "case_count": len({row["case_id"] for row in values}),
            **{
                metric: statistics.fmean(float(row[metric]) for row in values)
                for metric in (
                    "ndcg@10",
                    "mrr_relevance_0.5",
                    "hit@1_relevance_0.5",
                    "hit@5_relevance_0.5",
                    "hit@10_relevance_0.5",
                )
            },
        }
        for system, values in grouped.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen V10 retrieval confirmation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--r5-checkpoints", type=Path, default=DEFAULT_R5)
    parser.add_argument("--attention-checkpoints", type=Path, default=DEFAULT_ATTENTION)
    parser.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--shuffled-output", type=Path, default=DEFAULT_SHUFFLED)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    config = read_json(args.config)
    verify_frozen_inputs(
        config,
        {
            "cases": args.cases,
            "radgraph": args.radgraph,
            "split": args.split,
            "embeddings": args.embeddings,
            "calibrator": args.calibrator,
        },
    )
    calibrator = read_json(args.calibrator)
    assets = load_v10_runtime_assets(
        cases_path=args.cases,
        radgraph_path=args.radgraph,
        split_path=args.split,
        embeddings_path=args.embeddings,
        r5_checkpoint_dir=args.r5_checkpoints,
        attention_checkpoint_dir=args.attention_checkpoints,
    )
    runtime = assets.runtime
    test_ids = assets.partition_ids("test")
    prepared_labels = [dict(case.labels) for case in runtime.candidate_cases]
    prepared_facts = [case.radgraph_facts for case in runtime.candidate_cases]
    threshold = float(config["positive_gain_threshold"])
    evidence_policy = str(config["evidence_policy"])
    operating_threshold = float(
        calibrator["coverage_thresholds"][str(config["selective_coverage_operating_point"])]
    )
    rows: list[dict[str, Any]] = []
    prepared_queries: dict[tuple[str, str], dict[str, Any]] = {}
    gains_by_case: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for position, case_id in enumerate(test_ids, start=1):
        query = assets.cases[case_id]
        gains = relevance_array(
            query,
            runtime.candidate_cases,
            None,
            prepared_labels=prepared_labels,
            prepared_facts=prepared_facts,
        )
        gains_by_case[case_id] = gains
        query_image = assets.attention_image(case_id)
        for question_type in QUESTIONS:
            prepared = runtime.prepare_query(query, question_type=question_type)
            prepared_queries[(case_id, question_type)] = prepared
            result = runtime.score_prepared(prepared, query_image)
            systems = {
                "r0_bm25": result["bm25"],
                "r1_image_image": result["image_image"],
                "r2_image_report": result["image_report"],
                "r4_nine_feature": result["r4_scores"],
                "r5_fact_attention": result["ensemble_scores"],
            }
            for system, scores in systems.items():
                ranking = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
                rows.append(
                    {
                        "case_id": case_id,
                        "question_type": question_type,
                        "system": system,
                        "top_case_ids": [runtime.candidate_ids[int(index)] for index in ranking[:10]],
                        "top_scores": [float(scores[int(index)]) for index in ranking[:10]],
                        **retrieval_metrics(gains, ranking, threshold),
                    }
                )
            ranking = result["ranking"]
            top1, top2 = int(ranking[0]), int(ranking[1])
            evidence = []
            for index in ranking[:3]:
                retrieved_id = runtime.candidate_ids[int(index)]
                evidence.extend(
                    select_case_evidence(
                        assets.raw_cases[retrieved_id],
                        query=result["query_text"],
                        facts=assets.radgraph[retrieved_id].facts,
                        policy=evidence_policy,
                    )
                )
            diagnostics = evidence_diagnostics(evidence)
            feature_row = {
                "top1_score": float(result["ensemble_scores"][top1]),
                "top1_top2_margin": float(
                    result["ensemble_scores"][top1] - result["ensemble_scores"][top2]
                ),
                "component_agreement": component_agreement(result, top1),
                "ensemble_variance": float(np.var(result["seed_scores"][:, top1])),
                "evidence_score": float(diagnostics["mean_score"]),
                "evidence_redundancy": float(diagnostics["redundancy"]),
                "view_count": float(len(assets.views_by_id[case_id])),
                "question_findings": float(question_type == "findings"),
                "question_impression": float(question_type == "impression"),
                "question_acute": float(question_type == "acute"),
            }
            probability = float(predict_from_payload([feature_row], calibrator)[0])
            rows[-1]["retrieval_confidence"] = probability
            rows[-1]["no_reliable_history"] = probability < operating_threshold
            rows[-1]["top1_gain"] = float(gains[top1])
            rows[-1]["calibration_label"] = int(float(gains[top1]) >= threshold)
        if position % 25 == 0 or position == len(test_ids):
            print(f"test_aligned={position}/{len(test_ids)}", flush=True)

    r5_rows = [row for row in rows if row["system"] == "r5_fact_attention"]
    labels = [int(row["calibration_label"]) for row in r5_rows]
    probabilities = [float(row["retrieval_confidence"]) for row in r5_rows]
    aligned_mean = statistics.fmean(float(row["ndcg@10"]) for row in r5_rows)

    shuffled_means = []
    shuffled_records = []
    assignments = int(config["shuffled_image_assignments"])
    for assignment in range(assignments):
        mapping = deterministic_derangement(
            test_ids,
            assignment_index=assignment,
            seed=int(config["shuffle_seed"]),
        )
        values = []
        for case_id in test_ids:
            wrong_image = assets.attention_image(mapping[case_id])
            gains = gains_by_case[case_id]
            for question_type in QUESTIONS:
                result = runtime.score_prepared(prepared_queries[(case_id, question_type)], wrong_image)
                values.append(graded_ndcg(gains, result["ranking"], k=10))
        mean_value = statistics.fmean(values)
        shuffled_means.append(mean_value)
        shuffled_records.append(
            {
                "assignment": assignment,
                "mean_ndcg@10": mean_value,
                "mapping_sha256": __import__("hashlib").sha256(
                    "\n".join(f"{key}\t{mapping[key]}" for key in sorted(mapping)).encode("utf-8")
                ).hexdigest(),
            }
        )
        print(f"shuffled_assignment={assignment + 1}/{assignments}", flush=True)

    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    args.shuffled_output.write_text(json.dumps(shuffled_records, indent=2) + "\n", encoding="utf-8")
    systems = summarize_systems(rows)
    summary = {
        "study": "V10 cluster-disjoint retrieval confirmation",
        "status": "confirmation_complete_no_retuning",
        "counts": {
            "test_cases": len(test_ids),
            "question_rows_per_system": len(test_ids) * len(QUESTIONS),
            "systems": len(systems),
            "shuffled_assignments": assignments,
        },
        "metrics": systems,
        "primary_r5_minus_r4": case_grouped_bootstrap_difference(
            rows,
            left="r5_fact_attention",
            right="r4_nine_feature",
            metric="ndcg@10",
            iterations=int(config["bootstrap_iterations"]),
            seed=int(config["bootstrap_seed"]),
        ),
        "alignment_control": {
            "aligned_mean_ndcg@10": aligned_mean,
            "shuffled_mean_ndcg@10": statistics.fmean(shuffled_means),
            "shuffled_min_ndcg@10": min(shuffled_means),
            "shuffled_max_ndcg@10": max(shuffled_means),
            "plus_one_monte_carlo_p": plus_one_monte_carlo_p(aligned_mean, shuffled_means),
        },
        "calibration": {
            "metrics": calibration_metrics(labels, probabilities),
            "operating_coverage": config["selective_coverage_operating_point"],
            "operating_threshold": operating_threshold,
            "observed_answer_coverage": statistics.fmean(
                float(not row["no_reliable_history"]) for row in r5_rows
            ),
            "risk_coverage_curve": risk_coverage_curve(labels, probabilities),
        },
        "runtime_seconds": time.perf_counter() - started,
        "retrieval_rows_sha256": file_sha256(args.rows_output),
        "shuffled_summary_sha256": file_sha256(args.shuffled_output),
        "retuning_after_test": False,
        "claim_boundary": "Within-source cluster-disjoint retrieval relevance, not clinical safety.",
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("counts", "metrics", "primary_r5_minus_r4", "alignment_control")}, indent=2))


if __name__ == "__main__":
    main()
