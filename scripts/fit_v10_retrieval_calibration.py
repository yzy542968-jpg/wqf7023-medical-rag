from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.similar_case.v10_calibration import (  # noqa: E402
    RetrievalCalibrator,
    calibration_metrics,
    calibrator_payload,
    risk_coverage_curve,
    threshold_for_coverage,
)
from medical_rag.similar_case.v10_evidence import (  # noqa: E402
    evidence_diagnostics,
    select_case_evidence,
)
from medical_rag.similar_case.v10_loader import load_v10_runtime_assets, read_json  # noqa: E402
from medical_rag.similar_case.v10_runtime import QUESTIONS, component_agreement  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from train_v9_learned_reranker import relevance_array  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v10_calibration.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v10" / "v10_cluster_disjoint_split.json"
DEFAULT_EMBEDDINGS = ROOT / "data" / "processed" / "v10_medsiglip_embeddings.npz"
DEFAULT_R5 = ROOT / "experiments" / "v10_publication" / "reranker_checkpoints"
DEFAULT_ATTENTION = ROOT / "experiments" / "v10_publication" / "multiview_checkpoints"
DEFAULT_ROWS = ROOT / "experiments" / "v10_publication" / "v10_calibration_rows.jsonl"
DEFAULT_CHECKPOINT = ROOT / "artifacts" / "v10" / "retrieval_calibrator.json"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_calibration_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit frozen V10 retrieval confidence calibration.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--r5-checkpoints", type=Path, default=DEFAULT_R5)
    parser.add_argument("--attention-checkpoints", type=Path, default=DEFAULT_ATTENTION)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--checkpoint-output", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    config = read_json(args.config)
    if config["calibration_outcomes_inspected"] or config["test_outcomes_inspected"]:
        raise RuntimeError("calibration config records inspected outcomes")
    assets = load_v10_runtime_assets(
        cases_path=args.cases,
        radgraph_path=args.radgraph,
        split_path=args.split,
        embeddings_path=args.embeddings,
        r5_checkpoint_dir=args.r5_checkpoints,
        attention_checkpoint_dir=args.attention_checkpoints,
    )
    runtime = assets.runtime
    prepared_labels = [dict(case.labels) for case in runtime.candidate_cases]
    prepared_facts = [case.radgraph_facts for case in runtime.candidate_cases]
    evidence_policy = str(config["evidence_policy"])
    rows = []
    labels = []
    for position, case_id in enumerate(assets.partition_ids("calibration"), start=1):
        query = assets.cases[case_id]
        gains = relevance_array(
            query,
            runtime.candidate_cases,
            None,
            prepared_labels=prepared_labels,
            prepared_facts=prepared_facts,
        )
        query_image = assets.attention_image(case_id)
        for question_type, question in QUESTIONS.items():
            result = runtime.score(query, query_image, question_type=question_type)
            ranking = result["ranking"]
            top1 = int(ranking[0])
            top2 = int(ranking[1])
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
            features = {
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
            label = int(float(gains[top1]) >= float(config["positive_gain_threshold"]))
            labels.append(label)
            rows.append(
                {
                    "case_id": case_id,
                    "question_type": question_type,
                    "question": question,
                    "top_case_ids": [runtime.candidate_ids[int(index)] for index in ranking[:3]],
                    "top_scores": [float(result["ensemble_scores"][int(index)]) for index in ranking[:3]],
                    "top1_gain": float(gains[top1]),
                    "label": label,
                    "features": features,
                }
            )
        if position % 25 == 0:
            print(f"calibration_retrieval={position}", flush=True)

    feature_rows = [row["features"] for row in rows]
    calibrator = RetrievalCalibrator(seed=int(config["seed"])).fit(feature_rows, labels)
    probabilities = calibrator.predict_proba(feature_rows)
    for row, probability in zip(rows, probabilities, strict=True):
        row["retrieval_confidence"] = float(probability)
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    payload: dict[str, Any] = calibrator_payload(calibrator)
    payload.update(
        {
            "study": "V10 retrieval confidence calibration",
            "positive_gain_threshold": float(config["positive_gain_threshold"]),
            "evidence_policy": evidence_policy,
            "coverage_thresholds": {
                str(coverage): threshold_for_coverage(probabilities, float(coverage))
                for coverage in config["coverage_operating_points"]
            },
            "calibration_rows_sha256": file_sha256(args.rows_output),
        }
    )
    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = {
        "study": "V10 retrieval confidence calibration",
        "status": "calibration_complete_test_not_run",
        "inputs": {
            "config_sha256": file_sha256(args.config),
            "cases_sha256": file_sha256(args.cases),
            "radgraph_sha256": file_sha256(args.radgraph),
            "split_sha256": file_sha256(args.split),
            "embedding_signature": assets.embedding_signature,
        },
        "counts": {
            "calibration_cases": len({row["case_id"] for row in rows}),
            "calibration_rows": len(rows),
            "positive_rows": sum(labels),
            "negative_rows": len(labels) - sum(labels),
        },
        "apparent_fit_metrics": calibration_metrics(labels, probabilities),
        "coverage_thresholds": payload["coverage_thresholds"],
        "risk_coverage_curve": risk_coverage_curve(labels, probabilities),
        "calibrator_sha256": file_sha256(args.checkpoint_output),
        "calibration_rows_sha256": file_sha256(args.rows_output),
        "test_outcomes_inspected": False,
        "claim_boundary": "Calibration of offline retrieval relevance, not clinical safety.",
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("counts", "apparent_fit_metrics", "coverage_thresholds")}, indent=2))


if __name__ == "__main__":
    main()
