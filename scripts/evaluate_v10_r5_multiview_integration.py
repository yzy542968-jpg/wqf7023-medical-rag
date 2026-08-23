from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v10_multiview import (  # noqa: E402
    ViewAttention,
    attention_query_embedding,
    l2_normalize,
)
from medical_rag.similar_case.v10_runtime import FrozenR5Runtime, QUESTIONS  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from train_v9_learned_reranker import numeric_ndcg10, relevance_array  # noqa: E402


DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v10" / "v10_cluster_disjoint_split.json"
DEFAULT_EMBEDDINGS = ROOT / "data" / "processed" / "v10_medsiglip_embeddings.npz"
DEFAULT_R5 = ROOT / "experiments" / "v10_publication" / "reranker_checkpoints"
DEFAULT_ATTENTION = ROOT / "experiments" / "v10_publication" / "multiview_checkpoints"
DEFAULT_R5_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_reranker_development_summary.json"
DEFAULT_CONFIG = ROOT / "config" / "v10_r5_multiview_integration.json"
DEFAULT_ROWS = ROOT / "experiments" / "v10_publication" / "v10_r5_multiview_integration_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_r5_multiview_integration_summary.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen R5 with the selected view policy.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--r5-checkpoints", type=Path, default=DEFAULT_R5)
    parser.add_argument("--attention-checkpoints", type=Path, default=DEFAULT_ATTENTION)
    parser.add_argument("--r5-summary", type=Path, default=DEFAULT_R5_SUMMARY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    config = read_json(args.config)
    if config["calibration_outcomes_inspected"] or config["test_outcomes_inspected"]:
        raise RuntimeError("integration config records forbidden outcomes")
    split = read_json(args.split)
    raw_rows = read_jsonl(args.cases)
    raw_cases = {str(row["case_id"]): row for row in raw_rows}
    cases = read_openi_paired_cases(
        args.cases,
        source_unique_patient=True,
        radgraph_path=args.radgraph,
    )
    case_by_id = {case.study_id: case for case in cases}
    radgraph = read_radgraph_case_records(args.radgraph)
    with np.load(args.embeddings, allow_pickle=False) as encoded:
        case_ids = [str(value) for value in encoded["case_ids"]]
        case_images = l2_normalize(np.asarray(encoded["case_image_embeddings"], dtype=np.float32))
        report_ids = [str(value) for value in encoded["report_ids"]]
        reports = l2_normalize(np.asarray(encoded["report_embeddings"], dtype=np.float32))
        view_ids = [str(value) for value in encoded["view_case_ids"]]
        views = l2_normalize(np.asarray(encoded["view_embeddings"], dtype=np.float32))
        embedding_signature = str(encoded["signature"].item())
    image_by_id = {case_id: case_images[index] for index, case_id in enumerate(case_ids)}
    report_by_id = {case_id: reports[index] for index, case_id in enumerate(report_ids)}
    views_by_id: dict[str, list[np.ndarray]] = {}
    for case_id, embedding in zip(view_ids, views, strict=True):
        views_by_id.setdefault(case_id, []).append(embedding)

    eligible = {
        case_id
        for case_id in case_by_id
        if case_id in image_by_id
        and case_id in report_by_id
        and case_id in views_by_id
        and radgraph[case_id].status == "ok"
    }
    candidate_ids = sorted(set(split["partitions"]["train"]["case_ids"]) & eligible)
    validation_ids = sorted(set(split["partitions"]["validation"]["case_ids"]) & eligible)
    r5_states = [
        torch.load(args.r5_checkpoints / f"r5_seed_{seed}.pt", map_location="cpu", weights_only=True)
        for seed in (7041, 7042, 7043, 7044, 7045)
    ]
    attention_models = []
    for seed in (7051, 7052, 7053, 7054, 7055):
        model = ViewAttention()
        model.load_state_dict(
            torch.load(
                args.attention_checkpoints / f"attention_seed_{seed}.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        model.eval()
        attention_models.append(model)
    runtime = FrozenR5Runtime.build(
        candidate_ids=candidate_ids,
        cases=case_by_id,
        raw_cases=raw_cases,
        facts_by_case={case_id: tuple(radgraph[case_id].facts) for case_id in candidate_ids},
        image_by_id=image_by_id,
        report_by_id=report_by_id,
        checkpoint_states=r5_states,
    )
    prepared_labels = [dict(case.labels) for case in runtime.candidate_cases]
    prepared_facts = [case.radgraph_facts for case in runtime.candidate_cases]

    rows = []
    for position, case_id in enumerate(validation_ids, start=1):
        query = case_by_id[case_id]
        gains = relevance_array(
            query,
            runtime.candidate_cases,
            None,
            prepared_labels=prepared_labels,
            prepared_facts=prepared_facts,
        )
        mean_image = image_by_id[case_id]
        attention_image = attention_query_embedding(
            attention_models,
            np.stack(views_by_id[case_id]),
        )
        for question_type in QUESTIONS:
            conditions = {
                "mean": runtime.score(query, mean_image, question_type=question_type),
                "learned_attention": runtime.score(
                    query, attention_image, question_type=question_type
                ),
            }
            metrics = {}
            top3 = {}
            for name, result in conditions.items():
                ranking = result["ranking"]
                metrics[name] = numeric_ndcg10(gains, ranking)
                top3[name] = [candidate_ids[index] for index in ranking[:3]]
            rows.append(
                {
                    "case_id": case_id,
                    "question_type": question_type,
                    "ndcg@10": metrics,
                    "top3": top3,
                }
            )
        if position % 25 == 0 or position == len(validation_ids):
            print(f"integration_validation={position}/{len(validation_ids)}", flush=True)

    metrics = {
        name: statistics.fmean(row["ndcg@10"][name] for row in rows)
        for name in ("mean", "learned_attention")
    }
    frozen_r5 = read_json(args.r5_summary)
    expected = float(frozen_r5["selected_r5_ndcg@10"])
    if abs(metrics["mean"] - expected) > 1e-8:
        raise RuntimeError(
            f"runtime failed to reproduce frozen R5 mean: {metrics['mean']} != {expected}"
        )
    tolerance = float(config["validation_degradation_tolerance_ndcg10"])
    delta = metrics["learned_attention"] - metrics["mean"]
    accepted = delta > -tolerance
    selected = "learned_attention" if accepted else "mean"

    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary = {
        "study": "V10 R5 and multi-view integration",
        "status": "development_complete_test_not_run",
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "radgraph_sha256": file_sha256(args.radgraph),
            "split_sha256": file_sha256(args.split),
            "config_sha256": file_sha256(args.config),
            "embedding_signature": embedding_signature,
        },
        "counts": {
            "candidate_bank": len(candidate_ids),
            "validation_cases": len(validation_ids),
            "validation_rows": len(rows),
        },
        "validation_ndcg@10": metrics,
        "learned_attention_minus_mean": delta,
        "degradation_tolerance_ndcg10": tolerance,
        "attention_accepted": accepted,
        "selected_view_policy_for_r5": selected,
        "validation_rows_sha256": file_sha256(args.rows_output),
        "test_outcomes_inspected": False,
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
