"""Audit the V10 facts x image-view 2x2 on validation only.

This is a development attribution audit over already-frozen checkpoints. It
does not retrain R4/R5, inspect the V10 test partition, or alter V10 results.
The four cells are:

    R4 (no fact features) x mean image view
    R4 (no fact features) x learned attention image view
    R5 (fact-aware)      x mean image view
    R5 (fact-aware)      x learned attention image view
"""

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
from medical_rag.similar_case.v10_multiview import ViewAttention, attention_query_embedding, l2_normalize  # noqa: E402
from medical_rag.similar_case.v10_runtime import FrozenR5Runtime, QUESTIONS  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from train_v9_learned_reranker import numeric_ndcg10, relevance_array  # noqa: E402


DEFAULT_CASES = ROOT / "data/processed/openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data/processed/v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data/splits/v10/v10_cluster_disjoint_split.json"
DEFAULT_EMBEDDINGS = ROOT / "data/processed/v10_medsiglip_embeddings.npz"
DEFAULT_R5 = ROOT / "experiments/v10_publication/reranker_checkpoints"
DEFAULT_R4 = DEFAULT_R5 / "r4.pt"
DEFAULT_ATTENTION = ROOT / "experiments/v10_publication/multiview_checkpoints"
DEFAULT_ROWS = ROOT / "experiments/v10_publication/v10_fact_attention_2x2_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "data/splits/v10/v10_fact_attention_2x2_summary.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--r5-checkpoints", type=Path, default=DEFAULT_R5)
    parser.add_argument("--r4-checkpoint", type=Path, default=DEFAULT_R4)
    parser.add_argument("--attention-checkpoints", type=Path, default=DEFAULT_ATTENTION)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    split = read_json(args.split)
    raw_rows = read_jsonl(args.cases)
    raw_cases = {str(row["case_id"]): row for row in raw_rows}
    cases = read_openi_paired_cases(args.cases, source_unique_patient=True, radgraph_path=args.radgraph)
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
        case_id for case_id in case_by_id
        if case_id in image_by_id and case_id in report_by_id and case_id in views_by_id and radgraph[case_id].status == "ok"
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
        model.load_state_dict(torch.load(args.attention_checkpoints / f"attention_seed_{seed}.pt", map_location="cpu", weights_only=True))
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
        r4_checkpoint_state=torch.load(args.r4_checkpoint, map_location="cpu", weights_only=True),
    )
    prepared_labels = [dict(case.labels) for case in runtime.candidate_cases]
    prepared_facts = [case.radgraph_facts for case in runtime.candidate_cases]
    rows: list[dict[str, Any]] = []
    for position, case_id in enumerate(validation_ids, start=1):
        query = case_by_id[case_id]
        gains = relevance_array(query, runtime.candidate_cases, None, prepared_labels=prepared_labels, prepared_facts=prepared_facts)
        mean_image = image_by_id[case_id]
        attention_image = attention_query_embedding(attention_models, np.stack(views_by_id[case_id]))
        mean_result = runtime.prepare_query(query, question_type="findings")
        for question_type in QUESTIONS:
            prepared = runtime.prepare_query(query, question_type=question_type)
            results = {
                "r4_mean": runtime.score_prepared(prepared, mean_image)["r4_scores"],
                "r4_attention": runtime.score_prepared(prepared, attention_image)["r4_scores"],
                "r5_mean": runtime.score_prepared(prepared, mean_image)["ensemble_scores"],
                "r5_attention": runtime.score_prepared(prepared, attention_image)["ensemble_scores"],
            }
            metrics: dict[str, float] = {}
            top3: dict[str, list[str]] = {}
            for name, scores in results.items():
                ranking = np.lexsort((np.arange(len(scores)), -np.asarray(scores)))
                metrics[name] = numeric_ndcg10(gains, ranking)
                top3[name] = [candidate_ids[index] for index in ranking[:3]]
            rows.append({"case_id": case_id, "question_type": question_type, "ndcg@10": metrics, "top3": top3})
        if position % 25 == 0 or position == len(validation_ids):
            print(f"fact_attention_validation={position}/{len(validation_ids)}", flush=True)
    names = ("r4_mean", "r4_attention", "r5_mean", "r5_attention")
    metrics = {name: statistics.fmean(row["ndcg@10"][name] for row in rows) for name in names}
    summary = {
        "study": "V10 facts x image-view 2x2 attribution audit",
        "status": "development_only_test_not_run",
        "conditions": {
            "r4": "frozen non-fact-aware reranker",
            "r5": "frozen fact-aware reranker",
            "mean": "mean image embedding",
            "attention": "frozen learned multi-view attention embedding",
        },
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "radgraph_sha256": file_sha256(args.radgraph),
            "split_sha256": file_sha256(args.split),
            "r4_checkpoint_sha256": file_sha256(args.r4_checkpoint),
            "embedding_signature": embedding_signature,
        },
        "counts": {"candidate_bank": len(candidate_ids), "validation_cases": len(validation_ids), "validation_rows": len(rows)},
        "validation_ndcg@10": metrics,
        "main_effect_fact_aware_reranker": ((metrics["r5_mean"] + metrics["r5_attention"]) / 2) - ((metrics["r4_mean"] + metrics["r4_attention"]) / 2),
        "main_effect_attention_view": ((metrics["r4_attention"] + metrics["r5_attention"]) / 2) - ((metrics["r4_mean"] + metrics["r5_mean"]) / 2),
        "interaction_fact_x_attention": (metrics["r5_attention"] - metrics["r5_mean"]) - (metrics["r4_attention"] - metrics["r4_mean"]),
        "rows_sha256": file_sha256(args.rows_output) if args.rows_output.is_file() else None,
        "claim_boundary": "Validation attribution audit over frozen checkpoints; no causal claim beyond this fixed 2x2 comparison and no test outcome inspected.",
    }
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    summary["rows_sha256"] = file_sha256(args.rows_output)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
