"""Evaluate the saved V12 LambdaMART pilot on Validation only."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.retrieval.medcpt_retriever import encode_queries  # noqa: E402
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v10_runtime import FrozenR5Runtime, QUESTIONS  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from medical_rag.similar_case.v11_qrel import prepare_qrel_case, qrel_v2_profile_prepared  # noqa: E402
from medical_rag.similar_case.relevance import active_label_similarity, radgraph_fact_similarity  # noqa: E402
from run_v12_retrieval_pilot import build_retrieval_state, ndcg, spectrum  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def bootstrap(rows: list[dict[str, Any]], new_key: str, baseline_key: str, qrel_key: str) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    differences = np.asarray(
        [
            mean(
                [
                    float(item["metrics"][new_key][qrel_key])
                    - float(item["metrics"][baseline_key][qrel_key])
                    for item in grouped[case_id]
                ]
            )
            for case_id in sorted(grouped)
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(2026)
    sample = np.asarray(
        [rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(10000)],
        dtype=np.float64,
    )
    return {
        "difference": float(differences.mean()),
        "ci95_low": float(np.quantile(sample, 0.025)),
        "ci95_high": float(np.quantile(sample, 0.975)),
        "case_count": len(differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--radgraph", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/processed/v10_medsiglip_embeddings.npz")
    parser.add_argument("--medcpt", type=Path, default=ROOT / "data/processed/openi_medcpt_full.npz")
    parser.add_argument("--model", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_lambdamart.txt")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "experiments/v10_publication/reranker_checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_validation_rankings.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()

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
    facts_by_case = {case_id: tuple(radgraph[case_id].facts) for case_id in train_ids + validation_ids}
    prepared_by_case = {case_id: prepare_qrel_case(raw_cases[case_id], facts_by_case) for case_id in train_ids + validation_ids}

    with np.load(args.embeddings, allow_pickle=False) as encoded:
        image_ids = [str(case_id) for case_id in encoded["case_ids"].tolist()]
        report_ids = [str(case_id) for case_id in encoded["report_ids"].tolist()]
        image_matrix = np.asarray(encoded["case_image_embeddings"], dtype=np.float32)
        report_matrix = np.asarray(encoded["report_embeddings"], dtype=np.float32)
    image_by_id = {case_id: image_matrix[index] for index, case_id in enumerate(image_ids)}
    report_by_id = {case_id: report_matrix[index] for index, case_id in enumerate(report_ids)}
    with np.load(args.medcpt, allow_pickle=False) as encoded:
        medcpt_ids = [str(case_id) for case_id in encoded["case_ids"].tolist()]
        medcpt_matrix = np.asarray(encoded["embeddings"], dtype=np.float32)
    medcpt_by_id = {case_id: medcpt_matrix[index] for index, case_id in enumerate(medcpt_ids)}
    train_medcpt = np.stack([medcpt_by_id[case_id] for case_id in train_ids])

    query_ids = validation_ids
    query_texts = [
        "\n".join(part for part in (formal[case_id].indication, QUESTIONS[question_type]) if part)
        for case_id in query_ids
        for question_type in QUESTIONS
    ]
    encoded_queries = encode_queries(query_texts, batch_size=32, device=args.device, local_files_only=True)
    gc.collect()
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    query_embedding = {
        (case_id, question_type): encoded_queries[index * len(QUESTIONS) + question_index]
        for index, case_id in enumerate(query_ids)
        for question_index, question_type in enumerate(QUESTIONS)
    }

    checkpoint_states = [
        torch.load(args.checkpoints / f"r5_seed_{seed}.pt", map_location="cpu", weights_only=True)
        for seed in (7041, 7042, 7043, 7044, 7045)
    ]
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
    model = lgb.Booster(model_file=str(args.model))
    term_cache: dict[str, tuple[np.ndarray, int]] = {}
    rows: list[dict[str, Any]] = []
    for position, case_id in enumerate(validation_ids, start=1):
        for question_type in QUESTIONS:
            state = build_retrieval_state(
                runtime,
                formal[case_id],
                image_by_id[case_id],
                case_id,
                question_type,
                query_embedding[(case_id, question_type)],
                train_medcpt,
                raw_cases,
                facts_by_case,
                prepared_by_case,
                leave_one_out=False,
                term_cache=term_cache,
            )
            candidate_ids = state["rrf_rank"][:200]
            indices = [runtime.candidate_ids.index(candidate_id) for candidate_id in candidate_ids]
            light_scores = model.predict(state["features_by_index"][indices])
            light_rank = [candidate_id for _, candidate_id in sorted(zip(light_scores, candidate_ids), key=lambda item: (-float(item[0]), item[1]))]
            query_prepared = prepared_by_case[case_id]
            label_qrels: dict[str, float] = {}
            fact_qrels: dict[str, float] = {}
            for candidate_id in train_ids:
                candidate = prepared_by_case[candidate_id]
                label_qrels[candidate_id] = active_label_similarity(formal[case_id].labels, formal[candidate_id].labels)
                fact_qrels[candidate_id] = radgraph_fact_similarity(formal[case_id].radgraph_facts, formal[candidate_id].radgraph_facts)
            rankings = {
                "r5_full_bank": state["r5_full_rank"],
                "rrf_candidate": state["rrf_rank"][:200],
                "rrf_r5_rerank": state["rrf_r5_rank"][:200],
                "rrf_lambdamart": light_rank,
            }
            qrels = {
                "qrel_v2": state["qrels"],
                "label_only": label_qrels,
                "fact_only": fact_qrels,
            }
            row = {
                "case_id": case_id,
                "question_type": question_type,
                "spectrum": spectrum(raw_cases[case_id]),
                "rankings": {name: ranking[:200] for name, ranking in rankings.items()},
                "metrics": {
                    name: {variant: ndcg(ranking, qrels[variant], 10) for variant in qrels}
                    for name, ranking in rankings.items()
                },
            }
            rows.append(row)
        if position % 50 == 0:
            print(f"validation_rankings={position}/{len(validation_ids)}", flush=True)

    systems = ("r5_full_bank", "rrf_candidate", "rrf_r5_rerank", "rrf_lambdamart")
    variants = ("qrel_v2", "label_only", "fact_only")
    summary: dict[str, Any] = {
        "study": "V12 validation ranking and qrel sensitivity",
        "status": "validation_only_development",
        "no_test_evaluation": True,
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "radgraph_sha256": file_sha256(args.radgraph),
            "split_sha256": file_sha256(args.split),
            "model_sha256": file_sha256(args.model),
            "train_case_count": len(train_ids),
            "validation_case_count": len(validation_ids),
        },
        "metrics": {},
        "bootstrap_vs_r5": {},
        "rows_path": str(args.output.with_name("v12_validation_ranking_rows.jsonl").resolve().relative_to(ROOT)),
    }
    for system in systems:
        summary["metrics"][system] = {}
        for variant in variants:
            selected = [float(row["metrics"][system][variant]) for row in rows]
            summary["metrics"][system][variant] = {
                "ndcg10": mean(selected),
                "normal_ndcg10": mean([float(row["metrics"][system][variant]) for row in rows if row["spectrum"] == "normal"]),
                "abnormal_ndcg10": mean([float(row["metrics"][system][variant]) for row in rows if row["spectrum"] == "abnormal"]),
                "indeterminate_ndcg10": mean([float(row["metrics"][system][variant]) for row in rows if row["spectrum"] == "indeterminate"]),
            }
    for system in ("rrf_candidate", "rrf_r5_rerank", "rrf_lambdamart"):
        summary["bootstrap_vs_r5"][system] = {
            variant: bootstrap(rows, system, "r5_full_bank", variant)
            for variant in variants
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows_path = args.output.with_name("v12_validation_ranking_rows.jsonl").resolve()
    args.output = args.output.resolve()
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": summary["metrics"], "bootstrap_vs_r5": summary["bootstrap_vs_r5"]}, indent=2))


if __name__ == "__main__":
    main()
