"""Run the predeclared V12 retrieval pilot on Train/Validation only."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from lightgbm import LGBMRanker, early_stopping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.retrieval.candidate_generation import reciprocal_rank_fusion_union  # noqa: E402
from medical_rag.retrieval.medcpt_retriever import encode_queries  # noqa: E402
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v10_reranker import augment_r4_features  # noqa: E402
from medical_rag.similar_case.v10_runtime import FrozenR5Runtime, QUESTIONS, r4_feature_matrix  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from medical_rag.similar_case.v11_qrel import prepare_qrel_case, qrel_v2_profile_prepared  # noqa: E402
from train_v9_learned_reranker import exact_leave_one_out_bm25_scores  # noqa: E402

warnings.filterwarnings("ignore", message="Found 'eval_at' in params")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightgbm")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ndcg(ranked: Sequence[str], qrels: dict[str, float], k: int = 10) -> float:
    def dcg(values: Sequence[float]) -> float:
        return sum((2.0**value - 1.0) / np.log2(index + 2.0) for index, value in enumerate(values[:k]))

    ideal = dcg(sorted(qrels.values(), reverse=True))
    if ideal <= 0:
        return 0.0
    return float(dcg([qrels.get(case_id, 0.0) for case_id in ranked]) / ideal)


def spectrum(raw_case: dict[str, Any]) -> str:
    value = str(raw_case.get("problems", "")).strip().lower()
    if value == "normal":
        return "normal"
    if value in {"", "no indexing"}:
        return "indeterminate"
    return "abnormal"


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def bootstrap_difference(rows: list[dict[str, Any]], new_key: str, baseline_key: str) -> dict[str, float]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    case_ids = sorted(by_case)
    case_diffs = np.asarray(
        [mean([float(row[new_key]) - float(row[baseline_key]) for row in by_case[case_id]]) for case_id in case_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(2026)
    samples = np.empty(10000, dtype=np.float64)
    for index in range(len(samples)):
        samples[index] = float(rng.choice(case_diffs, size=len(case_diffs), replace=True).mean())
    return {
        "difference": float(case_diffs.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "case_count": len(case_ids),
    }


def build_retrieval_state(
    runtime: FrozenR5Runtime,
    query: Any,
    query_image: np.ndarray,
    query_id: str,
    question_type: str,
    query_medcpt: np.ndarray,
    bank_medcpt: np.ndarray,
    raw_cases: dict[str, dict[str, Any]],
    facts_by_case: dict[str, tuple[str, ...]],
    prepared_by_case: dict[str, Any],
    *,
    leave_one_out: bool,
    term_cache: dict[str, tuple[np.ndarray, int]],
) -> dict[str, Any]:
    query_text = "\n".join(part for part in (query.indication, QUESTIONS[question_type]) if part)
    bm25 = np.asarray(runtime.bm25.score_all(query_text), dtype=np.float32)
    excluded_index = runtime.candidate_ids.index(query_id) if query_id in runtime.candidate_ids else None
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
    text_rank = sorted(range(len(bm25)), key=lambda index: (-float(bm25[index]), index))
    dense_scores = np.asarray(bank_medcpt @ query_medcpt, dtype=np.float32)
    dense_rank = sorted(range(len(dense_scores)), key=lambda index: (-float(dense_scores[index]), index))
    image_rank = sorted(range(len(image)), key=lambda index: (-float(image[index]), index))
    rrf_ids = reciprocal_rank_fusion_union(
        [
            [runtime.candidate_ids[index] for index in text_rank],
            [runtime.candidate_ids[index] for index in dense_rank],
            [runtime.candidate_ids[index] for index in image_rank],
        ],
        source_top_k=100,
        output_k=200,
    )
    if query_id in runtime.candidate_ids:
        rrf_ids = [case_id for case_id in rrf_ids if case_id != query_id]
    rrf_indices = [runtime.candidate_ids.index(case_id) for case_id in rrf_ids]
    r4 = r4_feature_matrix(bm25, image, report, question_type=question_type)
    fact_features = runtime.fact_index.query_features(query_text)
    r5 = augment_r4_features(r4, fact_features)
    with torch.inference_mode():
        seed_scores = np.stack([model(torch.from_numpy(r5)).numpy() for model in runtime.models])
    r5_scores = seed_scores.mean(axis=0)
    r5_full_rank = [runtime.candidate_ids[index] for index in np.lexsort((np.arange(len(r5_scores)), -r5_scores))]
    rrf_r5_rank = [case_id for case_id in sorted(rrf_ids, key=lambda case_id: (-float(r5_scores[runtime.candidate_ids.index(case_id)]), case_id))]
    return {
        "query_id": query_id,
        "question_type": question_type,
        "spectrum": spectrum(raw_cases[query_id]),
        "rrf_rank": rrf_ids,
        "rrf_indices": rrf_indices,
        "r5_full_rank": r5_full_rank,
        "rrf_r5_rank": rrf_r5_rank,
        "features_by_index": r5,
        "qrels": {
            candidate_id: float(
                qrel_v2_profile_prepared(
                    prepared_by_case[query_id],
                    prepared_by_case[candidate_id],
                )["qrel_v2"]
            )
            for candidate_id in runtime.candidate_ids
            if candidate_id != query_id
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--radgraph", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/processed/v10_medsiglip_embeddings.npz")
    parser.add_argument("--medcpt", type=Path, default=ROOT / "data/processed/openi_medcpt_full.npz")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "experiments/v10_publication/reranker_checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_retrieval_pilot.json")
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
    facts_by_case = {
        case_id: tuple(radgraph[case_id].facts)
        for case_id in train_ids + validation_ids
        if case_id in radgraph
    }
    prepared_by_case = {
        case_id: prepare_qrel_case(raw_cases[case_id], facts_by_case)
        for case_id in train_ids + validation_ids
    }

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
    all_query_ids = train_ids + validation_ids
    query_texts = [
        "\n".join(part for part in (formal[case_id].indication, QUESTIONS[question_type]) if part)
        for case_id in all_query_ids
        for question_type in QUESTIONS
    ]
    encoded_queries = encode_queries(query_texts, batch_size=32, device=args.device, local_files_only=True)
    gc.collect()
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    query_embedding = {
        (case_id, question_type): encoded_queries[index * len(QUESTIONS) + question_index]
        for index, case_id in enumerate(all_query_ids)
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
    term_cache: dict[str, tuple[np.ndarray, int]] = {}
    validation_states: list[dict[str, Any]] = []
    for position, case_id in enumerate(validation_ids, start=1):
        for question_type in QUESTIONS:
            validation_states.append(
                build_retrieval_state(
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
            )
        if position % 50 == 0:
            print(f"validation_states={position}/{len(validation_ids)}", flush=True)

    # The R5 feature matrix is reused for one explicitly predeclared LightGBM
    # pilot. Training uses the V10 pairwise-fit role and RRF Top-200 groups.
    roles = read_json(ROOT / "data/splits/v10/v10_reranker_roles.json")
    fit_ids = [case_id for case_id in roles["roles"]["pairwise_fit"]["case_ids"] if case_id in train_ids]
    internal_ids = [case_id for case_id in roles["roles"]["internal_early_stop"]["case_ids"] if case_id in train_ids]
    train_states: list[dict[str, Any]] = []
    for position, case_id in enumerate(fit_ids + internal_ids, start=1):
        for question_type in QUESTIONS:
            train_states.append(
                build_retrieval_state(
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
                    leave_one_out=True,
                    term_cache=term_cache,
                )
            )
        if position % 100 == 0:
            print(f"train_states={position}/{len(fit_ids + internal_ids)}", flush=True)

    fit_set = set(fit_ids)
    internal_set = set(internal_ids)
    fit_rows = [state for state in train_states if state["query_id"] in fit_set]
    internal_rows = [state for state in train_states if state["query_id"] in internal_set]

    def matrix_and_labels(states: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[int]]:
        matrices = []
        labels = []
        groups = []
        for state in states:
            indices = [runtime.candidate_ids.index(case_id) for case_id in state["rrf_rank"][:200]]
            matrices.append(state["features_by_index"][indices])
            labels.extend([state["qrels"].get(runtime.candidate_ids[index], 0.0) for index in indices])
            groups.append(len(indices))
        return np.concatenate(matrices, axis=0), np.asarray(labels, dtype=np.float32), groups

    x_fit, y_fit, g_fit = matrix_and_labels(fit_rows)
    x_internal, y_internal, g_internal = matrix_and_labels(internal_rows)
    # LightGBM's LambdaRank objective requires integer relevance labels. The
    # evaluation remains on the original continuous qrel-v2 values; training
    # uses a fixed ten-level monotone quantization only.
    y_fit_rank = np.clip(np.rint(y_fit * 10.0), 0, 10).astype(np.int32)
    y_internal_rank = np.clip(np.rint(y_internal * 10.0), 0, 10).astype(np.int32)
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[10],
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=40,
        reg_lambda=1.0,
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

    def lightgbm_rank(state: dict[str, Any]) -> list[str]:
        candidate_ids = state["rrf_rank"][:200]
        indices = [runtime.candidate_ids.index(case_id) for case_id in candidate_ids]
        scores = ranker.predict(state["features_by_index"][indices])
        return [case_id for _, case_id in sorted(zip(scores, candidate_ids), key=lambda item: (-float(item[0]), item[1]))]

    pilot_rows: list[dict[str, Any]] = []
    for state in validation_states:
        qrels = state["qrels"]
        rankings = {
            "r5_full_bank": state["r5_full_rank"],
            "rrf_candidate": state["rrf_rank"][:200],
            "rrf_r5_rerank": state["rrf_r5_rank"][:200],
            "rrf_lambdamart": lightgbm_rank(state),
        }
        row = {
            "case_id": state["query_id"],
            "question_type": state["question_type"],
            "spectrum": state["spectrum"],
        }
        for name, ranking in rankings.items():
            row[name] = ndcg(ranking, qrels, 10)
            row[f"{name}_top200"] = ranking[:200]
        row["target_in_rrf_top200"] = float(state["query_id"] in state["rrf_rank"][:200])
        pilot_rows.append(row)

    systems = ("r5_full_bank", "rrf_candidate", "rrf_r5_rerank", "rrf_lambdamart")
    summary: dict[str, Any] = {
        "study": "V12 retrieval pilot",
        "status": "validation_only_development",
        "no_test_evaluation": True,
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "radgraph_sha256": file_sha256(args.radgraph),
            "split_sha256": file_sha256(args.split),
            "embeddings_sha256": file_sha256(args.embeddings),
            "medcpt_sha256": file_sha256(args.medcpt),
            "fit_query_count": len(fit_rows),
            "internal_query_count": len(internal_rows),
            "validation_query_count": len(validation_states),
            "candidate_bank_count": len(train_ids),
        },
        "ranker": {
            "type": "LightGBM LambdaMART",
            "version": "4.7.0",
            "features": 17,
            "candidate_pool": "RRF Top-200",
            "best_iteration": int(ranker.best_iteration_ or 0),
            "qrel": "qrel-v2 full report-derived proxy",
        },
        "metrics": {},
        "bootstrap_vs_r5": {},
        "rows_path": str(args.output.with_name("v12_retrieval_pilot_rows.jsonl").resolve().relative_to(ROOT)),
    }
    for system in systems:
        selected = [float(row[system]) for row in pilot_rows]
        summary["metrics"][system] = {
            "ndcg10": mean(selected),
            "normal_ndcg10": mean([float(row[system]) for row in pilot_rows if row["spectrum"] == "normal"]),
            "abnormal_ndcg10": mean([float(row[system]) for row in pilot_rows if row["spectrum"] == "abnormal"]),
            "indeterminate_ndcg10": mean([float(row[system]) for row in pilot_rows if row["spectrum"] == "indeterminate"]),
        }
    for system in ("rrf_candidate", "rrf_r5_rerank", "rrf_lambdamart"):
        summary["bootstrap_vs_r5"][system] = bootstrap_difference(pilot_rows, system, "r5_full_bank")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model_path = args.output.with_name("v12_lambdamart.txt").resolve()
    ranker.booster_.save_model(str(model_path))
    summary["ranker"]["model_path"] = str(model_path.relative_to(ROOT))
    summary["ranker"]["model_sha256"] = file_sha256(model_path)
    rows_path = args.output.with_name("v12_retrieval_pilot_rows.jsonl").resolve()
    rows_path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in pilot_rows),
        encoding="utf-8",
    )
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics": summary["metrics"], "bootstrap_vs_r5": summary["bootstrap_vs_r5"]}, indent=2))


if __name__ == "__main__":
    main()
