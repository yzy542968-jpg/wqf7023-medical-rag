"""Run development-only V12 bottleneck diagnostics from existing artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v11_candidate_generation import evaluate_pool, ndcg  # noqa: E402
from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from medical_rag.retrieval.candidate_generation import reciprocal_rank_fusion_union  # noqa: E402
from medical_rag.retrieval.medcpt_retriever import encode_queries  # noqa: E402
from medical_rag.similar_case.v11_qrel import (  # noqa: E402
    prepare_qrel_case,
    qrel_v2_profile_prepared,
)


QUESTIONS = {
    "findings": "What are the main radiographic findings?",
    "impression": "What is the most likely radiographic impression?",
    "acute": "Is there an acute cardiopulmonary abnormality? Explain briefly.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def spectrum(case: dict[str, Any]) -> str:
    value = str(case.get("problems", "")).strip().lower()
    if value == "normal":
        return "normal"
    if value in {"", "no indexing"}:
        return "indeterminate"
    return "abnormal"


def mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else 0.0


def retrieval_diagnostics(
    cases_path: Path,
    facts_path: Path,
    split_path: Path,
    medcpt_path: Path,
    medsiglip_path: Path,
    device: str | None,
) -> dict[str, Any]:
    cases = read_jsonl(cases_path)
    by_id = {str(row["case_id"]): row for row in cases}
    facts_by_case = {
        str(row["case_id"]): tuple(row.get("facts", ()))
        for row in read_jsonl(facts_path)
        if row.get("status") == "ok"
    }
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = [str(value) for value in split["partitions"]["train"]["case_ids"]]
    validation_ids = [str(value) for value in split["partitions"]["validation"]["case_ids"]]
    train_cases = [by_id[case_id] for case_id in train_ids]
    train_set = set(train_ids)

    medcpt = np.load(medcpt_path, allow_pickle=False)
    medcpt_ids = [str(value) for value in medcpt["case_ids"].tolist()]
    medcpt_map = {case_id: index for index, case_id in enumerate(medcpt_ids)}
    missing = train_set - set(medcpt_map)
    if missing:
        raise RuntimeError(f"MedCPT index misses Train cases: {sorted(missing)[:3]}")
    train_medcpt = np.asarray(medcpt["embeddings"], dtype=np.float32)[
        [medcpt_map[case_id] for case_id in train_ids]
    ]

    medsig = np.load(medsiglip_path, allow_pickle=False)
    medsig_ids = [str(value) for value in medsig["case_ids"].tolist()]
    medsig_map = {case_id: index for index, case_id in enumerate(medsig_ids)}
    missing = (set(train_ids) | set(validation_ids)) - set(medsig_map)
    if missing:
        raise RuntimeError(f"MedSigLIP artifact misses cases: {sorted(missing)[:3]}")
    all_image_embeddings = np.asarray(medsig["case_image_embeddings"], dtype=np.float32)
    train_images = all_image_embeddings[[medsig_map[case_id] for case_id in train_ids]]

    query_texts = [
        "\n".join(part for part in (str(by_id[case_id].get("indication", "")), question) if part)
        for case_id in validation_ids
        for question in QUESTIONS.values()
    ]
    query_medcpt = encode_queries(
        query_texts,
        batch_size=32,
        device=device,
        local_files_only=True,
    )
    retriever = BM25Retriever().fit(train_cases)
    prepared_candidates = {
        case_id: prepare_qrel_case(by_id[case_id], facts_by_case)
        for case_id in train_ids
    }
    rows: list[dict[str, Any]] = []
    query_index = 0
    for case_id in validation_ids:
        query = by_id[case_id]
        query_image = all_image_embeddings[medsig_map[case_id]]
        prepared_query = prepare_qrel_case(query, facts_by_case)
        query_spectrum = spectrum(query)
        for question_type, question in QUESTIONS.items():
            query_text = "\n".join(part for part in (str(query.get("indication", "")), question) if part)
            bm25_scores = retriever.score_all(query_text)
            bm25_map = {
                str(train_cases[index]["case_id"]): float(score)
                for index, score in enumerate(bm25_scores)
            }
            bm25_rank = sorted(bm25_map, key=lambda item: (-bm25_map[item], item))
            dense_scores = train_medcpt @ query_medcpt[query_index]
            dense_map = {candidate_id: float(score) for candidate_id, score in zip(train_ids, dense_scores, strict=True)}
            dense_rank = sorted(dense_map, key=lambda item: (-dense_map[item], item))
            image_scores = train_images @ query_image
            image_map = {candidate_id: float(score) for candidate_id, score in zip(train_ids, image_scores, strict=True)}
            image_rank = sorted(image_map, key=lambda item: (-image_map[item], item))
            rrf_rank = reciprocal_rank_fusion_union(
                [bm25_rank, dense_rank, image_rank],
                source_top_k=100,
                output_k=200,
            )
            qrels = {
                candidate_id: float(
                    qrel_v2_profile_prepared(
                        prepared_query,
                        prepared_candidates[candidate_id],
                    )["qrel_v2"]
                )
                for candidate_id in train_ids
            }
            rankings = {
                "bm25": bm25_rank,
                "medcpt_text": dense_rank,
                "medsiglip_image": image_rank,
                "rrf_union": rrf_rank,
            }
            for pool_k in (50, 100, 200):
                for system, ranking in rankings.items():
                    pool = ranking[:pool_k]
                    metrics = evaluate_pool(pool, qrels, pool_k)
                    oracle = sorted(pool, key=lambda candidate: (-qrels.get(candidate, 0.0), candidate))
                    row = {
                        "query_case_id": case_id,
                        "question_type": question_type,
                        "spectrum": query_spectrum,
                        "pool_k": pool_k,
                        "system": system,
                        **metrics,
                        "best_qrel_in_pool": max((qrels.get(candidate, 0.0) for candidate in pool), default=0.0),
                        "oracle_ndcg10_in_pool": ndcg(oracle, qrels, 10),
                    }
                    rows.append(row)
            query_index += 1

    summary: dict[str, Any] = {
        "study": "V12 development bottleneck diagnostics",
        "status": "development_only_no_confirmation",
        "inputs": {
            "cases_sha256": file_sha256(cases_path),
            "facts_sha256": file_sha256(facts_path),
            "split_sha256": file_sha256(split_path),
            "medcpt_sha256": file_sha256(medcpt_path),
            "medsiglip_sha256": file_sha256(medsiglip_path),
            "train_case_count": len(train_ids),
            "validation_case_count": len(validation_ids),
        },
        "metrics": {},
        "rows": rows,
    }
    for pool_k in (50, 100, 200):
        for system in ("bm25", "medcpt_text", "medsiglip_image", "rrf_union"):
            selected = [row for row in rows if row["pool_k"] == pool_k and row["system"] == system]
            summary["metrics"][f"{system}@{pool_k}"] = {
                "ndcg10": mean(selected, "ndcg10"),
                "relevant_recall_at_k": mean(selected, "relevant_recall_at_k"),
                "has_relevant_in_pool": mean(selected, "has_relevant_in_pool"),
                "best_qrel_in_pool": mean(selected, "best_qrel_in_pool"),
                "oracle_ndcg10_in_pool": mean(selected, "oracle_ndcg10_in_pool"),
            }
            for subgroup in ("normal", "abnormal", "indeterminate"):
                subgroup_rows = [row for row in selected if row["spectrum"] == subgroup]
                if subgroup_rows:
                    summary["metrics"][f"{system}@{pool_k}:{subgroup}"] = {
                        "query_count": len(subgroup_rows),
                        "ndcg10": mean(subgroup_rows, "ndcg10"),
                        "oracle_ndcg10_in_pool": mean(subgroup_rows, "oracle_ndcg10_in_pool"),
                        "has_relevant_in_pool": mean(subgroup_rows, "has_relevant_in_pool"),
                    }
    return summary


def generation_diagnostics(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    result: dict[str, Any] = {
        "study": "V12 post-hoc frozen generation diagnostics",
        "status": "descriptive_frozen_output_analysis",
        "source_sha256": file_sha256(path),
        "row_count": len(rows),
        "policies": {},
    }
    for policy in sorted({str(row.get("evidence_policy", "unknown")) for row in rows}):
        selected = [row for row in rows if str(row.get("evidence_policy", "unknown")) == policy]
        raw_scores = []
        final_scores = []
        for row in selected:
            raw = str(row.get("raw_answer_stage", ""))
            raw = raw.replace("<end_of_turn>", "").strip()
            raw_scores.append(token_f1(raw, str(row.get("reference_answer", ""))))
            final_scores.append(float(row.get("token_f1", 0.0)))
        result["policies"][policy] = {
            "row_count": len(selected),
            "raw_token_f1": statistics.fmean(raw_scores) if raw_scores else 0.0,
            "final_token_f1": statistics.fmean(final_scores) if final_scores else 0.0,
            "raw_minus_final_token_f1": (statistics.fmean(raw_scores) - statistics.fmean(final_scores))
            if selected
            else 0.0,
            "answer_ceiling_rate": statistics.fmean(
                bool(row.get("hit_answer_token_ceiling", False)) for row in selected
            )
            if selected
            else 0.0,
            "mean_final_output_tokens": statistics.fmean(
                float(row.get("answer_output_tokens", 0.0)) for row in selected
            )
            if selected
            else 0.0,
            "by_question_type": {},
        }
        for question_type in ("findings", "impression", "acute"):
            subset = [row for row in selected if row.get("question_type") == question_type]
            if not subset:
                continue
            raw_subset = [
                token_f1(str(row.get("raw_answer_stage", "")).replace("<end_of_turn>", "").strip(), str(row.get("reference_answer", "")))
                for row in subset
            ]
            result["policies"][policy]["by_question_type"][question_type] = {
                "row_count": len(subset),
                "raw_token_f1": statistics.fmean(raw_subset),
                "final_token_f1": statistics.fmean(float(row.get("token_f1", 0.0)) for row in subset),
                "ceiling_rate": statistics.fmean(bool(row.get("hit_answer_token_ceiling", False)) for row in subset),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--facts", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--medcpt", type=Path, default=ROOT / "data/processed/openi_medcpt_full.npz")
    parser.add_argument("--medsiglip", type=Path, default=ROOT / "data/processed/v10_medsiglip_embeddings.npz")
    parser.add_argument("--generation-rows", type=Path, default=ROOT / "experiments/v10_publication/v10_evidence_generation_final_rows.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v12_optimization/diagnostics/v12_diagnostics.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()

    retrieval = retrieval_diagnostics(
        args.cases,
        args.facts,
        args.split,
        args.medcpt,
        args.medsiglip,
        args.device,
    )
    generation = generation_diagnostics(args.generation_rows)
    output = {"retrieval": retrieval, "generation": generation}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serializable = json.dumps(output, indent=2, ensure_ascii=True)
    args.output.write_text(serializable + "\n", encoding="utf-8")
    print(json.dumps({"retrieval_metrics": retrieval["metrics"], "generation": generation["policies"]}, indent=2))


if __name__ == "__main__":
    main()
