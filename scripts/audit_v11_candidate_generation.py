"""Audit first-stage candidate generation on V10 Train/Validation only.

This is a development-only diagnostic. It compares BM25, MedCPT text,
MedSigLIP image and a deterministic RRF union before any fact-aware reranking
or answer generation. It does not touch V10 Test or instantiate a V11 cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.retrieval.candidate_generation import reciprocal_rank_fusion_union  # noqa: E402
from medical_rag.retrieval.medcpt_retriever import encode_queries  # noqa: E402
from medical_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from medical_rag.similar_case.v11_qrel import prepare_qrel_case, qrel_v2_profile_prepared  # noqa: E402


QUESTIONS = {
    "findings": "What are the main radiographic findings?",
    "impression": "What is the most likely radiographic impression?",
    "acute": "Is there an acute cardiopulmonary abnormality? Explain briefly.",
}
DEFAULT_CASES = ROOT / "data/processed/openi_cases.jsonl"
DEFAULT_FACTS = ROOT / "data/processed/v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data/splits/v10/v10_cluster_disjoint_split.json"
DEFAULT_MEDCPT = ROOT / "data/processed/openi_medcpt_full.npz"
DEFAULT_MEDSIGLIP = ROOT / "data/processed/v10_medsiglip_embeddings.npz"
DEFAULT_OUTPUT = ROOT / "data/splits/v11/v11_candidate_generation_audit_summary.json"
DEFAULT_ROWS = ROOT / "experiments/v11_development/v11_candidate_generation_rows.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ndcg(ranked: list[str], qrels: dict[str, float], k: int = 10) -> float:
    import math

    def dcg(values: list[float]) -> float:
        return sum((2.0**value - 1.0) / math.log2(index + 2.0) for index, value in enumerate(values[:k]))

    ideal = dcg(sorted(qrels.values(), reverse=True))
    return dcg([qrels.get(case_id, 0.0) for case_id in ranked]) / ideal if ideal else 0.0


def evaluate_pool(ranked: list[str], qrels: dict[str, float], k: int) -> dict[str, float]:
    relevant = {case_id for case_id, value in qrels.items() if value >= 0.5}
    selected = set(ranked[:k])
    in_pool = selected & relevant
    return {
        "ndcg10": ndcg(ranked, qrels, 10),
        "relevant_count": float(len(relevant)),
        "relevant_in_pool_count": float(len(in_pool)),
        "relevant_recall_at_k": len(in_pool) / len(relevant) if relevant else 0.0,
        "has_relevant_in_pool": float(bool(in_pool)),
        "relevant_outside_pool": float(bool(relevant - selected)),
    }


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--medcpt-index", type=Path, default=DEFAULT_MEDCPT)
    parser.add_argument("--medsiglip", type=Path, default=DEFAULT_MEDSIGLIP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--source-top-k", type=int, default=100)
    parser.add_argument("--pool-k", type=int, default=100)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()

    cases = read_jsonl(args.cases)
    by_id = {str(row["case_id"]): row for row in cases}
    facts_by_case = {
        str(row["case_id"]): tuple(row.get("facts", ()))
        for row in read_jsonl(args.facts)
        if row.get("status") == "ok"
    }
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_ids = [str(value) for value in split["partitions"]["train"]["case_ids"]]
    validation_ids = [str(value) for value in split["partitions"]["validation"]["case_ids"]]
    if args.max_cases > 0:
        validation_ids = sorted(validation_ids)[: args.max_cases]
    train_cases = [by_id[case_id] for case_id in train_ids]
    train_set = set(train_ids)

    medcpt = np.load(args.medcpt_index, allow_pickle=False)
    medcpt_ids = [str(value) for value in medcpt["case_ids"].tolist()]
    medcpt_map = {case_id: index for index, case_id in enumerate(medcpt_ids)}
    missing_medcpt = train_set - set(medcpt_map)
    if missing_medcpt:
        raise RuntimeError(f"MedCPT index misses Train cases: {sorted(missing_medcpt)[:3]}")
    medcpt_embeddings = np.asarray(medcpt["embeddings"], dtype=np.float32)
    train_medcpt = medcpt_embeddings[[medcpt_map[case_id] for case_id in train_ids]]

    medsig = np.load(args.medsiglip, allow_pickle=False)
    medsig_ids = [str(value) for value in medsig["case_ids"].tolist()]
    medsig_map = {case_id: index for index, case_id in enumerate(medsig_ids)}
    missing_medsig = (set(train_ids) | set(validation_ids)) - set(medsig_map)
    if missing_medsig:
        raise RuntimeError(f"MedSigLIP artifact misses cases: {sorted(missing_medsig)[:3]}")
    train_medsig = np.asarray(medsig["case_image_embeddings"], dtype=np.float32)[[medsig_map[case_id] for case_id in train_ids]]
    query_medsig = np.asarray(medsig["case_image_embeddings"], dtype=np.float32)

    query_texts = [
        "\n".join(part for part in (str(by_id[case_id].get("indication", "")), question) if part)
        for case_id in validation_ids
        for question in QUESTIONS.values()
    ]
    query_medcpt = encode_queries(
        query_texts,
        batch_size=args.query_batch_size,
        device=args.device,
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
        query_image = query_medsig[medsig_map[case_id]]
        prepared_query = prepare_qrel_case(query, facts_by_case)
        for question_type, question in QUESTIONS.items():
            query_text = "\n".join(part for part in (str(query.get("indication", "")), question) if part)
            bm25_scores = retriever.score_all(query_text)
            bm25_map = {str(train_cases[index]["case_id"]): float(score) for index, score in enumerate(bm25_scores)}
            bm25_rank = sorted(bm25_map, key=lambda item: (-bm25_map[item], item))
            dense_scores = train_medcpt @ query_medcpt[query_index]
            dense_map = {case_id_: float(score) for case_id_, score in zip(train_ids, dense_scores, strict=True)}
            dense_rank = sorted(dense_map, key=lambda item: (-dense_map[item], item))
            image_scores = train_medsig @ query_image
            image_map = {case_id_: float(score) for case_id_, score in zip(train_ids, image_scores, strict=True)}
            image_rank = sorted(image_map, key=lambda item: (-image_map[item], item))
            union_rank = reciprocal_rank_fusion_union(
                [bm25_rank, dense_rank, image_rank],
                source_top_k=args.source_top_k,
                output_k=args.pool_k,
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
            system_ranks = {
                "bm25": bm25_rank[:args.pool_k],
                "medcpt_text": dense_rank[:args.pool_k],
                "medsiglip_image": image_rank[:args.pool_k],
                "rrf_union": union_rank,
            }
            row: dict[str, Any] = {"query_case_id": case_id, "question_type": question_type}
            for system, ranking in system_ranks.items():
                metrics = evaluate_pool(ranking, qrels, args.pool_k)
                row[system] = metrics
            row["rrf_union_source_overlap"] = {
                "bm25_medcpt": len(set(bm25_rank[:args.source_top_k]) & set(dense_rank[:args.source_top_k])),
                "bm25_medsiglip": len(set(bm25_rank[:args.source_top_k]) & set(image_rank[:args.source_top_k])),
                "medcpt_medsiglip": len(set(dense_rank[:args.source_top_k]) & set(image_rank[:args.source_top_k])),
            }
            rows.append(row)
            query_index += 1

    systems = ("bm25", "medcpt_text", "medsiglip_image", "rrf_union")
    summary = {
        "study": "V11 first-stage candidate generation audit",
        "status": "development_only_no_confirmation",
        "inputs": {
            "cases": str(args.cases.relative_to(ROOT)),
            "facts": str(args.facts.relative_to(ROOT)),
            "split": str(args.split.relative_to(ROOT)),
            "medcpt_index": str(args.medcpt_index.relative_to(ROOT)),
            "medsiglip": str(args.medsiglip.relative_to(ROOT)),
            "cases_sha256": sha256(args.cases),
            "facts_sha256": sha256(args.facts),
            "split_sha256": sha256(args.split),
            "medcpt_index_sha256": sha256(args.medcpt_index),
            "medsiglip_sha256": sha256(args.medsiglip),
            "train_case_count": len(train_ids),
            "validation_case_count": len(validation_ids),
        },
        "design": {
            "source_top_k": args.source_top_k,
            "pool_k": args.pool_k,
            "rrf_constant": 60,
            "query": "indication + fixed question",
            "qrel": "full-bank report-derived qrel-v2; threshold=0.5",
            "target_outside_pool_policy": "retain as first-stage retrieval failure",
        },
        "metrics": {
            system: {
                metric: mean([row[system] for row in rows], metric)
                for metric in ("ndcg10", "relevant_recall_at_k", "has_relevant_in_pool", "relevant_outside_pool")
            }
            for system in systems
        },
        "interpretation_boundary": "All relevance and candidate labels are report-derived development proxies; no clinical correctness, safety, human review, external validation or confirmation claim is made.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
