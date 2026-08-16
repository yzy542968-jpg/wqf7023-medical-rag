from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.case_scoped_benchmark import expected_section
from medical_rag.evaluation.metrics import evaluate_retrieval
from medical_rag.retrieval.scoped_chunk_retriever import ScopedBM25ChunkRetriever


SYSTEMS = ("global_bm25", "case_scoped_bm25", "case_scoped_agent_routed_bm25")


def evaluate_system(
    questions: list[dict[str, Any]],
    retriever: ScopedBM25ChunkRetriever,
    system: str,
    top_k: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    qrels: dict[str, set[str]] = {}
    rankings: dict[str, list[str]] = {}
    rows: list[dict[str, Any]] = []
    for question in questions:
        case_id = None if system == "global_bm25" else question["scope_case_id"]
        sections = (
            {expected_section(question["question_type"])}
            if system == "case_scoped_agent_routed_bm25"
            else None
        )
        results = retriever.search(
            question["question"],
            top_k=top_k,
            case_id=case_id,
            allowed_sections=sections,
        )
        retrieved = [row["chunk_id"] for row in results]
        relevant = set(question["relevant_chunk_ids"])
        qrels[question["qid"]] = relevant
        rankings[question["qid"]] = retrieved
        rows.append(
            {
                "qid": question["qid"],
                "case_id": question["case_id"],
                "question_type": question["question_type"],
                "system": system,
                "relevant_chunk_ids": sorted(relevant),
                "retrieved_chunk_ids": retrieved,
                "retrieved_case_ids": [row["case_id"] for row in results],
                "retrieved_sections": [row["section"] for row in results],
                "scores": [row["score"] for row in results],
            }
        )
    metrics = evaluate_retrieval(qrels, rankings, k_values=(1, 3, 5))
    metrics["case_scope_accuracy@1"] = mean(
        bool(row["retrieved_case_ids"]) and row["retrieved_case_ids"][0] == row["case_id"]
        for row in rows
    )
    metrics["n"] = len(rows)
    return metrics, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate global, case-scoped, and routed BM25 on benchmark v2.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_scoped_benchmark_v2.json",
    )
    parser.add_argument(
        "--split",
        choices=["development", "calibration", "test", "confirmation"],
        default="test",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "benchmark_v2" / "retrieval",
    )
    args = parser.parse_args()

    payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
    qids = set(payload["split"][args.split]["qids"])
    questions = [row for row in payload["questions"] if row["qid"] in qids]
    retriever = ScopedBM25ChunkRetriever().fit(payload["chunks"])
    summary: dict[str, Any] = {
        "benchmark": payload["benchmark"],
        "split": args.split,
        "case_count": payload["split"][args.split]["case_count"],
        "question_count": len(questions),
        "top_k": args.top_k,
        "systems": {},
    }
    all_rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        metrics, rows = evaluate_system(questions, retriever, system, args.top_k)
        summary["systems"][system] = metrics
        all_rows.extend(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.split}_retrieval_summary.json"
    rows_path = args.output_dir / f"{args.split}_retrieval_rows.jsonl"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({**summary, "summary_path": str(summary_path), "rows_path": str(rows_path)}, indent=2))


if __name__ == "__main__":
    main()
