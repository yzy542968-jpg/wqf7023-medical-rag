from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import file_sha256, read_json  # noqa: E402
from medical_rag.evaluation.graded_retrieval import (  # noqa: E402
    binary_recall_at_k,
    ndcg_at_k,
    reciprocal_rank_at_threshold,
)
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.relevance import build_query_qrels  # noqa: E402
from medical_rag.similar_case.text_baseline import SimilarCaseBM25Retriever  # noqa: E402


DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v9" / "v9_full_source_split.json"
DEFAULT_PROTOCOL = ROOT / "config" / "v9_similar_case_rag_development.json"
DEFAULT_PREPROCESSING = ROOT / "config" / "v9_radgraph_preprocessing.json"
DEFAULT_ROWS = ROOT / "experiments" / "post_submission_v9" / "v9_bm25_validation_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v9" / "v9_bm25_validation_summary.json"


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    names = ("ndcg@1", "ndcg@5", "ndcg@10", "recall@1", "recall@5", "recall@10", "mrr")
    return {
        name: statistics.fmean(float(row[name]) for row in rows) for name in names
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V9 BM25 validation baseline.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--preprocessing", type=Path, default=DEFAULT_PREPROCESSING)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument("--binary-threshold", type=float, default=0.50)
    args = parser.parse_args()

    started = time.perf_counter()
    preprocessing = read_json(args.preprocessing)
    split = read_json(args.split)
    protocol = read_json(args.protocol)
    if file_sha256(args.cases) != preprocessing["source"]["sha256"]:
        raise RuntimeError("V9 source differs from the frozen preprocessing protocol.")
    if file_sha256(args.split) != preprocessing["split"]["sha256"]:
        raise RuntimeError("V9 split differs from the frozen preprocessing protocol.")

    all_cases = read_openi_paired_cases(
        args.cases,
        source_unique_patient=True,
        radgraph_path=args.radgraph,
    )
    by_id = {case.study_id: case for case in all_cases}
    train_ids = set(split["partitions"]["train"]["case_ids"])
    validation_ids = set(split["partitions"]["validation"]["case_ids"])
    test_ids = set(split["partitions"]["test"]["case_ids"])
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise RuntimeError("V9 split partitions overlap.")

    bank = sorted(
        (
            by_id[case_id]
            for case_id in train_ids
            if by_id[case_id].metadata["radgraph_annotation_available"] is True
        ),
        key=lambda case: case.study_id,
    )
    validation = sorted(
        (
            by_id[case_id]
            for case_id in validation_ids
            if by_id[case_id].metadata["radgraph_annotation_available"] is True
        ),
        key=lambda case: case.study_id,
    )
    expected = preprocessing["primary_frames"]
    if len(bank) != expected["shared_candidate_bank"]:
        raise RuntimeError("Shared V9 candidate-bank count changed.")
    if len(validation) != expected["validation_qrel_queries"]:
        raise RuntimeError("V9 validation qrel-frame count changed.")
    if any(case.study_id in test_ids for case in bank + validation):
        raise RuntimeError("Test cases entered the V9 development runner.")

    retriever = SimilarCaseBM25Retriever(k1=args.k1, b=args.b).fit(
        bank, require_patient_ids=True
    )
    questions = protocol["question_suite"]
    rows: list[dict[str, Any]] = []
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    with args.rows_output.open("w", encoding="utf-8", newline="\n") as handle:
        for case_index, query in enumerate(validation, start=1):
            qrels = build_query_qrels(query, bank)
            relevant_count = sum(
                float(gain) >= args.binary_threshold for gain in qrels.values()
            )
            ideal_max_gain = max(qrels.values(), default=0.0)
            for question_type, question in questions.items():
                ranked = retriever.search(query, question, top_k=len(bank))
                ranking = [row.study_id for row in ranked]
                metrics = {
                    "ndcg@1": ndcg_at_k(qrels, ranking, 1),
                    "ndcg@5": ndcg_at_k(qrels, ranking, 5),
                    "ndcg@10": ndcg_at_k(qrels, ranking, 10),
                    "recall@1": binary_recall_at_k(
                        qrels, ranking, 1, threshold=args.binary_threshold
                    ),
                    "recall@5": binary_recall_at_k(
                        qrels, ranking, 5, threshold=args.binary_threshold
                    ),
                    "recall@10": binary_recall_at_k(
                        qrels, ranking, 10, threshold=args.binary_threshold
                    ),
                    "mrr": reciprocal_rank_at_threshold(
                        qrels, ranking, threshold=args.binary_threshold
                    ),
                }
                output = {
                    "qid": f"{query.study_id}:{question_type}",
                    "case_id": query.study_id,
                    "question_type": question_type,
                    "system": "bm25_text",
                    "candidate_bank_count": len(bank),
                    "relevant_candidate_count": relevant_count,
                    "ideal_max_gain": ideal_max_gain,
                    "top10_case_ids": ranking[:10],
                    "top10_scores": [row.score for row in ranked[:10]],
                    "top10_gains": [qrels[case_id] for case_id in ranking[:10]],
                    **metrics,
                }
                rows.append(output)
                handle.write(json.dumps(output, ensure_ascii=True, sort_keys=True) + "\n")
            if case_index % 25 == 0 or case_index == len(validation):
                print(f"validation_cases={case_index}/{len(validation)}", flush=True)

    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_question[row["question_type"]].append(row)
    summary = {
        "study": "V9 development BM25 validation baseline",
        "status": "development_validation_outcome_test_not_executed",
        "system": "bm25_text",
        "parameters": {
            "k1": args.k1,
            "b": args.b,
            "binary_relevance_threshold": args.binary_threshold,
            "query": "indication + fixed question",
            "document": "historical findings + impression",
            "tie_rule": "score_descending_then_canonical_case_id_via_sorted_bank",
        },
        "source": {
            "path": portable_path(args.cases),
            "sha256": file_sha256(args.cases),
        },
        "radgraph": {
            "path": portable_path(args.radgraph),
            "sha256": file_sha256(args.radgraph),
            "available_to_retrieval": False,
        },
        "split": {
            "path": portable_path(args.split),
            "sha256": file_sha256(args.split),
            "test_queries_executed": 0,
        },
        "candidate_bank_count": len(bank),
        "validation_case_count": len(validation),
        "validation_question_count": len(rows),
        "aggregate_case_grouped_equal_question_metrics": mean_metrics(rows),
        "metrics_by_question_type": {
            question_type: mean_metrics(question_rows)
            for question_type, question_rows in sorted(by_question.items())
        },
        "relevance_diagnostics": {
            "mean_relevant_candidates_per_case": statistics.fmean(
                row["relevant_candidate_count"] for row in rows[:: len(questions)]
            ),
            "minimum_relevant_candidates_per_case": min(
                row["relevant_candidate_count"] for row in rows[:: len(questions)]
            ),
            "maximum_relevant_candidates_per_case": max(
                row["relevant_candidate_count"] for row in rows[:: len(questions)]
            ),
        },
        "rows_output": {
            "path": portable_path(args.rows_output),
            "sha256": file_sha256(args.rows_output),
            "committed_to_public_repository": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "hidden_target_report_available_to_retrieval": False,
        "v9_test_outcomes_inspected": False,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
