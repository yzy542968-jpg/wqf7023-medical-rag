from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.metrics import evaluate_retrieval
from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.hybrid_retriever import minmax
from medical_rag.retrieval.medcpt_retriever import MedCPTRetriever, encode_queries
from medical_rag.retrieval.tfidf_retriever import _tokens, load_cases_jsonl


def rankings_for_alpha(
    questions: list[dict],
    bm25_scores: np.ndarray,
    medcpt_scores: np.ndarray,
    case_ids: list[str],
    alpha: float,
    top_k: int,
) -> dict[str, list[str]]:
    rankings: dict[str, list[str]] = {}
    for row_index, item in enumerate(questions):
        scores = alpha * minmax(medcpt_scores[row_index]) + (1.0 - alpha) * minmax(
            bm25_scores[row_index]
        )
        ranked = scores.argsort()[::-1][:top_k]
        rankings[str(item["qid"])] = [case_ids[int(index)] for index in ranked]
    return rankings


def subset_metrics(
    questions: list[dict], rankings: dict[str, list[str]], selected_qids: set[str]
) -> dict[str, float]:
    qrels = {
        str(item["qid"]): {str(case_id) for case_id in item["relevant_case_ids"]}
        for item in questions
        if str(item["qid"]) in selected_qids
    }
    selected_rankings = {qid: rankings[qid] for qid in qrels}
    return evaluate_retrieval(qrels, selected_rankings, k_values=(1, 3, 5, 10, 20))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select hybrid alpha on a grouped development split and evaluate once on test."
    )
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "data" / "processed" / "openi_cases.jsonl"
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_medcpt_full.npz",
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_qa_seed_clean.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "openi_qa_grouped_case_seed7023.json",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--selection-metric", default="mrr", choices=["mrr", "hit@1", "hit@5"])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "final_optimized" / "retrieval",
    )
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    case_ids = [str(case["case_id"]) for case in cases]
    case_position = {case_id: index for index, case_id in enumerate(case_ids)}
    qa_payload = json.loads(args.qa.read_text(encoding="utf-8"))
    questions = qa_payload["questions"]
    split = json.loads(args.split.read_text(encoding="utf-8"))
    development_qids = set(split["development"]["qids"])
    test_qids = set(split["test"]["qids"])

    bm25 = BM25Retriever().fit(cases)
    medcpt = MedCPTRetriever.from_index(args.cases, args.index)
    medcpt_position = {case_id: index for index, case_id in enumerate(medcpt.case_ids)}
    query_embeddings = encode_queries(
        [str(item["question"]) for item in questions],
        batch_size=args.batch_size,
        device=args.device,
    )

    bm25_matrix = np.zeros((len(questions), len(cases)), dtype="float32")
    medcpt_matrix = np.zeros((len(questions), len(cases)), dtype="float32")
    for query_index, item in enumerate(questions):
        query_terms = _tokens(str(item["question"]))
        bm25_matrix[query_index] = np.array(
            [bm25._score_document(query_terms, index) for index in range(len(cases))],
            dtype="float32",
        )
        indexed_scores = medcpt.embeddings @ query_embeddings[query_index]
        for case_id, medcpt_index in medcpt_position.items():
            case_index = case_position.get(case_id)
            if case_index is not None:
                medcpt_matrix[query_index, case_index] = indexed_scores[medcpt_index]

    development_results: list[dict] = []
    rankings_by_alpha: dict[float, dict[str, list[str]]] = {}
    for alpha in args.alphas:
        rankings = rankings_for_alpha(
            questions, bm25_matrix, medcpt_matrix, case_ids, alpha, args.top_k
        )
        rankings_by_alpha[alpha] = rankings
        metrics = subset_metrics(questions, rankings, development_qids)
        development_results.append({"alpha": alpha, "split": "development", **metrics})

    selected = max(
        development_results,
        key=lambda row: (
            row[args.selection_metric],
            row["hit@1"],
            row["hit@5"],
            -abs(row["alpha"] - 0.5),
        ),
    )
    selected_alpha = float(selected["alpha"])
    test_metrics = subset_metrics(questions, rankings_by_alpha[selected_alpha], test_qids)
    test_result = {"alpha": selected_alpha, "split": "test", **test_metrics}

    selected_score_details = {}
    for row_index, item in enumerate(questions):
        hybrid_scores = selected_alpha * minmax(medcpt_matrix[row_index]) + (
            1.0 - selected_alpha
        ) * minmax(bm25_matrix[row_index])
        ranked = hybrid_scores.argsort()[::-1][: args.top_k]
        selected_score_details[str(item["qid"])] = [
            {
                "rank": rank,
                "case_id": case_ids[int(case_index)],
                "hybrid_score": float(hybrid_scores[int(case_index)]),
                "bm25_score": float(bm25_matrix[row_index, int(case_index)]),
                "medcpt_score": float(medcpt_matrix[row_index, int(case_index)]),
            }
            for rank, case_index in enumerate(ranked, start=1)
        ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "hybrid_alpha_development_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(development_results[0]))
        writer.writeheader()
        writer.writerows(development_results)

    selected_rankings = rankings_by_alpha[selected_alpha]
    payload = {
        "method": "hybrid_bm25_medcpt",
        "selection_rule": f"maximize development {args.selection_metric}",
        "selected_alpha": selected_alpha,
        "development_sweep": development_results,
        "selected_development_metrics": selected,
        "held_out_test_metrics": test_result,
        "split_manifest": str(args.split),
        "qa": str(args.qa),
        "top_k": args.top_k,
        "selected_rankings": {
            qid: ranking for qid, ranking in selected_rankings.items() if qid in development_qids | test_qids
        },
        "selected_score_details": selected_score_details,
    }
    json_path = args.output_dir / "hybrid_alpha_selection.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_alpha": selected_alpha,
                "selected_development_metrics": selected,
                "held_out_test_metrics": test_result,
                "development_sweep_csv": str(csv_path),
                "selection_json": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
