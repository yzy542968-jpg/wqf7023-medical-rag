from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.metrics import evaluate_retrieval
from medical_rag.retrieval.medcpt_retriever import MedCPTRetriever, encode_queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MedCPT dense retrieval on case-grounded QA seed questions.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    qa_payload = json.loads(args.qa.read_text(encoding="utf-8"))
    retriever = MedCPTRetriever.from_index(args.cases, args.index)

    qrels: dict[str, set[str]] = {}
    rankings: dict[str, list[str]] = {}
    detailed_rankings = {}
    questions = qa_payload["questions"]
    query_embeddings = encode_queries(
        [item["question"] for item in questions],
        batch_size=args.batch_size,
        device=args.device,
    )

    for query_index, item in enumerate(questions):
        qid = item["qid"]
        qrels[qid] = set(item["relevant_case_ids"])
        scores = retriever.embeddings @ query_embeddings[query_index]
        ranked_indices = scores.argsort()[::-1][: args.top_k]
        results = []
        for rank, index in enumerate(ranked_indices, start=1):
            case = retriever.cases[int(index)]
            results.append(
                {
                    "rank": rank,
                    "case_id": case["case_id"],
                    "score": float(scores[int(index)]),
                    "findings": case.get("findings", ""),
                    "impression": case.get("impression", ""),
                    "images": case.get("images", []),
                }
            )
        rankings[qid] = [result["case_id"] for result in results]
        detailed_rankings[qid] = {
            "case_id": item["case_id"],
            "question_type": item["question_type"],
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "top_results": results,
        }

    metrics = evaluate_retrieval(qrels, rankings, k_values=(1, 3, 5, 10, 20))
    output = {
        "retriever": "MedCPT dense retrieval",
        "cases": str(args.cases),
        "index": str(args.index),
        "qa": str(args.qa),
        "query_count": len(qrels),
        "top_k": args.top_k,
        "metrics": metrics,
        "rankings": detailed_rankings,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"query_count": len(qrels), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
