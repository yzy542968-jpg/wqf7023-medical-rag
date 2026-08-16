from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.metrics import evaluate_retrieval
from medical_rag.retrieval.tfidf_retriever import TfidfRetriever, load_cases_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TF-IDF retrieval on keyword qrels.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    qrels_payload = json.loads(args.qrels.read_text(encoding="utf-8"))
    retriever = TfidfRetriever().fit(cases)

    rankings: dict[str, list[str]] = {}
    detailed_rankings = {}
    qrels: dict[str, set[str]] = {}

    for qid, item in qrels_payload.items():
        qrels[qid] = set(item["relevant_case_ids"])
        results = retriever.search(item["query"], top_k=args.top_k)
        rankings[qid] = [result["case_id"] for result in results]
        detailed_rankings[qid] = {
            "query": item["query"],
            "relevant_count": item["relevant_count"],
            "top_results": results,
        }

    metrics = evaluate_retrieval(qrels, rankings, k_values=(1, 3, 5, 10, 20))
    output = {
        "retriever": "standard-library TF-IDF baseline",
        "cases": str(args.cases),
        "qrels": str(args.qrels),
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

