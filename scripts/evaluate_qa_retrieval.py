from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.metrics import evaluate_retrieval
from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.tfidf_retriever import TfidfRetriever, load_cases_jsonl


def _make_retriever(name: str):
    if name == "tfidf":
        return TfidfRetriever()
    if name == "bm25":
        return BM25Retriever()
    raise ValueError(f"Unsupported retriever: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on case-grounded QA seed questions.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--retriever", choices=["tfidf", "bm25"], required=True)
    parser.add_argument("--top-k", default=20, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    qa_payload = json.loads(args.qa.read_text(encoding="utf-8"))
    retriever = _make_retriever(args.retriever).fit(cases)

    qrels: dict[str, set[str]] = {}
    rankings: dict[str, list[str]] = {}
    detailed_rankings = {}

    for item in qa_payload["questions"]:
        qid = item["qid"]
        qrels[qid] = set(item["relevant_case_ids"])
        results = retriever.search(item["question"], top_k=args.top_k)
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
        "retriever": args.retriever,
        "cases": str(args.cases),
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

