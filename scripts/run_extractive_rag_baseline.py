from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.answer_metrics import token_f1
from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.tfidf_retriever import TfidfRetriever, load_cases_jsonl


def _make_retriever(name: str):
    if name == "tfidf":
        return TfidfRetriever()
    if name == "bm25":
        return BM25Retriever()
    raise ValueError(f"Unsupported retriever: {name}")


def _answer_from_case(question_type: str, case: dict) -> str:
    findings = case.get("findings", "")
    impression = case.get("impression", "")
    if question_type == "findings_from_indication":
        return findings
    if question_type == "impression_from_indication":
        return impression
    if impression:
        return impression
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run extractive report-RAG baseline on QA seed.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--retriever", choices=["tfidf", "bm25"], required=True)
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    case_by_id = {case["case_id"]: case for case in cases}
    qa_payload = json.loads(args.qa.read_text(encoding="utf-8"))
    retriever = _make_retriever(args.retriever).fit(cases)

    answers = []
    f1_scores = []
    top1_hits = []
    supported_by_retrieved_case = []

    for item in qa_payload["questions"]:
        results = retriever.search(item["question"], top_k=args.top_k)
        top_case_id = results[0]["case_id"] if results else ""
        top_case = case_by_id[top_case_id]
        prediction = _answer_from_case(item["question_type"], top_case)
        score = token_f1(prediction, item["reference_answer"])
        is_top1_hit = top_case_id in set(item["relevant_case_ids"])

        f1_scores.append(score)
        top1_hits.append(float(is_top1_hit))
        supported_by_retrieved_case.append(float(bool(prediction)))
        answers.append(
            {
                "qid": item["qid"],
                "question": item["question"],
                "question_type": item["question_type"],
                "reference_case_id": item["case_id"],
                "retrieved_case_id": top_case_id,
                "top1_hit": is_top1_hit,
                "prediction": prediction,
                "reference_answer": item["reference_answer"],
                "token_f1": score,
                "retrieval_results": results,
            }
        )

    metrics = {
        "answer_token_f1": sum(f1_scores) / len(f1_scores),
        "top1_case_accuracy": sum(top1_hits) / len(top1_hits),
        "non_empty_answer_rate": sum(supported_by_retrieved_case) / len(supported_by_retrieved_case),
    }
    output = {
        "system": f"extractive report-RAG with {args.retriever}",
        "cases": str(args.cases),
        "qa": str(args.qa),
        "question_count": len(answers),
        "metrics": metrics,
        "answers": answers,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

