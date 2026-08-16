from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def _matches_problem_terms(case: dict[str, Any], terms: list[str]) -> bool:
    problems = (case.get("problems") or "").lower()
    return any(term.lower() in problems for term in terms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build keyword qrels from OpenI Problems labels.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    queries = json.loads(args.queries.read_text(encoding="utf-8"))

    qrels = {}
    for query in queries:
        qid = query["qid"]
        terms = query["problem_terms"]
        relevant = [
            case["case_id"]
            for case in cases
            if _matches_problem_terms(case, terms)
        ]
        qrels[qid] = {
            "query": query["query"],
            "problem_terms": terms,
            "relevant_case_ids": relevant,
            "relevant_count": len(relevant),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(qrels, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote qrels for {len(qrels)} queries to {args.output}")


if __name__ == "__main__":
    main()

