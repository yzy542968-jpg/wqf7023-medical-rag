from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.case_scoped_benchmark import (
    benchmark_summary,
    build_case_scoped_benchmark,
    dumps_json,
)
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the unused-case OpenI case-scoped QA benchmark v2.")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "processed" / "openi_cases.jsonl")
    parser.add_argument(
        "--prior-qa",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_qa_seed_clean.json",
    )
    parser.add_argument("--max-cases", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7023)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_scoped_benchmark_v2.json",
    )
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    prior_questions = json.loads(args.prior_qa.read_text(encoding="utf-8"))["questions"]
    excluded = {row["case_id"] for row in prior_questions}
    payload = build_case_scoped_benchmark(cases, excluded, args.max_cases, args.seed)
    payload["source_cases"] = str(args.cases)
    payload["excluded_prior_qa"] = str(args.prior_qa)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(dumps_json(payload), encoding="utf-8")
    print(dumps_json({"output": str(args.output), **benchmark_summary(payload)}))


if __name__ == "__main__":
    main()
