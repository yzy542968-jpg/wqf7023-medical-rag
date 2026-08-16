from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.case_scoped_benchmark import (
    build_case_chunks,
    build_case_questions,
    clean_text,
    content_fingerprint,
    is_clean_eligible_case,
)
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a once-only unused-case confirmation set for benchmark v2.")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "processed" / "openi_cases.jsonl")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_scoped_benchmark_v2.json",
    )
    parser.add_argument(
        "--prior-qa",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_qa_seed_clean.json",
    )
    parser.add_argument("--case-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=17023)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_scoped_confirmation_v2.json",
    )
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    prior = json.loads(args.prior_qa.read_text(encoding="utf-8"))["questions"]
    excluded = {row["case_id"] for row in benchmark["questions"]}
    excluded.update(row["case_id"] for row in prior)
    eligible = [
        case
        for case in load_cases_jsonl(args.cases)
        if case["case_id"] not in excluded and is_clean_eligible_case(case)
    ]
    random.Random(args.seed).shuffle(eligible)
    selected = sorted(eligible[: args.case_count], key=lambda row: row["case_id"])
    if len(selected) != args.case_count:
        raise ValueError(f"Requested {args.case_count} confirmation cases, found {len(selected)}.")

    chunks = []
    questions = []
    for case in selected:
        case_chunks = build_case_chunks(case)
        chunks.extend(case_chunks)
        questions.extend(build_case_questions(case, case_chunks))
    case_ids = [row["case_id"] for row in selected]
    fingerprint = hashlib.sha256("\n".join(case_ids).encode("utf-8")).hexdigest()
    payload = {
        "benchmark": "OpenI case-scoped evidence QA v2 confirmation",
        "version": "2.0-confirmation",
        "seed": args.seed,
        "selection_status": "locked once-only confirmation cohort",
        "excluded_case_count": len(excluded),
        "available_unused_eligible_case_count": len(eligible),
        "case_count": len(selected),
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "case_id_fingerprint_sha256": fingerprint,
        "content_fingerprint_sha256": content_fingerprint(questions, chunks),
        "questions": questions,
        "chunks": chunks,
        "split": {
            "confirmation": {
                "case_count": len(selected),
                "question_count": len(questions),
                "chunk_count": len(chunks),
                "case_ids": case_ids,
                "qids": [row["qid"] for row in questions],
                "chunk_ids": [row["chunk_id"] for row in chunks],
            }
        },
        "source_cases": str(args.cases),
        "excluded_benchmark": str(args.benchmark),
        "excluded_prior_qa": str(args.prior_qa),
        "mean_reference_characters": sum(len(clean_text(row["reference_answer"])) for row in questions)
        / len(questions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "case_count": len(selected),
                "question_count": len(questions),
                "chunk_count": len(chunks),
                "excluded_case_count": len(excluded),
                "remaining_unused_eligible_after_selection": len(eligible) - len(selected),
                "fingerprint": fingerprint,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
