from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.evidence_checker import check_evidence_support
from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _evidence_text(cases: list[dict]) -> str:
    blocks = []
    for case in cases:
        blocks.append(
            "\n".join(
                [
                    f"Case ID: {case.get('case_id', '')}",
                    f"Findings: {case.get('findings', '')}",
                    f"Impression: {case.get('impression', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evidence-checking agent over generated RAG answers.")
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--min-sentence-support", default=0.65, type=float)
    parser.add_argument(
        "--evidence-scope",
        choices=["top1", "all"],
        default="top1",
        help="Use only the top retrieved case or all retrieved cases as evidence for support checking.",
    )
    args = parser.parse_args()

    rows = _read_jsonl(args.generations)
    if not rows:
        raise ValueError(f"No generation rows found: {args.generations}")

    cases = load_cases_jsonl(args.cases)
    case_by_id = {case["case_id"]: case for case in cases}

    draft_f1_scores = []
    final_f1_scores = []
    top1_hits = []
    retrieved_hits = []
    support_rates = []
    revised_flags = []
    abstained_flags = []
    non_empty_flags = []
    unsupported_sentence_count = 0
    sentence_count = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in rows:
            raw_answer = row.get("answer", "")
            draft_answer = extract_final_answer(raw_answer)
            retrieved_ids = row.get("retrieved_case_ids", [])
            evidence_ids = retrieved_ids[:1] if args.evidence_scope == "top1" else retrieved_ids
            evidence_cases = [case_by_id[case_id] for case_id in evidence_ids if case_id in case_by_id]
            evidence_check = check_evidence_support(
                draft_answer,
                _evidence_text(evidence_cases),
                min_sentence_support=args.min_sentence_support,
            )
            final_answer = evidence_check.revised_answer
            revised = final_answer.strip() != draft_answer.strip()

            relevant = set(row.get("relevant_case_ids", []))
            top1_hits.append(float(bool(retrieved_ids) and retrieved_ids[0] in relevant))
            retrieved_hits.append(float(bool(relevant.intersection(retrieved_ids))))
            draft_f1_scores.append(token_f1(draft_answer, row.get("reference_answer", "")))
            final_f1_scores.append(token_f1(final_answer, row.get("reference_answer", "")))
            support_rates.append(evidence_check.support_rate)
            revised_flags.append(float(revised))
            abstained_flags.append(float(evidence_check.abstained))
            non_empty_flags.append(float(bool(final_answer.strip())))
            unsupported_sentence_count += len(evidence_check.unsupported_sentences)
            sentence_count += len(evidence_check.sentence_checks)

            output_record = dict(row)
            output_record["source_generation_answer"] = draft_answer
            output_record["raw_generation_answer"] = raw_answer
            output_record["answer"] = final_answer
            output_record["agent"] = {
                "min_sentence_support": args.min_sentence_support,
                "evidence_scope": args.evidence_scope,
                "evidence_case_ids": evidence_ids,
                "supported_sentences": evidence_check.supported_sentences,
                "unsupported_sentences": evidence_check.unsupported_sentences,
                "sentence_checks": [asdict(check) for check in evidence_check.sentence_checks],
                "support_rate": evidence_check.support_rate,
                "revised": revised,
                "abstained": evidence_check.abstained,
            }
            file.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    metrics = {
        "draft_answer_token_f1": sum(draft_f1_scores) / len(draft_f1_scores),
        "final_answer_token_f1": sum(final_f1_scores) / len(final_f1_scores),
        "top1_case_accuracy": sum(top1_hits) / len(top1_hits),
        "retrieved_case_hit_rate": sum(retrieved_hits) / len(retrieved_hits),
        "average_evidence_support_rate": sum(support_rates) / len(support_rates),
        "revision_rate": sum(revised_flags) / len(revised_flags),
        "abstention_rate": sum(abstained_flags) / len(abstained_flags),
        "unsupported_sentence_rate": (
            unsupported_sentence_count / sentence_count if sentence_count else 0.0
        ),
        "non_empty_answer_rate": sum(non_empty_flags) / len(non_empty_flags),
    }

    output = {
        "generations": str(args.generations),
        "cases": str(args.cases),
        "output": str(args.output),
        "record_count": len(rows),
        "min_sentence_support": args.min_sentence_support,
        "evidence_scope": args.evidence_scope,
        "metrics": metrics,
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
