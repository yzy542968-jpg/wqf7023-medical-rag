from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.evidence_checker import check_evidence_support
from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


DEFAULT_SYSTEMS = {
    "report_bm25": ROOT / "experiments" / "generations_report_rag_bm25_qwen15_full360.jsonl",
    "case_bm25_top1": ROOT / "experiments" / "generations_case_rag_bm25_top1_qwen15_full360.jsonl",
    "case_hybrid_top1": ROOT / "experiments" / "generations_case_rag_hybrid_top1_qwen15_full360.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evidence_text(cases: list[dict]) -> str:
    return "\n\n".join(
        "\n".join(
            [
                f"Case ID: {case.get('case_id', '')}",
                f"Findings: {case.get('findings', '')}",
                f"Impression: {case.get('impression', '')}",
            ]
        )
        for case in cases
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate(
    rows: list[dict],
    case_by_id: dict[str, dict],
    threshold: float,
    evidence_scope: str,
) -> tuple[dict, list[dict]]:
    draft_f1: list[float] = []
    final_f1: list[float] = []
    support_rates: list[float] = []
    revised: list[float] = []
    abstained: list[float] = []
    top1_hits: list[float] = []
    retrieved_hits: list[float] = []
    sentence_count = 0
    unsupported_count = 0
    negation_conflicts = 0
    checked_rows: list[dict] = []

    for row in rows:
        draft = extract_final_answer(row.get("answer", ""))
        retrieved_ids = [str(value) for value in row.get("retrieved_case_ids", [])]
        selected_ids = retrieved_ids[:1] if evidence_scope == "top1" else retrieved_ids
        selected_cases = [case_by_id[value] for value in selected_ids if value in case_by_id]
        check = check_evidence_support(
            draft,
            evidence_text(selected_cases),
            min_sentence_support=threshold,
        )
        final = check.revised_answer
        reference = row.get("reference_answer", "")
        relevant = {str(value) for value in row.get("relevant_case_ids", [])}

        draft_f1.append(token_f1(draft, reference))
        final_f1.append(token_f1(final, reference))
        support_rates.append(check.support_rate)
        revised.append(float(final.strip() != draft.strip()))
        abstained.append(float(check.abstained))
        top1_hits.append(float(bool(retrieved_ids) and retrieved_ids[0] in relevant))
        retrieved_hits.append(float(bool(relevant.intersection(retrieved_ids))))
        sentence_count += len(check.sentence_checks)
        unsupported_count += len(check.unsupported_sentences)
        negation_conflicts += sum(
            1 for sentence_check in check.sentence_checks if not sentence_check.negation_consistent
        )

        checked_rows.append(
            {
                "qid": row.get("qid"),
                "case_id": row.get("case_id"),
                "question_type": row.get("question_type"),
                "question": row.get("question"),
                "reference_answer": reference,
                "retrieved_case_ids": retrieved_ids,
                "evidence_case_ids": selected_ids,
                "draft_answer": draft,
                "final_answer": final,
                "draft_token_f1": draft_f1[-1],
                "final_token_f1": final_f1[-1],
                "support_rate": check.support_rate,
                "revised": bool(revised[-1]),
                "abstained": check.abstained,
                "sentence_checks": [asdict(value) for value in check.sentence_checks],
            }
        )

    summary = {
        "n": len(rows),
        "threshold": threshold,
        "evidence_scope": evidence_scope,
        "draft_token_f1": mean(draft_f1),
        "final_token_f1": mean(final_f1),
        "top1_case_accuracy": mean(top1_hits),
        "retrieved_case_hit_rate": mean(retrieved_hits),
        "evidence_support_rate": mean(support_rates),
        "revision_rate": mean(revised),
        "abstention_rate": mean(abstained),
        "unsupported_sentence_rate": unsupported_count / sentence_count if sentence_count else 0.0,
        "negation_conflict_count": negation_conflicts,
        "sentence_count": sentence_count,
    }
    return summary, checked_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep evidence-checking thresholds on P2 outputs.")
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_cases.jsonl",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.40, 0.50, 0.60, 0.65, 0.70],
    )
    parser.add_argument("--evidence-scope", choices=["top1", "all"], default="top1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "final_p2",
    )
    parser.add_argument(
        "--export-threshold",
        type=float,
        default=0.50,
        help="Write per-row checked outputs for this threshold.",
    )
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    case_by_id = {str(case["case_id"]): case for case in cases}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for system_name, path in DEFAULT_SYSTEMS.items():
        rows = read_jsonl(path)
        for threshold in args.thresholds:
            summary, checked_rows = evaluate(rows, case_by_id, threshold, args.evidence_scope)
            summary["system"] = system_name
            summaries.append(summary)
            if abs(threshold - args.export_threshold) < 1e-9:
                output_path = args.output_dir / f"{system_name}_checked_t{threshold:.2f}.jsonl"
                with output_path.open("w", encoding="utf-8") as handle:
                    for checked_row in checked_rows:
                        handle.write(json.dumps(checked_row, ensure_ascii=False) + "\n")

    json_path = args.output_dir / "evidence_threshold_sweep.json"
    json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = args.output_dir / "evidence_threshold_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(json.dumps({"json": str(json_path), "csv": str(csv_path), "rows": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
