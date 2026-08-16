from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.semantic_evidence_checker import MedicalNLIPredictor, check_semantic_evidence_support
from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


PROMPT_OUTPUTS = {
    "direct": ROOT
    / "experiments"
    / "final_optimized"
    / "generations"
    / "adaptive_development_direct_qwen15.jsonl",
    "evidence_guided": ROOT
    / "experiments"
    / "final_optimized"
    / "generations"
    / "adaptive_development_evidence_guided_qwen15.jsonl",
    "structured_case_aware": ROOT
    / "experiments"
    / "final_optimized"
    / "generations"
    / "adaptive_development_structured_case_aware_qwen15.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evidence_text(case: dict | None) -> str:
    if not case:
        return ""
    return "\n".join(
        [
            f"Case ID: {case['case_id']}",
            f"Findings: {case.get('findings', '')}",
            f"Impression: {case.get('impression', '')}",
        ]
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final P2 prompt ablation on development cases.")
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "data" / "processed" / "openi_cases.jsonl"
    )
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "semantic_agent"
        / "semantic_agent_selection.json",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "final_optimized" / "prompt_ablation",
    )
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    case_by_id = {str(case["case_id"]): case for case in cases}
    semantic_selection = json.loads(args.semantic_config.read_text(encoding="utf-8"))
    config = semantic_selection["selected_config"]
    predictor = MedicalNLIPredictor(
        semantic_selection["nli_model"],
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=True,
    )

    summaries = []
    checked_rows = []
    for prompt_mode, path in PROMPT_OUTPUTS.items():
        rows = read_jsonl(path)
        draft_f1s = []
        final_f1s = []
        support_rates = []
        abstentions = []
        revisions = []
        draft_words = []
        final_words = []
        contradiction_count = 0
        for row in rows:
            retrieved_ids = [str(value) for value in row.get("retrieved_case_ids", [])]
            case = case_by_id.get(retrieved_ids[0]) if retrieved_ids else None
            draft = extract_final_answer(row.get("answer", ""))
            check = check_semantic_evidence_support(
                draft,
                evidence_text(case),
                predictor,
                min_combined_support=float(config["support_threshold"]),
                entailment_threshold=float(config["entailment_threshold"]),
                contradiction_threshold=float(config["contradiction_threshold"]),
                lexical_weight=float(config["lexical_weight"]),
            )
            reference = str(row.get("reference_answer", ""))
            draft_f1s.append(token_f1(draft, reference))
            final_f1s.append(token_f1(check.revised_answer, reference))
            support_rates.append(check.support_rate)
            abstentions.append(float(check.abstained))
            revisions.append(float(check.revised_answer.strip() != draft.strip()))
            draft_words.append(len(draft.split()))
            final_words.append(len(check.revised_answer.split()))
            contradiction_count += sum(
                sentence_check.contradiction_probability
                >= float(config["contradiction_threshold"])
                for sentence_check in check.sentence_checks
            )
            checked_rows.append(
                {
                    "prompt_mode": prompt_mode,
                    "qid": row["qid"],
                    "case_id": row["case_id"],
                    "question_type": row.get("question_type"),
                    "reference_answer": reference,
                    "retrieved_case_ids": retrieved_ids,
                    "draft_answer": draft,
                    "final_answer": check.revised_answer,
                    "draft_token_f1": draft_f1s[-1],
                    "final_token_f1": final_f1s[-1],
                    "support_rate": check.support_rate,
                    "abstained": check.abstained,
                    "revised": bool(revisions[-1]),
                    "sentence_checks": [asdict(value) for value in check.sentence_checks],
                }
            )
        summaries.append(
            {
                "prompt_mode": prompt_mode,
                "n": len(rows),
                "draft_token_f1": mean(draft_f1s),
                "verified_token_f1": mean(final_f1s),
                "evidence_support_rate": mean(support_rates),
                "abstention_rate": mean(abstentions),
                "revision_rate": mean(revisions),
                "mean_draft_words": mean(draft_words),
                "mean_final_words": mean(final_words),
                "nli_contradiction_count": contradiction_count,
            }
        )

    selected = max(
        summaries,
        key=lambda row: (
            row["verified_token_f1"],
            row["evidence_support_rate"],
            -row["abstention_rate"],
            -row["mean_final_words"],
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "development_prompt_ablation.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    rows_path = args.output_dir / "development_prompt_checked_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in checked_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output = {
        "selection_rule": "maximize development verified Token-F1",
        "semantic_agent_config": config,
        "summaries": summaries,
        "selected_prompt_mode": selected["prompt_mode"],
        "selected_development_summary": selected,
        "summary_csv": str(summary_path),
        "checked_rows": str(rows_path),
    }
    output_path = args.output_dir / "prompt_selection.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
