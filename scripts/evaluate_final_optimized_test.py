from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.semantic_evidence_checker import MedicalNLIPredictor, check_semantic_evidence_support
from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


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
    parser = argparse.ArgumentParser(description="Evaluate the locked final optimized P2 system.")
    parser.add_argument(
        "--generations",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "generations"
        / "adaptive_test_direct_qwen15.jsonl",
    )
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
        "--system-name", default="final_adaptive_direct_semantic_agent"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "final_optimized" / "final_test",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.generations)
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

    output_rows = []
    draft_f1s = []
    verified_f1s = []
    support_rates = []
    abstentions = []
    revisions = []
    retrieval_abstentions = []
    contradiction_count = 0
    for row in rows:
        retrieved_ids = [str(value) for value in row.get("retrieved_case_ids", [])]
        selected_case = case_by_id.get(retrieved_ids[0]) if retrieved_ids else None
        draft = extract_final_answer(row.get("answer", ""))
        check = check_semantic_evidence_support(
            draft,
            evidence_text(selected_case),
            predictor,
            min_combined_support=float(config["support_threshold"]),
            entailment_threshold=float(config["entailment_threshold"]),
            contradiction_threshold=float(config["contradiction_threshold"]),
            lexical_weight=float(config["lexical_weight"]),
        )
        reference = str(row.get("reference_answer", ""))
        draft_f1s.append(token_f1(draft, reference))
        verified_f1s.append(token_f1(check.revised_answer, reference))
        support_rates.append(check.support_rate)
        abstentions.append(float(check.abstained))
        revisions.append(float(check.revised_answer.strip() != draft.strip()))
        retrieval_abstentions.append(float(not retrieved_ids))
        contradiction_count += sum(
            value.contradiction_probability >= float(config["contradiction_threshold"])
            for value in check.sentence_checks
        )
        output_rows.append(
            {
                "system": args.system_name,
                "qid": row["qid"],
                "case_id": row["case_id"],
                "question_type": row.get("question_type"),
                "question": row.get("question"),
                "reference_answer": reference,
                "retrieved_case_ids": retrieved_ids,
                "draft_answer": draft,
                "final_answer": check.revised_answer,
                "draft_token_f1": draft_f1s[-1],
                "final_token_f1": verified_f1s[-1],
                "support_rate": check.support_rate,
                "retrieval_abstained": not retrieved_ids,
                "agent_abstained": check.abstained,
                "revised": bool(revisions[-1]),
                "sentence_checks": [asdict(value) for value in check.sentence_checks],
            }
        )

    summary = {
        "system": args.system_name,
        "n": len(rows),
        "draft_token_f1": mean(draft_f1s),
        "verified_token_f1": mean(verified_f1s),
        "evidence_support_rate": mean(support_rates),
        "retrieval_abstention_rate": mean(retrieval_abstentions),
        "final_abstention_rate": mean(abstentions),
        "revision_rate": mean(revisions),
        "nli_contradiction_count": contradiction_count,
        "semantic_agent_config": config,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "final_optimized_test_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["rows_path"] = str(rows_path)
    summary_path = args.output_dir / "final_optimized_test_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
