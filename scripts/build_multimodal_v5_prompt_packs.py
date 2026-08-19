from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.planner import plan_question


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def context_for_case(case: dict | None, plan) -> str:
    if not case:
        return "No sufficiently confident report case was retrieved."
    lines = [f"Case ID: {case['case_id']}"]
    if plan.answer_field == "findings":
        lines.append(f"Findings: {case.get('findings', '')}")
    elif plan.answer_field == "impression":
        lines.append(f"Impression: {case.get('impression', '')}")
    else:
        lines.extend([
            f"Findings: {case.get('findings', '')}",
            f"Impression: {case.get('impression', '')}",
        ])
    return "\n".join(lines)


def build_prompt(question: str, context: str) -> str:
    return "\n".join([
        "Answer the medical question using only the selected radiology report evidence.",
        "Do not add unsupported findings, diagnoses, locations, or severity.",
        "If the evidence is insufficient, state that it is insufficient.",
        "Question:",
        question,
        "",
        "Selected report evidence:",
        context,
        "",
        "Answer clearly and concisely.",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build non-oracle V5 Qwen prompt packs.")
    parser.add_argument("--cohort", type=Path, default=ROOT / "data" / "processed" / "openi_multimodal_v5_cohort.json")
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "processed" / "openi_cases.jsonl")
    parser.add_argument("--retrieval-dir", type=Path, default=ROOT / "experiments" / "post_submission_v5")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed" / "prompt_packs" / "multimodal_v5")
    parser.add_argument("--split", choices=["development", "confirmation"], default="confirmation")
    args = parser.parse_args()

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    question_by_qid = {
        str(row["qid"]): row for row in cohort["questions"]
    }
    case_by_id = {
        str(row["case_id"]): row
        for row in read_jsonl(args.cases)
    }
    allowed_qids = set(cohort["split"][args.split]["qids"])
    retrieval_rows = read_jsonl(args.retrieval_dir / f"{args.split}_retrieval_rows.jsonl")
    grouped: dict[str, dict[str, dict]] = {}
    for row in retrieval_rows:
        if str(row["qid"]) not in allowed_qids:
            continue
        grouped.setdefault(str(row["system"]), {})[str(row["qid"])] = row

    systems = {
        "indication_question_bm25": "v5_report_only",
        "indication_question_correct_image": "v5_multimodal",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for retrieval_system, output_system in systems.items():
        output_path = args.output_dir / f"{args.split}_{output_system}.jsonl"
        rows = grouped.get(retrieval_system, {})
        with output_path.open("w", encoding="utf-8") as handle:
            for qid in cohort["split"][args.split]["qids"]:
                row = rows[str(qid)]
                question = str(question_by_qid[str(qid)]["question"])
                plan = plan_question(question)
                selected_id = str(row["selected_case_id"]) if row.get("selected_case_id") else None
                context = context_for_case(case_by_id.get(selected_id), plan)
                record = {
                    "qid": qid,
                    "case_id": str(row["case_id"]),
                    "question_type": row.get("question_type"),
                    "question": question,
                    "reference_answer": row["reference_answer"],
                    "relevant_case_ids": [str(row["case_id"])],
                    "system": output_system,
                    "prompt_mode": "direct_non_oracle_planner",
                    "retriever": retrieval_system,
                    "planner": asdict(plan),
                    "retrieved_case_ids": [selected_id] if selected_id else [],
                    "retrieved_context": context,
                    "prompt": build_prompt(question, context),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        outputs[output_system] = str(output_path)
    print(json.dumps({"split": args.split, "records": len(allowed_qids), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
