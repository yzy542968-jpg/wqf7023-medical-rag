from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def case_context(case: dict) -> str:
    return "\n".join(
        [
            f"Case ID: {case['case_id']}",
            f"Findings: {case.get('findings', '')}",
            f"Impression: {case.get('impression', '')}",
        ]
    )


def direct_prompt(question: str, context: str) -> str:
    return "\n".join(
        [
            "Answer the medical question using the selected radiology case.",
            "Question:",
            question,
            "",
            "Selected radiology case:",
            context,
            "",
            "Answer clearly and concisely.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a diagnostic oracle-retrieval prompt pack for the frozen test split."
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_qa_seed_clean.json",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_cases.jsonl",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "openi_qa_grouped_case_seed7023.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "prompt_packs"
        / "final_optimized"
        / "oracle_test_direct.jsonl",
    )
    args = parser.parse_args()

    questions = json.loads(args.qa.read_text(encoding="utf-8"))["questions"]
    question_by_qid = {str(item["qid"]): item for item in questions}
    cases = {str(item["case_id"]): item for item in read_jsonl(args.cases)}
    split = json.loads(args.split.read_text(encoding="utf-8"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for qid in split["test"]["qids"]:
            item = question_by_qid[str(qid)]
            target_case = cases[str(item["case_id"])]
            context = case_context(target_case)
            row = {
                "qid": item["qid"],
                "case_id": item["case_id"],
                "question_type": item["question_type"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "relevant_case_ids": item["relevant_case_ids"],
                "system": "oracle_case_rag",
                "prompt_mode": "direct",
                "retriever": "oracle_target_case_diagnostic",
                "retrieved_case_ids": [item["case_id"]],
                "retrieved_context": context,
                "prompt": direct_prompt(item["question"], context),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(split["test"]["qids"])}, indent=2))


if __name__ == "__main__":
    main()
