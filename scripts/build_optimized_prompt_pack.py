from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def context_for_case(case: dict | None) -> str:
    if not case:
        return "No sufficiently confident case was retrieved."
    image_names = ", ".join(
        str(image.get("filename", "")) for image in case.get("images", [])
    )
    return "\n".join(
        [
            f"Case ID: {case['case_id']}",
            f"Linked images: {image_names}",
            f"Findings: {case.get('findings', '')}",
            f"Impression: {case.get('impression', '')}",
        ]
    )


def build_prompt(question: str, context: str, prompt_mode: str) -> str:
    common = [
        "Question:",
        question,
        "",
        "Selected radiology case:",
        context,
        "",
    ]
    if prompt_mode == "direct":
        return "\n".join(
            [
                "Answer the medical question using the selected radiology case.",
                *common,
                "Answer clearly and concisely.",
            ]
        )
    if prompt_mode == "evidence_guided":
        return "\n".join(
            [
                "Answer the medical question using only the selected radiology case evidence.",
                "Do not add unsupported findings, diagnoses, locations, or severity.",
                "If the evidence is insufficient, state that it is insufficient.",
                *common,
                "Return only one concise answer paragraph.",
            ]
        )
    if prompt_mode == "structured_case_aware":
        return "\n".join(
            [
                "Answer this case-grounded research question using only the selected evidence.",
                "Do not combine facts from other patients or add outside clinical knowledge.",
                "If the evidence is insufficient, state that it is insufficient.",
                *common,
                "Respond in this structure:",
                "Evidence: one short statement with the selected Case ID.",
                "Final answer: one concise paragraph supported by that case.",
            ]
        )
    raise ValueError(f"unsupported prompt mode: {prompt_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final adaptive retrieval prompt packs.")
    parser.add_argument(
        "--qa",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_qa_seed_clean.json",
    )
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "data" / "processed" / "openi_cases.jsonl"
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "openi_qa_grouped_case_seed7023.json",
    )
    parser.add_argument("--split-name", choices=["development", "test"], required=True)
    parser.add_argument(
        "--decisions",
        type=Path,
        required=True,
        help="Adaptive retrieval decision JSONL for the selected split.",
    )
    parser.add_argument(
        "--prompt-modes",
        nargs="+",
        choices=["direct", "evidence_guided", "structured_case_aware"],
        default=["direct", "evidence_guided", "structured_case_aware"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "processed" / "prompt_packs" / "final_optimized",
    )
    args = parser.parse_args()

    questions = json.loads(args.qa.read_text(encoding="utf-8"))["questions"]
    question_by_qid = {str(item["qid"]): item for item in questions}
    cases = read_jsonl(args.cases)
    case_by_id = {str(case["case_id"]): case for case in cases}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    expected_qids = set(split[args.split_name]["qids"])
    decisions = {str(row["qid"]): row for row in read_jsonl(args.decisions)}
    if set(decisions) != expected_qids:
        raise ValueError("decision QIDs do not exactly match the requested split")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for prompt_mode in args.prompt_modes:
        output_path = args.output_dir / f"adaptive_{args.split_name}_{prompt_mode}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for qid in split[args.split_name]["qids"]:
                item = question_by_qid[qid]
                decision = decisions[qid]
                selected_case_id = decision.get("selected_case_id")
                selected_case = case_by_id.get(str(selected_case_id)) if selected_case_id else None
                context = context_for_case(selected_case)
                record = {
                    "qid": qid,
                    "case_id": item["case_id"],
                    "question_type": item["question_type"],
                    "question": item["question"],
                    "reference_answer": item["reference_answer"],
                    "relevant_case_ids": item["relevant_case_ids"],
                    "system": "adaptive_case_rag",
                    "prompt_mode": prompt_mode,
                    "retriever": "adaptive_hybrid_medcpt_reranker",
                    "retrieval_abstained": bool(decision["abstained"]),
                    "retrieval_source": decision["source"],
                    "retrieval_reason": decision["reason"],
                    "retrieved_case_ids": [selected_case_id] if selected_case_id else [],
                    "evidence_case_ids": [selected_case_id] if selected_case_id else [],
                    "retrieved_context": context,
                    "prompt": build_prompt(item["question"], context, prompt_mode),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        outputs[prompt_mode] = str(output_path)

    print(json.dumps({"split": args.split_name, "records": len(expected_qids), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
