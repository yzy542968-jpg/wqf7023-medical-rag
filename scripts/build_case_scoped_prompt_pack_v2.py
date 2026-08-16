from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.case_scoped_benchmark import expected_section
from medical_rag.retrieval.scoped_chunk_retriever import ScopedBM25ChunkRetriever


def build_prompt(question: dict, evidence: list[dict]) -> str:
    evidence_text = "\n".join(
        f"[{row['section']} {row['position']}] {row['text']}" for row in evidence
    )
    return "\n".join(
        [
            "Answer the question using only the retrieved evidence from the specified radiology case.",
            "Do not add findings that are absent from the evidence.",
            f"Case scope: {question['scope_case_id']}",
            f"Question: {question['question']}",
            "",
            "Retrieved evidence:",
            evidence_text,
            "",
            "Answer clearly and concisely.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the locked case-scoped v2 generation prompt pack.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_scoped_benchmark_v2.json",
    )
    parser.add_argument(
        "--top-k-selection",
        type=Path,
        default=ROOT / "experiments" / "benchmark_v2" / "calibration" / "locked_top_k.json",
    )
    parser.add_argument(
        "--split",
        choices=["development", "calibration", "test", "confirmation"],
        default="test",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "prompt_packs" / "benchmark_v2" / "test_case_scoped_routed.jsonl",
    )
    args = parser.parse_args()

    payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
    selection = json.loads(args.top_k_selection.read_text(encoding="utf-8"))
    top_k = int(selection["selected_top_k"])
    split_qids = set(payload["split"][args.split]["qids"])
    questions = [row for row in payload["questions"] if row["qid"] in split_qids]
    retriever = ScopedBM25ChunkRetriever().fit(payload["chunks"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    recalls = []
    with args.output.open("w", encoding="utf-8") as handle:
        for question in questions:
            evidence = retriever.search(
                question["question"],
                top_k=top_k,
                case_id=question["scope_case_id"],
                allowed_sections={expected_section(question["question_type"])},
            )
            retrieved_ids = [row["chunk_id"] for row in evidence]
            relevant = set(question["relevant_chunk_ids"])
            recall = len(set(retrieved_ids) & relevant) / len(relevant)
            recalls.append(recall)
            row = {
                **question,
                "system": "v2_case_scoped_agent_routed_rag",
                "prompt_mode": "direct_evidence_only",
                "retriever": "case_scoped_agent_routed_bm25",
                "top_k": top_k,
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_case_ids": [item["case_id"] for item in evidence],
                "retrieved_sections": [item["section"] for item in evidence],
                "retrieval_recall": recall,
                "retrieved_context": "\n".join(item["text"] for item in evidence),
                "relevant_case_ids": [question["case_id"]],
                "prompt": build_prompt(question, evidence),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "split": args.split,
                "records": len(questions),
                "locked_top_k": top_k,
                "mean_evidence_recall": sum(recalls) / len(recalls),
                "complete_evidence_coverage_rate": sum(value == 1 for value in recalls) / len(recalls),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
