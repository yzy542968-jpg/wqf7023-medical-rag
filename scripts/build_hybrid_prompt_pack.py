from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.medcpt_retriever import MedCPTRetriever, encode_queries
from medical_rag.retrieval.tfidf_retriever import _tokens, load_cases_jsonl


def _minmax(values: np.ndarray) -> np.ndarray:
    lower = float(values.min())
    upper = float(values.max())
    if upper == lower:
        return np.zeros_like(values)
    return (values - lower) / (upper - lower)


def _case_context(results: list[dict[str, Any]]) -> str:
    blocks = []
    for result in results:
        image_list = ", ".join(
            f"{image.get('projection', '')}: {image.get('filename', '')}"
            for image in result.get("images", [])
        )
        blocks.append(
            "\n".join(
                [
                    f"Case ID: {result['case_id']}",
                    f"Images: {image_list}",
                    f"Findings: {result.get('findings', '')}",
                    f"Impression: {result.get('impression', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _structured_prompt(question: str, context: str) -> str:
    return "\n".join(
        [
            "You are answering a medical question for research purposes.",
            "Use only the provided radiology report evidence.",
            "Do not use outside clinical knowledge unless it is explicitly supported by the evidence.",
            "",
            "Question:",
            question,
            "",
            "Retrieved case evidence:",
            context,
            "",
            "Respond in this structure:",
            "Evidence:",
            "- List the relevant evidence from the retrieved report.",
            "",
            "Reasoning:",
            "- Briefly connect the evidence to the answer.",
            "",
            "Final answer:",
            "- Provide the answer in one concise paragraph.",
        ]
    )


def _top1_structured_prompt(question: str, context: str) -> str:
    return "\n".join(
        [
            "You are answering a medical question for a case-grounded research experiment.",
            "Use only the selected top-ranked radiology case evidence below.",
            "Do not summarize other cases.",
            "Do not combine findings or impressions from multiple cases.",
            "If the selected case evidence is insufficient, say that the selected case evidence is insufficient.",
            "",
            "Question:",
            question,
            "",
            "Selected top-ranked case evidence:",
            context,
            "",
            "Respond in this structure:",
            "Evidence:",
            "- Quote or paraphrase only the relevant evidence from the selected case.",
            "",
            "Final answer:",
            "- Answer the question for the selected case only in one concise paragraph.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a hybrid BM25 + MedCPT case-RAG prompt pack.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument(
        "--prompt-mode",
        choices=["structured", "top1_structured"],
        default="structured",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    qa_payload = json.loads(args.qa.read_text(encoding="utf-8"))
    questions = qa_payload["questions"]

    bm25 = BM25Retriever().fit(cases)
    medcpt = MedCPTRetriever.from_index(args.cases, args.index)
    query_embeddings = encode_queries(
        [item["question"] for item in questions],
        batch_size=args.batch_size,
        device=args.device,
    )

    case_position = {case["case_id"]: idx for idx, case in enumerate(cases)}
    medcpt_case_position = {case_id: pos for pos, case_id in enumerate(medcpt.case_ids)}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as file:
        for query_index, item in enumerate(questions):
            bm25_query_terms = _tokens(item["question"])
            bm25_scores = np.array(
                [bm25._score_document(bm25_query_terms, idx) for idx in range(len(cases))],
                dtype="float32",
            )
            medcpt_scores_indexed = medcpt.embeddings @ query_embeddings[query_index]
            medcpt_scores = np.zeros(len(cases), dtype="float32")
            for case_id, med_pos in medcpt_case_position.items():
                medcpt_scores[case_position[case_id]] = medcpt_scores_indexed[med_pos]

            hybrid_scores = args.alpha * _minmax(medcpt_scores) + (1 - args.alpha) * _minmax(bm25_scores)
            ranked_indices = hybrid_scores.argsort()[::-1][: args.top_k]

            results = []
            for rank, index in enumerate(ranked_indices, start=1):
                case = cases[int(index)]
                results.append(
                    {
                        "rank": rank,
                        "case_id": case["case_id"],
                        "score": float(hybrid_scores[int(index)]),
                        "bm25_score": float(bm25_scores[int(index)]),
                        "medcpt_score": float(medcpt_scores[int(index)]),
                        "findings": case.get("findings", ""),
                        "impression": case.get("impression", ""),
                        "images": case.get("images", []),
                    }
                )

            evidence_results = results[:1] if args.prompt_mode == "top1_structured" else results
            context = _case_context(evidence_results)
            prompt = (
                _top1_structured_prompt(item["question"], context)
                if args.prompt_mode == "top1_structured"
                else _structured_prompt(item["question"], context)
            )
            record = {
                "qid": item["qid"],
                "case_id": item["case_id"],
                "question_type": item["question_type"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "relevant_case_ids": item["relevant_case_ids"],
                "system": "case_rag",
                "prompt_mode": args.prompt_mode,
                "retriever": "hybrid_bm25_medcpt",
                "alpha": args.alpha,
                "top_k": args.top_k,
                "retrieved_case_ids": [result["case_id"] for result in results],
                "evidence_case_ids": [result["case_id"] for result in evidence_results],
                "retrieved_context": context,
                "prompt": prompt,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(json.dumps({"output": str(args.output), "records": count, "alpha": args.alpha}, indent=2))


if __name__ == "__main__":
    main()
