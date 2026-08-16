from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.tfidf_retriever import TfidfRetriever, load_cases_jsonl


def _make_retriever(name: str):
    if name == "tfidf":
        return TfidfRetriever()
    if name == "bm25":
        return BM25Retriever()
    raise ValueError(f"Unsupported retriever: {name}")


def _report_context(results: list[dict[str, Any]]) -> str:
    blocks = []
    for result in results:
        blocks.append(
            "\n".join(
                [
                    f"Case ID: {result['case_id']}",
                    f"Findings: {result.get('findings', '')}",
                    f"Impression: {result.get('impression', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


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


def _direct_prompt(question: str) -> str:
    return "\n".join(
        [
            "You are answering a medical question for research purposes.",
            "",
            "Question:",
            question,
            "",
            "Answer clearly and concisely.",
        ]
    )


def _evidence_guided_prompt(question: str, context: str) -> str:
    return "\n".join(
        [
            "You are answering a medical question for research purposes.",
            "Use only the provided radiology report evidence.",
            "If the evidence is insufficient, say that the evidence is insufficient.",
            "Do not add unsupported medical claims.",
            "",
            "Question:",
            question,
            "",
            "Retrieved report evidence:",
            context,
            "",
            "Answer:",
        ]
    )


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


def _build_prompt(system: str, prompt_mode: str, question: str, results: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]]]:
    if system == "llm_only":
        return _direct_prompt(question), "", []

    evidence_results = results[:1] if prompt_mode == "top1_structured" else results
    context = _report_context(evidence_results) if system == "report_rag" else _case_context(evidence_results)
    if prompt_mode == "evidence_guided":
        return _evidence_guided_prompt(question, context), context, evidence_results
    if prompt_mode == "structured":
        return _structured_prompt(question, context), context, evidence_results
    if prompt_mode == "top1_structured":
        return _top1_structured_prompt(question, context), context, evidence_results
    raise ValueError(f"Unsupported prompt mode for RAG system: {prompt_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build prompt pack for LLM and RAG experiments.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--system", choices=["llm_only", "report_rag", "case_rag"], required=True)
    parser.add_argument("--prompt-mode", choices=["direct", "evidence_guided", "structured", "top1_structured"], required=True)
    parser.add_argument("--retriever", choices=["tfidf", "bm25"], default="bm25")
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    qa_payload = json.loads(args.qa.read_text(encoding="utf-8"))
    retriever = None
    if args.system != "llm_only":
        retriever = _make_retriever(args.retriever).fit(cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", encoding="utf-8") as file:
        for item in qa_payload["questions"]:
            results = [] if retriever is None else retriever.search(item["question"], top_k=args.top_k)
            prompt, context, evidence_results = _build_prompt(
                system=args.system,
                prompt_mode=args.prompt_mode,
                question=item["question"],
                results=results,
            )
            record = {
                "qid": item["qid"],
                "case_id": item["case_id"],
                "question_type": item["question_type"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "relevant_case_ids": item["relevant_case_ids"],
                "system": args.system,
                "prompt_mode": args.prompt_mode,
                "retriever": None if args.system == "llm_only" else args.retriever,
                "top_k": 0 if args.system == "llm_only" else args.top_k,
                "retrieved_case_ids": [result["case_id"] for result in results],
                "evidence_case_ids": [result["case_id"] for result in evidence_results],
                "retrieved_context": context,
                "prompt": prompt,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(json.dumps({"output": str(args.output), "records": count}, indent=2))


if __name__ == "__main__":
    main()
