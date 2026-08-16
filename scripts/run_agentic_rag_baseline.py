from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic import run_rule_based_agent
from medical_rag.evaluation.answer_metrics import token_f1
from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.hybrid_retriever import HybridBM25MedCPTRetriever
from medical_rag.retrieval.medcpt_retriever import MedCPTRetriever, encode_queries
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rule-based evidence-checking agentic RAG baseline.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--qa", required=True, type=Path)
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--min-sentence-support", default=0.65, type=float)
    parser.add_argument("--min-retrieval-score", default=0.0, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    qa_payload = json.loads(args.qa.read_text(encoding="utf-8"))
    questions = qa_payload["questions"]

    bm25 = BM25Retriever().fit(cases)
    medcpt = MedCPTRetriever.from_index(args.cases, args.index)
    hybrid = HybridBM25MedCPTRetriever.from_components(cases, bm25, medcpt, alpha=args.alpha)
    query_embeddings = encode_queries(
        [item["question"] for item in questions],
        batch_size=args.batch_size,
        device=args.device,
    )

    answers = []
    token_f1_scores = []
    top1_hits = []
    retrieved_hits = []
    support_rates = []
    revised_flags = []
    abstained_flags = []
    non_empty_flags = []

    for query_index, item in enumerate(questions):
        retrieved = hybrid.search_with_embedding(
            query=item["question"],
            query_embedding=query_embeddings[query_index],
            top_k=args.top_k,
        )
        agent_result = run_rule_based_agent(
            question=item["question"],
            question_type=item["question_type"],
            retrieved_cases=retrieved,
            min_sentence_support=args.min_sentence_support,
            min_retrieval_score=args.min_retrieval_score,
        )

        relevant = set(item["relevant_case_ids"])
        retrieved_case_ids = [result["case_id"] for result in retrieved]
        top1_hit = bool(retrieved_case_ids and retrieved_case_ids[0] in relevant)
        retrieved_hit = bool(relevant.intersection(retrieved_case_ids))
        final_answer = agent_result.final_answer

        token_f1_scores.append(token_f1(final_answer, item["reference_answer"]))
        top1_hits.append(float(top1_hit))
        retrieved_hits.append(float(retrieved_hit))
        support_rates.append(agent_result.evidence_check.support_rate)
        revised_flags.append(float(agent_result.revised))
        abstained_flags.append(float(agent_result.abstained))
        non_empty_flags.append(float(bool(final_answer.strip())))

        answers.append(
            {
                "qid": item["qid"],
                "case_id": item["case_id"],
                "question_type": item["question_type"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "relevant_case_ids": item["relevant_case_ids"],
                "retrieved_case_ids": retrieved_case_ids,
                "top1_hit": top1_hit,
                "retrieved_hit": retrieved_hit,
                "draft_answer": agent_result.draft_answer,
                "final_answer": final_answer,
                "answer_token_f1": token_f1_scores[-1],
                "agent": agent_result.to_dict(),
            }
        )

    metrics = {
        "answer_token_f1": sum(token_f1_scores) / len(token_f1_scores),
        "top1_case_accuracy": sum(top1_hits) / len(top1_hits),
        "retrieved_case_hit_rate": sum(retrieved_hits) / len(retrieved_hits),
        "average_evidence_support_rate": sum(support_rates) / len(support_rates),
        "revision_rate": sum(revised_flags) / len(revised_flags),
        "abstention_rate": sum(abstained_flags) / len(abstained_flags),
        "non_empty_answer_rate": sum(non_empty_flags) / len(non_empty_flags),
    }

    output = {
        "system": "rule-based evidence-checking agentic RAG",
        "retriever": "hybrid BM25 + MedCPT",
        "alpha": args.alpha,
        "top_k": args.top_k,
        "min_sentence_support": args.min_sentence_support,
        "min_retrieval_score": args.min_retrieval_score,
        "cases": str(args.cases),
        "index": str(args.index),
        "qa": str(args.qa),
        "question_count": len(answers),
        "metrics": metrics,
        "answers": answers,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
