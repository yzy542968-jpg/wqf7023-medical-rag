from __future__ import annotations

import json
from pathlib import Path


EXPERIMENTS = [
    ("Keyword qrels", "TF-IDF", Path("experiments/tfidf_keyword_eval.json")),
    ("Keyword qrels", "BM25", Path("experiments/bm25_keyword_eval.json")),
    ("QA seed", "TF-IDF", Path("experiments/tfidf_qa_seed_eval.json")),
    ("QA seed", "BM25", Path("experiments/bm25_qa_seed_eval.json")),
    ("Clean QA seed", "TF-IDF", Path("experiments/tfidf_qa_seed_clean_eval.json")),
    ("Clean QA seed", "BM25", Path("experiments/bm25_qa_seed_clean_eval.json")),
    ("Clean QA seed", "MedCPT", Path("experiments/medcpt_qa_seed_clean_eval.json")),
    ("Clean QA seed", "Hybrid a=0.25", Path("experiments/hybrid_bm25_medcpt_a025_qa_seed_clean_eval.json")),
    ("Clean QA seed", "Hybrid a=0.50", Path("experiments/hybrid_bm25_medcpt_a050_qa_seed_clean_eval.json")),
    ("Clean QA seed", "Hybrid a=0.75", Path("experiments/hybrid_bm25_medcpt_a075_qa_seed_clean_eval.json")),
]

ANSWER_EXPERIMENTS = [
    ("QA seed", "Extractive TF-IDF RAG", Path("experiments/extractive_tfidf_qa_seed_answers.json")),
    ("QA seed", "Extractive BM25 RAG", Path("experiments/extractive_bm25_qa_seed_answers.json")),
    ("Clean QA seed", "Extractive TF-IDF RAG", Path("experiments/extractive_tfidf_qa_seed_clean_answers.json")),
    ("Clean QA seed", "Extractive BM25 RAG", Path("experiments/extractive_bm25_qa_seed_clean_answers.json")),
    ("Clean QA seed", "Agentic Hybrid RAG a=0.50", Path("experiments/agentic_hybrid_a050_qa_seed_clean_answers.json")),
]

GENERATION_PILOTS = [
    ("Pilot 30", "LLM-only Qwen2.5-0.5B", Path("experiments/generations_llm_only_qwen05_pilot30_eval.json")),
    ("Pilot 30", "Report RAG BM25 Qwen2.5-0.5B", Path("experiments/generations_report_rag_bm25_qwen05_pilot30_eval.json")),
    ("Pilot 30", "Case RAG BM25 Qwen2.5-0.5B", Path("experiments/generations_case_rag_bm25_qwen05_pilot30_eval.json")),
    ("Pilot 30", "Case RAG Hybrid Qwen2.5-0.5B", Path("experiments/generations_case_rag_hybrid_qwen05_pilot30_eval.json")),
    ("Pilot 30", "LLM-only Qwen2.5-1.5B", Path("experiments/generations_llm_only_qwen15_pilot30_eval.json")),
    ("Pilot 30", "Report RAG BM25 Qwen2.5-1.5B", Path("experiments/generations_report_rag_bm25_qwen15_pilot30_eval.json")),
    ("Pilot 30", "Case RAG BM25 top-1 Qwen2.5-1.5B", Path("experiments/generations_case_rag_bm25_top1_qwen15_pilot30_eval.json")),
    ("Pilot 30", "Case RAG Hybrid top-1 Qwen2.5-1.5B", Path("experiments/generations_case_rag_hybrid_top1_qwen15_pilot30_eval.json")),
    ("Full 360", "LLM-only Qwen2.5-1.5B", Path("experiments/generations_llm_only_qwen15_full360_eval.json")),
    ("Full 360", "Report RAG BM25 Qwen2.5-1.5B", Path("experiments/generations_report_rag_bm25_qwen15_full360_eval.json")),
    ("Full 360", "Case RAG BM25 top-1 Qwen2.5-1.5B", Path("experiments/generations_case_rag_bm25_top1_qwen15_full360_eval.json")),
    ("Full 360", "Case RAG Hybrid top-1 Qwen2.5-1.5B", Path("experiments/generations_case_rag_hybrid_top1_qwen15_full360_eval.json")),
]

AGENTIC_GENERATION_PILOTS = [
    (
        "Pilot 30",
        "Report RAG BM25 + top-1 evidence check",
        Path("experiments/generations_report_rag_bm25_qwen05_pilot30_agentic_top1_eval.json"),
    ),
    (
        "Pilot 30",
        "Case RAG BM25 + top-1 evidence check",
        Path("experiments/generations_case_rag_bm25_qwen05_pilot30_agentic_top1_eval.json"),
    ),
    (
        "Pilot 30",
        "Case RAG Hybrid + top-1 evidence check",
        Path("experiments/generations_case_rag_hybrid_qwen05_pilot30_agentic_top1_eval.json"),
    ),
    (
        "Pilot 30",
        "Case RAG Hybrid top-1 Qwen2.5-1.5B + top-1 evidence check",
        Path("experiments/generations_case_rag_hybrid_top1_qwen15_pilot30_agentic_top1_eval.json"),
    ),
    (
        "Pilot 30",
        "Report RAG BM25 Qwen2.5-1.5B + top-1 evidence check",
        Path("experiments/generations_report_rag_bm25_qwen15_pilot30_agentic_top1_eval.json"),
    ),
    (
        "Pilot 30",
        "Case RAG BM25 top-1 Qwen2.5-1.5B + top-1 evidence check",
        Path("experiments/generations_case_rag_bm25_top1_qwen15_pilot30_agentic_top1_eval.json"),
    ),
    (
        "Full 360",
        "Report RAG BM25 Qwen2.5-1.5B + top-1 evidence check",
        Path("experiments/generations_report_rag_bm25_qwen15_full360_agentic_top1_eval.json"),
    ),
    (
        "Full 360",
        "Case RAG BM25 top-1 Qwen2.5-1.5B + top-1 evidence check",
        Path("experiments/generations_case_rag_bm25_top1_qwen15_full360_agentic_top1_eval.json"),
    ),
    (
        "Full 360",
        "Case RAG Hybrid top-1 Qwen2.5-1.5B + top-1 evidence check",
        Path("experiments/generations_case_rag_hybrid_top1_qwen15_full360_agentic_top1_eval.json"),
    ),
]


def _fmt(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def main() -> None:
    rows = []
    for dataset, retriever, path in EXPERIMENTS:
        if not path.exists():
            rows.append([dataset, retriever, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", f"missing: `{path}`"])
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        rows.append(
            [
                dataset,
                retriever,
                _fmt(metrics.get("hit@1")),
                _fmt(metrics.get("hit@3")),
                _fmt(metrics.get("hit@5")),
                _fmt(metrics.get("hit@10")),
                _fmt(metrics.get("hit@20")),
                _fmt(metrics.get("mrr")),
                f"`{path}`",
            ]
        )

    lines = [
        "# Experiment Summary",
        "",
        "| Dataset | Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | Source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Note: `N/A` means the experiment has not been run in the current environment.")
    lines.append("")
    lines.append("## Extractive Answer Baselines")
    lines.append("")
    lines.append(
        "| Dataset | System | Answer Token-F1 | Top-1 Case Accuracy | Retrieved Hit Rate | Evidence Support | Revision Rate | Abstention Rate | Non-Empty Answer Rate | Source |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for dataset, system, path in ANSWER_EXPERIMENTS:
        if not path.exists():
            lines.append(f"| {dataset} | {system} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | missing: `{path}` |")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    system,
                    _fmt(metrics.get("answer_token_f1")),
                    _fmt(metrics.get("top1_case_accuracy")),
                    _fmt(metrics.get("retrieved_case_hit_rate")),
                    _fmt(metrics.get("average_evidence_support_rate")),
                    _fmt(metrics.get("revision_rate")),
                    _fmt(metrics.get("abstention_rate")),
                    _fmt(metrics.get("non_empty_answer_rate")),
                    f"`{path}`",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Qwen2.5 Generation Results")
    lines.append("")
    lines.append("| Dataset | System | Answer Token-F1 | Top-1 Case Accuracy | Retrieved Hit Rate | Avg Answer Words | Insufficient Rate | Source |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---|")
    for dataset, system, path in GENERATION_PILOTS:
        if not path.exists():
            lines.append(f"| {dataset} | {system} | N/A | N/A | N/A | N/A | N/A | missing: `{path}` |")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    system,
                    _fmt(metrics.get("answer_token_f1")),
                    _fmt(metrics.get("top1_case_accuracy")),
                    _fmt(metrics.get("retrieved_case_hit_rate")),
                    _fmt(metrics.get("average_answer_words")),
                    _fmt(metrics.get("insufficient_answer_rate")),
                    f"`{path}`",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Agentic Evidence-Checking Results")
    lines.append("")
    lines.append("| Dataset | System | Draft Token-F1 | Final Token-F1 | Evidence Support | Revision Rate | Abstention Rate | Unsupported Sentence Rate | Source |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for dataset, system, path in AGENTIC_GENERATION_PILOTS:
        if not path.exists():
            lines.append(f"| {dataset} | {system} | N/A | N/A | N/A | N/A | N/A | N/A | missing: `{path}` |")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = payload["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    system,
                    _fmt(metrics.get("draft_answer_token_f1")),
                    _fmt(metrics.get("final_answer_token_f1")),
                    _fmt(metrics.get("average_evidence_support_rate")),
                    _fmt(metrics.get("revision_rate")),
                    _fmt(metrics.get("abstention_rate")),
                    _fmt(metrics.get("unsupported_sentence_rate")),
                    f"`{path}`",
                ]
            )
            + " |"
        )

    output = Path("experiments/EXPERIMENT_SUMMARY.md")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
