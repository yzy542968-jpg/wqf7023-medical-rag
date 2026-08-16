from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.semantic_evidence_checker import (
    DEFAULT_NLI_MODEL,
    MedicalNLIPredictor,
    check_semantic_evidence_support,
)
from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


SYSTEMS = {
    "report_bm25": ROOT / "experiments" / "generations_report_rag_bm25_qwen15_full360.jsonl",
    "case_bm25_top1": ROOT / "experiments" / "generations_case_rag_bm25_top1_qwen15_full360.jsonl",
    "case_hybrid_top1_a050": ROOT
    / "experiments"
    / "generations_case_rag_hybrid_top1_qwen15_full360.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evidence_text(case: dict) -> str:
    return "\n".join(
        [
            f"Case ID: {case.get('case_id', '')}",
            f"Findings: {case.get('findings', '')}",
            f"Impression: {case.get('impression', '')}",
        ]
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def derive_answer(
    checks: list[dict],
    *,
    lexical_weight: float,
    min_combined_support: float,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> tuple[str, list[bool]]:
    decisions = []
    supported_sentences = []
    for check in checks:
        combined = (
            lexical_weight * float(check["lexical_score"])
            + (1.0 - lexical_weight) * float(check["entailment_probability"])
        )
        supported = (
            bool(check["negation_consistent"])
            and float(check["contradiction_probability"]) < contradiction_threshold
            and (
                combined >= min_combined_support
                or float(check["entailment_probability"]) >= entailment_threshold
            )
        )
        decisions.append(supported)
        if supported:
            supported_sentences.append(str(check["sentence"]))
    answer = (
        " ".join(supported_sentences)
        if supported_sentences
        else "The retrieved report evidence is insufficient to answer this question."
    )
    return answer, decisions


def evaluate_config(
    rows: list[dict],
    selected_qids: set[str],
    *,
    lexical_weight: float,
    min_combined_support: float,
    entailment_threshold: float,
    contradiction_threshold: float,
) -> tuple[dict, list[dict]]:
    selected = [row for row in rows if str(row["qid"]) in selected_qids]
    final_f1s = []
    draft_f1s = []
    revised = []
    abstained = []
    support_rates = []
    contradiction_rejections = 0
    contradiction_count = 0
    outputs = []
    for row in selected:
        final_answer, decisions = derive_answer(
            row["sentence_checks"],
            lexical_weight=lexical_weight,
            min_combined_support=min_combined_support,
            entailment_threshold=entailment_threshold,
            contradiction_threshold=contradiction_threshold,
        )
        sentence_count = len(decisions)
        support_rate = sum(decisions) / sentence_count if sentence_count else 0.0
        draft = row["draft_answer"]
        reference = row["reference_answer"]
        draft_f1s.append(token_f1(draft, reference))
        final_f1s.append(token_f1(final_answer, reference))
        revised.append(float(final_answer.strip() != draft.strip()))
        abstained.append(float(not any(decisions)))
        support_rates.append(support_rate)
        for check, decision in zip(row["sentence_checks"], decisions, strict=True):
            is_contradiction = float(check["contradiction_probability"]) >= contradiction_threshold
            contradiction_count += int(is_contradiction)
            contradiction_rejections += int(is_contradiction and not decision)
        outputs.append({**row, "final_answer": final_answer, "decisions": decisions})

    summary = {
        "n": len(selected),
        "draft_token_f1": mean(draft_f1s),
        "final_token_f1": mean(final_f1s),
        "revision_rate": mean(revised),
        "abstention_rate": mean(abstained),
        "evidence_support_rate": mean(support_rates),
        "nli_contradiction_count": contradiction_count,
        "nli_contradiction_rejection_rate": (
            contradiction_rejections / contradiction_count if contradiction_count else 1.0
        ),
    }
    return summary, outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run semantic evidence-checker ablation on P2 outputs.")
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "data" / "processed" / "openi_cases.jsonl"
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "openi_qa_grouped_case_seed7023.json",
    )
    parser.add_argument("--model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--lexical-weights", nargs="+", type=float, default=[0.0, 0.2, 0.35, 0.5, 0.7, 1.0]
    )
    parser.add_argument(
        "--support-thresholds",
        nargs="+",
        type=float,
        default=[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    )
    parser.add_argument("--entailment-threshold", type=float, default=0.75)
    parser.add_argument("--contradiction-threshold", type=float, default=0.50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "final_optimized" / "semantic_agent",
    )
    parser.add_argument("--reuse-scores", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "semantic_sentence_scores.jsonl"
    cases = load_cases_jsonl(args.cases)
    case_by_id = {str(case["case_id"]): case for case in cases}

    if args.reuse_scores and score_path.exists():
        scored_rows = read_jsonl(score_path)
    else:
        predictor = MedicalNLIPredictor(
            args.model,
            device=args.device,
            batch_size=args.batch_size,
            local_files_only=False,
        )
        scored_rows = []
        with score_path.open("w", encoding="utf-8") as handle:
            for system, path in SYSTEMS.items():
                for row in read_jsonl(path):
                    retrieved_ids = [str(value) for value in row.get("retrieved_case_ids", [])]
                    top_case = case_by_id.get(retrieved_ids[0]) if retrieved_ids else None
                    context = evidence_text(top_case) if top_case else ""
                    draft = extract_final_answer(row.get("answer", ""))
                    result = check_semantic_evidence_support(
                        draft,
                        context,
                        predictor,
                        min_combined_support=0.55,
                        entailment_threshold=args.entailment_threshold,
                        contradiction_threshold=args.contradiction_threshold,
                        lexical_weight=0.35,
                    )
                    scored = {
                        "system": system,
                        "qid": row.get("qid"),
                        "case_id": row.get("case_id"),
                        "question_type": row.get("question_type"),
                        "question": row.get("question"),
                        "reference_answer": row.get("reference_answer", ""),
                        "retrieved_case_ids": retrieved_ids,
                        "evidence_case_id": retrieved_ids[0] if retrieved_ids else None,
                        "draft_answer": draft,
                        "sentence_checks": [asdict(check) for check in result.sentence_checks],
                    }
                    scored_rows.append(scored)
                    handle.write(json.dumps(scored, ensure_ascii=False) + "\n")

    split = json.loads(args.split.read_text(encoding="utf-8"))
    development_qids = set(split["development"]["qids"])
    test_qids = set(split["test"]["qids"])
    development_sweep = []
    for lexical_weight in args.lexical_weights:
        for support_threshold in args.support_thresholds:
            system_summaries = []
            for system in SYSTEMS:
                summary, _ = evaluate_config(
                    [row for row in scored_rows if row["system"] == system],
                    development_qids,
                    lexical_weight=lexical_weight,
                    min_combined_support=support_threshold,
                    entailment_threshold=args.entailment_threshold,
                    contradiction_threshold=args.contradiction_threshold,
                )
                system_summaries.append(summary)
            development_sweep.append(
                {
                    "lexical_weight": lexical_weight,
                    "support_threshold": support_threshold,
                    "entailment_threshold": args.entailment_threshold,
                    "contradiction_threshold": args.contradiction_threshold,
                    "macro_draft_token_f1": mean(
                        [summary["draft_token_f1"] for summary in system_summaries]
                    ),
                    "macro_final_token_f1": mean(
                        [summary["final_token_f1"] for summary in system_summaries]
                    ),
                    "macro_revision_rate": mean(
                        [summary["revision_rate"] for summary in system_summaries]
                    ),
                    "macro_abstention_rate": mean(
                        [summary["abstention_rate"] for summary in system_summaries]
                    ),
                    "macro_evidence_support_rate": mean(
                        [summary["evidence_support_rate"] for summary in system_summaries]
                    ),
                    "contradiction_rejection_rate": mean(
                        [summary["nli_contradiction_rejection_rate"] for summary in system_summaries]
                    ),
                }
            )

    eligible = [row for row in development_sweep if row["macro_abstention_rate"] <= 0.50]
    selected = max(
        eligible,
        key=lambda row: (
            row["macro_final_token_f1"],
            row["contradiction_rejection_rate"],
            -row["macro_abstention_rate"],
        ),
    )
    selected_kwargs = {
        "lexical_weight": selected["lexical_weight"],
        "min_combined_support": selected["support_threshold"],
        "entailment_threshold": selected["entailment_threshold"],
        "contradiction_threshold": selected["contradiction_threshold"],
    }

    held_out_test = {}
    selected_test_rows = []
    for system in SYSTEMS:
        summary, outputs = evaluate_config(
            [row for row in scored_rows if row["system"] == system],
            test_qids,
            **selected_kwargs,
        )
        held_out_test[system] = summary
        selected_test_rows.extend(outputs)

    csv_path = args.output_dir / "semantic_agent_development_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(development_sweep[0]))
        writer.writeheader()
        writer.writerows(development_sweep)
    selected_rows_path = args.output_dir / "semantic_agent_selected_test_rows.jsonl"
    with selected_rows_path.open("w", encoding="utf-8") as handle:
        for row in selected_test_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "nli_model": args.model,
        "split_manifest": str(args.split),
        "selection_rule": "maximize development macro final Token-F1 with abstention <= 0.50",
        "selected_config": selected,
        "held_out_test": held_out_test,
        "score_rows": len(scored_rows),
        "score_path": str(score_path),
        "development_sweep_csv": str(csv_path),
        "selected_test_rows": str(selected_rows_path),
    }
    result_path = args.output_dir / "semantic_agent_selection.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({**result, "score_path": str(score_path)}, indent=2))


if __name__ == "__main__":
    main()
