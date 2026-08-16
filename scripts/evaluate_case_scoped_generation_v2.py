from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.semantic_evidence_checker import MedicalNLIPredictor, check_semantic_evidence_support
from medical_rag.agentic.action_policy import apply_verifier_action
from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def grouped_bootstrap_ci(
    rows: list[dict[str, Any]],
    value_key: str,
    samples: int = 10_000,
    seed: int = 7023,
) -> list[float]:
    by_case: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(float(row[value_key]))
    case_values = [mean(values) for values in by_case.values()]
    rng = random.Random(seed)
    bootstrapped = [
        mean(rng.choices(case_values, k=len(case_values))) for _ in range(samples)
    ]
    return [percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)]


def summarize_rows(rows: list[dict[str, Any]], bootstrap_samples: int = 10_000) -> dict[str, Any]:
    def summarize(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(selected),
            "case_count": len({row["case_id"] for row in selected}),
            "draft_token_f1": mean(float(row["draft_token_f1"]) for row in selected),
            "verified_token_f1": mean(float(row["final_token_f1"]) for row in selected),
            "evidence_support_rate": mean(float(row["support_rate"]) for row in selected),
            "agent_abstention_rate": mean(float(row["agent_abstained"]) for row in selected),
            "revision_rate": mean(float(row["revised"]) for row in selected),
            "mean_retrieval_recall": mean(float(row["retrieval_recall"]) for row in selected),
        }

    overall = summarize(rows)
    overall["draft_token_f1_case_bootstrap_95_ci"] = grouped_bootstrap_ci(
        rows, "draft_token_f1", bootstrap_samples
    )
    overall["verified_token_f1_case_bootstrap_95_ci"] = grouped_bootstrap_ci(
        rows, "final_token_f1", bootstrap_samples
    )
    overall["by_question_type"] = {
        question_type: summarize([row for row in rows if row["question_type"] == question_type])
        for question_type in sorted({row["question_type"] for row in rows})
    }
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the locked case-scoped v2 generator and verifier.")
    parser.add_argument(
        "--generations",
        type=Path,
        default=ROOT / "experiments" / "benchmark_v2" / "generations" / "test_case_scoped_routed_qwen15.jsonl",
    )
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=ROOT / "data" / "processed" / "prompt_packs" / "benchmark_v2" / "test_case_scoped_routed.jsonl",
    )
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=ROOT / "experiments" / "final_optimized" / "semantic_agent" / "semantic_agent_selection.json",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--split-label", default="test")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "benchmark_v2" / "generation_evaluation",
    )
    args = parser.parse_args()

    generations = read_jsonl(args.generations)
    prompts = {row["qid"]: row for row in read_jsonl(args.prompt_pack)}
    missing = sorted({row["qid"] for row in generations} - prompts.keys())
    if missing:
        raise ValueError(f"Generation qids missing from prompt pack: {missing[:3]}")
    semantic_selection = json.loads(args.semantic_config.read_text(encoding="utf-8"))
    config = semantic_selection["selected_config"]
    predictor = MedicalNLIPredictor(
        semantic_selection["nli_model"],
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=True,
    )

    output_rows = []
    for generation in generations:
        prompt = prompts[generation["qid"]]
        draft = extract_final_answer(generation.get("answer", ""))
        check = check_semantic_evidence_support(
            draft,
            prompt["retrieved_context"],
            predictor,
            min_combined_support=float(config["support_threshold"]),
            entailment_threshold=float(config["entailment_threshold"]),
            contradiction_threshold=float(config["contradiction_threshold"]),
            lexical_weight=float(config["lexical_weight"]),
        )
        action = apply_verifier_action(
            draft,
            check.sentence_checks,
            action_policy=str(config.get("action_policy", "sentence_filter")),
            contradiction_threshold=float(config["contradiction_threshold"]),
        )
        reference = str(generation["reference_answer"])
        output_rows.append(
            {
                "system": "v2_case_scoped_agent_routed_qwen15_semantic_agent",
                "qid": generation["qid"],
                "case_id": generation["case_id"],
                "question_type": generation["question_type"],
                "question": generation["question"],
                "reference_answer": reference,
                "draft_answer": draft,
                "final_answer": action.answer,
                "draft_token_f1": token_f1(draft, reference),
                "final_token_f1": token_f1(action.answer, reference),
                "support_rate": check.support_rate,
                "agent_abstained": action.abstained,
                "revised": action.revised,
                "rejected_sentences": action.rejected_sentences,
                "top_k": prompt["top_k"],
                "retrieval_recall": prompt["retrieval_recall"],
                "retrieved_chunk_ids": prompt["retrieved_chunk_ids"],
                "retrieved_sections": prompt["retrieved_sections"],
                "sentence_checks": [asdict(value) for value in check.sentence_checks],
            }
        )

    summary = {
        "benchmark": "OpenI case-scoped evidence QA v2",
        "split": args.split_label,
        "system": "v2_case_scoped_agent_routed_qwen15_semantic_agent",
        "top_k": prompts[generations[0]["qid"]]["top_k"],
        **summarize_rows(output_rows, args.bootstrap_samples),
        "semantic_agent_config": config,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "test_generation_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["rows_path"] = str(rows_path)
    summary_path = args.output_dir / "test_generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
