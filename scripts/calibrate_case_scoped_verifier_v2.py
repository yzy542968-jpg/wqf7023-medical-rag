from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from medical_rag.agentic.semantic_evidence_checker import DEFAULT_NLI_MODEL, MedicalNLIPredictor, check_semantic_evidence_support
from medical_rag.agentic.action_policy import apply_verifier_action
from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1
from scripts.run_semantic_evidence_ablation import derive_answer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_config(
    scored_rows: list[dict[str, Any]],
    lexical_weight: float,
    support_threshold: float,
    entailment_threshold: float,
    contradiction_threshold: float,
    action_policy: str = "sentence_filter",
) -> dict[str, Any]:
    draft_scores = []
    final_scores = []
    abstentions = []
    revisions = []
    support_rates = []
    for row in scored_rows:
        checks = row["sentence_checks"]
        filtered_answer, support_decisions = derive_answer(
            checks,
            lexical_weight=lexical_weight,
            min_combined_support=support_threshold,
            entailment_threshold=entailment_threshold,
            contradiction_threshold=contradiction_threshold,
        )
        if action_policy == "sentence_filter":
            final_answer = filtered_answer
            action_abstained = not any(support_decisions)
        else:
            action = apply_verifier_action(
                row["draft_answer"],
                checks,
                action_policy=action_policy,
                contradiction_threshold=contradiction_threshold,
            )
            final_answer = action.answer
            action_abstained = action.abstained
        draft_scores.append(token_f1(row["draft_answer"], row["reference_answer"]))
        final_scores.append(token_f1(final_answer, row["reference_answer"]))
        abstentions.append(float(action_abstained))
        revisions.append(float(final_answer.strip() != row["draft_answer"].strip()))
        support_rates.append(
            sum(support_decisions) / len(support_decisions) if support_decisions else 0.0
        )
    return {
        "lexical_weight": lexical_weight,
        "support_threshold": support_threshold,
        "entailment_threshold": entailment_threshold,
        "contradiction_threshold": contradiction_threshold,
        "action_policy": action_policy,
        "n": len(scored_rows),
        "draft_token_f1": mean(draft_scores),
        "final_token_f1": mean(final_scores),
        "abstention_rate": mean(abstentions),
        "revision_rate": mean(revisions),
        "evidence_support_rate": mean(support_rates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the semantic verifier only on v2 calibration answers.")
    parser.add_argument(
        "--generations",
        type=Path,
        default=ROOT / "experiments" / "benchmark_v2" / "generations" / "calibration_case_scoped_routed_qwen15.jsonl",
    )
    parser.add_argument("--reuse-scores", action="store_true")
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=ROOT / "data" / "processed" / "prompt_packs" / "benchmark_v2" / "calibration_case_scoped_routed.jsonl",
    )
    parser.add_argument("--model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-abstention-rate", type=float, default=0.10)
    parser.add_argument("--entailment-threshold", type=float, default=0.75)
    parser.add_argument("--contradiction-threshold", type=float, default=0.50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "benchmark_v2" / "calibration" / "semantic_verifier",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "calibration_sentence_scores.jsonl"
    if args.reuse_scores and score_path.exists():
        scored_rows = read_jsonl(score_path)
    else:
        generations = read_jsonl(args.generations)
        prompts = {row["qid"]: row for row in read_jsonl(args.prompt_pack)}
        predictor = MedicalNLIPredictor(
            args.model,
            device=args.device,
            batch_size=args.batch_size,
            local_files_only=True,
        )
        scored_rows = []
        for generation in generations:
            prompt = prompts[generation["qid"]]
            draft = extract_final_answer(generation.get("answer", ""))
            check = check_semantic_evidence_support(
                draft,
                prompt["retrieved_context"],
                predictor,
                min_combined_support=0.0,
                entailment_threshold=0.0,
                contradiction_threshold=args.contradiction_threshold,
                lexical_weight=0.0,
            )
            scored_rows.append(
                {
                    "qid": generation["qid"],
                    "case_id": generation["case_id"],
                    "question_type": generation["question_type"],
                    "reference_answer": generation["reference_answer"],
                    "draft_answer": draft,
                    "sentence_checks": [asdict(value) for value in check.sentence_checks],
                }
            )

    grid = []
    for lexical_weight in (0.0, 0.2, 0.35, 0.5, 0.7, 1.0):
        for support_threshold in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            grid.append(
                evaluate_config(
                    scored_rows,
                    lexical_weight,
                    support_threshold,
                    args.entailment_threshold,
                    args.contradiction_threshold,
                )
            )
    for contradiction_threshold in (0.50, 0.70, 0.90):
        grid.append(
            evaluate_config(
                scored_rows,
                lexical_weight=1.0,
                support_threshold=0.40,
                entailment_threshold=args.entailment_threshold,
                contradiction_threshold=contradiction_threshold,
                action_policy="contradiction_only",
            )
        )
    grid.append(
        evaluate_config(
            scored_rows,
            lexical_weight=1.0,
            support_threshold=0.40,
            entailment_threshold=args.entailment_threshold,
            contradiction_threshold=args.contradiction_threshold,
            action_policy="audit_only",
        )
    )
    eligible = [row for row in grid if row["abstention_rate"] <= args.max_abstention_rate]
    if not eligible:
        raise RuntimeError("No verifier configuration satisfies the abstention constraint.")
    selected = max(
        eligible,
        key=lambda row: (
            row["final_token_f1"],
            -row["abstention_rate"],
            -row["revision_rate"],
            row["support_threshold"],
        ),
    )
    selection = {
        "selection_partition": "calibration",
        "selection_rule": "maximize final Token-F1 across calibrated action policies subject to abstention <= 0.10; audit-only remains advisory and does not rewrite",
        "nli_model": args.model,
        "max_abstention_rate": args.max_abstention_rate,
        "selected_config": selected,
        "all_configs": sorted(grid, key=lambda row: row["final_token_f1"], reverse=True),
    }
    with score_path.open("w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_path = args.output_dir / "semantic_agent_selection.json"
    output_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps({**selection, "all_configs": f"{len(grid)} configurations", "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
