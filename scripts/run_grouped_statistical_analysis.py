from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1


GENERATION_SYSTEMS = {
    "llm_only": ROOT / "experiments" / "generations_llm_only_qwen15_full360.jsonl",
    "report_bm25_draft": ROOT / "experiments" / "generations_report_rag_bm25_qwen15_full360.jsonl",
    "case_bm25_draft": ROOT
    / "experiments"
    / "generations_case_rag_bm25_top1_qwen15_full360.jsonl",
    "case_hybrid_a050_draft": ROOT
    / "experiments"
    / "generations_case_rag_hybrid_top1_qwen15_full360.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def grouped_bootstrap_ci(
    values_by_case: dict[str, list[float]], *, iterations: int, seed: int
) -> tuple[float, float, float]:
    cases = sorted(values_by_case)
    observed_values = [value for case_id in cases for value in values_by_case[case_id]]
    observed = sum(observed_values) / len(observed_values) if observed_values else 0.0
    rng = random.Random(seed)
    samples = []
    for _ in range(iterations):
        selected_cases = [rng.choice(cases) for _ in cases]
        selected_values = [
            value for case_id in selected_cases for value in values_by_case[case_id]
        ]
        samples.append(sum(selected_values) / len(selected_values))
    return observed, percentile(samples, 0.025), percentile(samples, 0.975)


def paired_grouped_bootstrap(
    first: dict[str, list[float]],
    second: dict[str, list[float]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    cases = sorted(set(first).intersection(second))
    differences = {
        case_id: sum(first[case_id]) / len(first[case_id])
        - sum(second[case_id]) / len(second[case_id])
        for case_id in cases
    }
    observed = sum(differences.values()) / len(differences)
    rng = random.Random(seed)
    bootstrap_differences = [
        sum(differences[rng.choice(cases)] for _ in cases) / len(cases)
        for _ in range(iterations)
    ]
    opposite_or_zero = sum(value <= 0.0 for value in bootstrap_differences)
    same_or_zero = sum(value >= 0.0 for value in bootstrap_differences)
    p_value = min(1.0, 2.0 * min(opposite_or_zero, same_or_zero) / iterations)
    randomization_rng = random.Random(seed + 1_000_000)
    observed_absolute = abs(observed)
    randomization_extreme = 0
    difference_values = list(differences.values())
    for _ in range(iterations):
        randomized = sum(
            value if randomization_rng.random() < 0.5 else -value
            for value in difference_values
        ) / len(difference_values)
        randomization_extreme += abs(randomized) >= observed_absolute
    return {
        "mean_difference": observed,
        "ci_low": percentile(bootstrap_differences, 0.025),
        "ci_high": percentile(bootstrap_differences, 0.975),
        "two_sided_bootstrap_p": p_value,
        "paired_randomization_p": (randomization_extreme + 1) / (iterations + 1),
        "case_count": len(cases),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise error adjusted p-values in original order."""
    count = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda value: value[1])
    adjusted = [0.0] * count
    running_maximum = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_value)
        running_maximum = max(running_maximum, candidate)
        adjusted[original_index] = running_maximum
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped bootstrap analysis for final P2 systems.")
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "openi_qa_grouped_case_seed7023.json",
    )
    parser.add_argument(
        "--semantic-test-rows",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "semantic_agent"
        / "semantic_agent_selected_test_rows.jsonl",
    )
    parser.add_argument(
        "--final-test-rows",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "final_test"
        / "final_optimized_test_rows.jsonl",
    )
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7023)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "final_optimized" / "statistics",
    )
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    test_qids = set(split["test"]["qids"])
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for system, path in GENERATION_SYSTEMS.items():
        for row in read_jsonl(path):
            if str(row["qid"]) not in test_qids:
                continue
            answer = extract_final_answer(row.get("answer", ""))
            values[system][str(row["case_id"])].append(
                token_f1(answer, row.get("reference_answer", ""))
            )

    semantic_rows = read_jsonl(args.semantic_test_rows)
    for row in semantic_rows:
        system = f"{row['system']}_semantic_agent"
        values[system][str(row["case_id"])].append(
            token_f1(row["final_answer"], row["reference_answer"])
        )
    if args.final_test_rows.exists():
        for row in read_jsonl(args.final_test_rows):
            values["final_adaptive_direct_draft"][str(row["case_id"])].append(
                token_f1(row["draft_answer"], row["reference_answer"])
            )
            values["final_adaptive_direct_semantic_agent"][str(row["case_id"])].append(
                token_f1(row["final_answer"], row["reference_answer"])
            )

    summary_rows = []
    for index, (system, by_case) in enumerate(sorted(values.items())):
        observed, ci_low, ci_high = grouped_bootstrap_ci(
            by_case, iterations=args.iterations, seed=args.seed + index
        )
        summary_rows.append(
            {
                "system": system,
                "mean_token_f1": observed,
                "ci_low_95": ci_low,
                "ci_high_95": ci_high,
                "case_count": len(by_case),
                "question_count": sum(len(case_values) for case_values in by_case.values()),
            }
        )

    comparisons = [
        ("report_bm25_semantic_agent", "report_bm25_draft"),
        ("case_bm25_top1_semantic_agent", "case_bm25_draft"),
        ("case_hybrid_top1_a050_semantic_agent", "case_hybrid_a050_draft"),
        ("case_hybrid_a050_draft", "case_bm25_draft"),
        ("case_hybrid_top1_a050_semantic_agent", "case_bm25_top1_semantic_agent"),
        ("case_hybrid_top1_a050_semantic_agent", "llm_only"),
        ("final_adaptive_direct_semantic_agent", "final_adaptive_direct_draft"),
        ("final_adaptive_direct_semantic_agent", "case_hybrid_top1_a050_semantic_agent"),
        ("final_adaptive_direct_semantic_agent", "case_bm25_top1_semantic_agent"),
        ("final_adaptive_direct_semantic_agent", "llm_only"),
    ]
    pairwise_rows = []
    for index, (first, second) in enumerate(comparisons):
        result = paired_grouped_bootstrap(
            values[first],
            values[second],
            iterations=args.iterations,
            seed=args.seed + 100 + index,
        )
        pairwise_rows.append({"system_a": first, "system_b": second, **result})
    adjusted_p_values = holm_adjust(
        [float(row["paired_randomization_p"]) for row in pairwise_rows]
    )
    for row, adjusted_p in zip(pairwise_rows, adjusted_p_values, strict=True):
        row["holm_adjusted_randomization_p"] = adjusted_p

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "held_out_test_grouped_bootstrap_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    pairwise_path = args.output_dir / "held_out_test_pairwise_bootstrap.csv"
    with pairwise_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairwise_rows[0]))
        writer.writeheader()
        writer.writerows(pairwise_rows)
    output = {
        "split_manifest": str(args.split),
        "iterations": args.iterations,
        "seed": args.seed,
        "summary": summary_rows,
        "pairwise": pairwise_rows,
    }
    json_path = args.output_dir / "held_out_test_grouped_bootstrap.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
