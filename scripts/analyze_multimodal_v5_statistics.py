from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_grouped_statistical_analysis import paired_grouped_bootstrap


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def grouped_rows(rows: list[dict], metric: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["case_id"])].append(float(row[metric]))
    return dict(grouped)


def paired_metrics(
    rows: list[dict], first_system: str, second_system: str, metrics: list[str], *, iterations: int, seed: int
) -> dict[str, dict[str, float]]:
    by_system = {
        system: [row for row in rows if str(row["system"]) == system]
        for system in (first_system, second_system)
    }
    result = {}
    for offset, metric in enumerate(metrics):
        result[metric] = paired_grouped_bootstrap(
            grouped_rows(by_system[first_system], metric),
            grouped_rows(by_system[second_system], metric),
            iterations=iterations,
            seed=seed + offset,
        )
    return result


def qa_rows(path: Path) -> list[dict]:
    return read_jsonl(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze fresh-cohort V5 retrieval and end-to-end QA results.")
    parser.add_argument("--retrieval-summary", type=Path, default=ROOT / "experiments" / "post_submission_v5" / "confirmation_retrieval_summary.json")
    parser.add_argument("--retrieval-rows", type=Path, default=ROOT / "experiments" / "post_submission_v5" / "confirmation_retrieval_rows.jsonl")
    parser.add_argument("--report-qa", type=Path, default=ROOT / "experiments" / "post_submission_v5" / "qa_report_only" / "final_optimized_test_rows.jsonl")
    parser.add_argument("--multimodal-qa", type=Path, default=ROOT / "experiments" / "post_submission_v5" / "qa_multimodal" / "final_optimized_test_rows.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments" / "post_submission_v5" / "v5_statistics.json")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7023)
    args = parser.parse_args()

    retrieval_rows = read_jsonl(args.retrieval_rows)
    retrieval_comparisons = {
        "indication_question_correct_image_vs_bm25": paired_metrics(
            retrieval_rows,
            "indication_question_correct_image",
            "indication_question_bm25",
            ["mrr", "hit@1", "hit@5", "hit@10", "token_f1"],
            iterations=args.iterations,
            seed=args.seed,
        ),
        "indication_question_correct_image_vs_question_only": paired_metrics(
            retrieval_rows,
            "indication_question_correct_image",
            "question_only_bm25",
            ["mrr", "hit@1", "hit@5", "hit@10", "token_f1"],
            iterations=args.iterations,
            seed=args.seed + 100,
        ),
    }

    report_qa = qa_rows(args.report_qa)
    multimodal_qa = qa_rows(args.multimodal_qa)
    qa_metrics = paired_metrics(
        [
            *[{**row, "system": "v5_report_only"} for row in report_qa],
            *[{**row, "system": "v5_multimodal"} for row in multimodal_qa],
        ],
        "v5_multimodal",
        "v5_report_only",
        ["draft_token_f1", "final_token_f1", "support_rate", "agent_abstained"],
        iterations=args.iterations,
        seed=args.seed + 200,
    )

    summary = json.loads(args.retrieval_summary.read_text(encoding="utf-8"))
    random_metrics = summary["random_image_control"]["metrics"]
    random_mrr = [float(row["mrr"]) for row in random_metrics]
    random_f1 = [float(row["token_f1"]) for row in random_metrics]
    correct_mrr = float(summary["metrics"]["indication_question_correct_image"]["mrr"])
    correct_f1 = float(summary["metrics"]["indication_question_correct_image"]["token_f1"])
    mrr_exceedances = sum(value >= correct_mrr for value in random_mrr)
    f1_exceedances = sum(value >= correct_f1 for value in random_f1)
    result = {
        "experiment": "v5_end_to_end_multimodal_qa",
        "retrieval_comparisons": retrieval_comparisons,
        "qa_comparison": qa_metrics,
        "random_image_control": {
            "permutations": len(random_mrr),
            "mrr_mean": mean(random_mrr),
            "mrr_min": min(random_mrr),
            "mrr_max": max(random_mrr),
            "token_f1_mean": mean(random_f1),
            "token_f1_min": min(random_f1),
            "token_f1_max": max(random_f1),
            "random_mrr_at_least_correct_count": mrr_exceedances,
            "random_token_f1_at_least_correct_count": f1_exceedances,
            "empirical_fraction_random_mrr_at_least_correct": mrr_exceedances / len(random_mrr),
            "empirical_fraction_random_token_f1_at_least_correct": f1_exceedances / len(random_f1),
            "plus_one_monte_carlo_p_mrr": (mrr_exceedances + 1) / (len(random_mrr) + 1),
            "plus_one_monte_carlo_p_token_f1": (f1_exceedances + 1) / (len(random_f1) + 1),
        },
        "n": {
            "retrieval_questions": len(retrieval_rows) // 4,
            "retrieval_cases": len({str(row["case_id"]) for row in retrieval_rows}),
            "qa_questions": len(report_qa),
            "qa_cases": len({str(row["case_id"]) for row in report_qa}),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
