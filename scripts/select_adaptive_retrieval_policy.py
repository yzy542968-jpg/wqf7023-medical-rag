from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.retrieval.adaptive_retrieval import select_adaptive_top1


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def decisions_for_config(
    qids: list[str],
    base: dict,
    reranker: dict,
    target_by_qid: dict[str, str],
    *,
    reranker_margin_threshold: float,
    base_margin_threshold: float,
    minimum_base_score: float,
    minimum_selected_margin: float,
) -> tuple[dict, list[dict]]:
    rows = []
    for qid in qids:
        base_details = base["selected_score_details"][qid]
        base_ids = [str(item["case_id"]) for item in base_details]
        base_scores = [float(item["hybrid_score"]) for item in base_details]
        reranked_ids = [str(value) for value in reranker["selected_rankings"][qid]]
        score_map = reranker["reranker_scores"][qid]
        scored_reranked_ids = [case_id for case_id in reranked_ids if case_id in score_map]
        reranker_scores = [float(score_map[case_id]) for case_id in scored_reranked_ids]
        decision = select_adaptive_top1(
            base_case_ids=base_ids,
            base_scores=base_scores,
            reranked_case_ids=scored_reranked_ids,
            reranker_scores=reranker_scores,
            reranker_margin_threshold=reranker_margin_threshold,
            base_margin_threshold=base_margin_threshold,
            minimum_base_score=minimum_base_score,
            minimum_selected_margin=minimum_selected_margin,
        )
        correct = decision.selected_case_id == target_by_qid[qid]
        rows.append(
            {
                "qid": qid,
                "target_case_id": target_by_qid[qid],
                "selected_case_id": decision.selected_case_id,
                "correct": correct,
                **decision.__dict__,
            }
        )

    answered = [row for row in rows if not row["abstained"]]
    correct_count = sum(bool(row["correct"]) for row in rows)
    summary = {
        "n": len(rows),
        "coverage": len(answered) / len(rows) if rows else 0.0,
        "overall_accuracy": correct_count / len(rows) if rows else 0.0,
        "selective_accuracy": (
            sum(bool(row["correct"]) for row in answered) / len(answered) if answered else 0.0
        ),
        "abstention_rate": 1.0 - len(answered) / len(rows) if rows else 0.0,
        "reranker_selection_rate": mean(
            [float(row["source"] == "reranker") for row in rows]
        ),
        "hybrid_selection_rate": mean([float(row["source"] == "hybrid") for row in rows]),
        "agreement_rate": mean([float(row["source"] == "agreement") for row in rows]),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Select an adaptive Hybrid/reranker policy.")
    parser.add_argument(
        "--qa",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_qa_seed_clean.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "openi_qa_grouped_case_seed7023.json",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "retrieval"
        / "hybrid_alpha_selection.json",
    )
    parser.add_argument(
        "--reranker",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "reranking"
        / "medcpt_reranker_selection.json",
    )
    parser.add_argument(
        "--reranker-margin-thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0],
    )
    parser.add_argument(
        "--base-margin-thresholds",
        nargs="+",
        type=float,
        default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.25, 1.0],
    )
    parser.add_argument(
        "--minimum-base-scores", nargs="+", type=float, default=[0.0, 0.90, 0.925, 0.95]
    )
    parser.add_argument(
        "--minimum-selected-margins", nargs="+", type=float, default=[0.0, 0.02, 0.05, 0.10]
    )
    parser.add_argument("--minimum-coverage", type=float, default=0.80)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "final_optimized" / "adaptive_retrieval",
    )
    args = parser.parse_args()

    questions = json.loads(args.qa.read_text(encoding="utf-8"))["questions"]
    target_by_qid = {str(item["qid"]): str(item["case_id"]) for item in questions}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    base = json.loads(args.base.read_text(encoding="utf-8"))
    reranker = json.loads(args.reranker.read_text(encoding="utf-8"))

    development_results = []
    for reranker_margin in args.reranker_margin_thresholds:
        for base_margin in args.base_margin_thresholds:
            for minimum_base_score in args.minimum_base_scores:
                for minimum_selected_margin in args.minimum_selected_margins:
                    summary, _ = decisions_for_config(
                        split["development"]["qids"],
                        base,
                        reranker,
                        target_by_qid,
                        reranker_margin_threshold=reranker_margin,
                        base_margin_threshold=base_margin,
                        minimum_base_score=minimum_base_score,
                        minimum_selected_margin=minimum_selected_margin,
                    )
                    development_results.append(
                        {
                            "reranker_margin_threshold": reranker_margin,
                            "base_margin_threshold": base_margin,
                            "minimum_base_score": minimum_base_score,
                            "minimum_selected_margin": minimum_selected_margin,
                            **summary,
                        }
                    )

    eligible = [row for row in development_results if row["coverage"] >= args.minimum_coverage]
    selected = max(
        eligible,
        key=lambda row: (
            row["overall_accuracy"],
            row["selective_accuracy"],
            row["coverage"],
            -row["reranker_selection_rate"],
        ),
    )
    policy_kwargs = {
        key: selected[key]
        for key in (
            "reranker_margin_threshold",
            "base_margin_threshold",
            "minimum_base_score",
            "minimum_selected_margin",
        )
    }
    test_summary, test_rows = decisions_for_config(
        split["test"]["qids"], base, reranker, target_by_qid, **policy_kwargs
    )
    development_summary, development_rows = decisions_for_config(
        split["development"]["qids"], base, reranker, target_by_qid, **policy_kwargs
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "adaptive_policy_development_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(development_results[0]))
        writer.writeheader()
        writer.writerows(development_results)
    test_path = args.output_dir / "adaptive_policy_test_decisions.jsonl"
    with test_path.open("w", encoding="utf-8") as handle:
        for row in test_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    development_path = args.output_dir / "adaptive_policy_development_decisions.jsonl"
    with development_path.open("w", encoding="utf-8") as handle:
        for row in development_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    output = {
        "selection_rule": (
            "maximize development overall accuracy with minimum coverage "
            f"{args.minimum_coverage:.2f}"
        ),
        "selected_policy": selected,
        "selected_development": development_summary,
        "held_out_test": test_summary,
        "split_manifest": str(args.split),
        "test_decisions": str(test_path),
        "development_decisions": str(development_path),
    }
    output_path = args.output_dir / "adaptive_policy_selection.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({**output, "development_sweep_csv": str(csv_path)}, indent=2))


if __name__ == "__main__":
    main()
