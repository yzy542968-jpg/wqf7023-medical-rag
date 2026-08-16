from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ambiguity_summary(questions: list[dict], selected_qids: set[str]) -> dict:
    cases_by_query: dict[str, set[str]] = defaultdict(set)
    for item in questions:
        cases_by_query[str(item["question"]).lower()].add(str(item["case_id"]))
    selected = [item for item in questions if str(item["qid"]) in selected_qids]
    ambiguous = [
        item
        for item in selected
        if len(cases_by_query[str(item["question"]).lower()]) > 1
    ]
    return {
        "question_count": len(selected),
        "unique_query_count": len({str(item["question"]).lower() for item in selected}),
        "ambiguous_question_rows": len(ambiguous),
        "ambiguous_question_rate": len(ambiguous) / len(selected),
        "by_question_type": {
            question_type: {
                "n": len(rows),
                "ambiguous_n": sum(
                    len(cases_by_query[str(item["question"]).lower()]) > 1
                    for item in rows
                ),
            }
            for question_type in sorted({str(item["question_type"]) for item in selected})
            for rows in [
                [item for item in selected if str(item["question_type"]) == question_type]
            ]
        },
    }


def mean(rows: list[dict], field: str) -> float:
    return statistics.mean(float(row[field]) for row in rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit research-validity risks and headroom.")
    parser.add_argument(
        "--qa",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_qa_seed_clean.json",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_cases.jsonl",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "openi_qa_grouped_case_seed7023.json",
    )
    parser.add_argument(
        "--adaptive-decisions",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "adaptive_retrieval"
        / "adaptive_policy_test_decisions.jsonl",
    )
    parser.add_argument(
        "--final-rows",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "final_test"
        / "final_optimized_test_rows.jsonl",
    )
    parser.add_argument(
        "--oracle-summary",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "oracle_test"
        / "final_optimized_test_summary.json",
    )
    parser.add_argument(
        "--statistics",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "statistics"
        / "held_out_test_grouped_bootstrap.json",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=ROOT / "data" / "raw" / "images" / "images_normalized",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "validity_audit"
        / "research_validity_audit.json",
    )
    args = parser.parse_args()

    questions = json.loads(args.qa.read_text(encoding="utf-8"))["questions"]
    question_by_qid = {str(item["qid"]): item for item in questions}
    cases = {str(item["case_id"]): item for item in read_jsonl(args.cases)}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    test_qids = set(str(value) for value in split["test"]["qids"])
    decisions = {str(item["qid"]): item for item in read_jsonl(args.adaptive_decisions)}
    final_rows = read_jsonl(args.final_rows)
    oracle = json.loads(args.oracle_summary.read_text(encoding="utf-8"))
    statistics_payload = json.loads(args.statistics.read_text(encoding="utf-8"))

    shortcut_by_type = {}
    for question_type in sorted({str(item["question_type"]) for item in questions}):
        rows = [item for item in questions if str(item["question_type"]) == question_type]
        bm25_exact = 0
        medcpt_title_exact = 0
        for item in rows:
            case = cases[str(item["case_id"])]
            source = (
                str(case.get("indication", ""))
                if "indication" in question_type
                else str(case.get("problems", ""))
            ).strip()
            report_text = str(case.get("report_text", ""))
            medcpt_title = " ".join(
                [str(case.get("indication", "")), str(case.get("problems", ""))]
            )
            bm25_exact += bool(source and source.lower() in report_text.lower())
            medcpt_title_exact += bool(source and source.lower() in medcpt_title.lower())
        shortcut_by_type[question_type] = {
            "n": len(rows),
            "source_exact_in_bm25_document_rate": bm25_exact / len(rows),
            "source_exact_in_medcpt_title_rate": medcpt_title_exact / len(rows),
        }

    condition_groups = {
        "correct_retrieval": [
            row for row in final_rows if bool(decisions[str(row["qid"])]["correct"])
        ],
        "wrong_retrieval": [
            row
            for row in final_rows
            if not bool(decisions[str(row["qid"])]["correct"])
            and not bool(decisions[str(row["qid"])]["abstained"])
        ],
        "retrieval_abstained": [
            row for row in final_rows if bool(decisions[str(row["qid"])]["abstained"])
        ],
    }
    conditioned = {
        label: {
            "n": len(rows),
            "draft_token_f1": mean(rows, "draft_token_f1"),
            "verified_token_f1": mean(rows, "final_token_f1"),
            "evidence_support_rate": mean(rows, "support_rate"),
            "agent_abstention_rate": mean(rows, "agent_abstained"),
        }
        for label, rows in condition_groups.items()
    }

    by_type = {}
    for question_type in sorted({str(row["question_type"]) for row in final_rows}):
        rows = [row for row in final_rows if str(row["question_type"]) == question_type]
        type_decisions = [decisions[str(row["qid"])] for row in rows]
        answered = [row for row in type_decisions if not bool(row["abstained"])]
        by_type[question_type] = {
            "n": len(rows),
            "strict_hit_at_1": sum(bool(row["correct"]) for row in type_decisions) / len(rows),
            "coverage": len(answered) / len(rows),
            "selective_accuracy": (
                sum(bool(row["correct"]) for row in answered) / len(answered)
                if answered
                else 0.0
            ),
            "verified_token_f1": mean(rows, "final_token_f1"),
        }

    target_pair = next(
        row
        for row in statistics_payload["pairwise"]
        if row["system_a"] == "final_adaptive_direct_semantic_agent"
        and row["system_b"] == "case_bm25_top1_semantic_agent"
    )
    image_files = list(args.image_root.rglob("*.png")) if args.image_root.exists() else []
    final_summary = json.loads(
        (args.final_rows.parent / "final_optimized_test_summary.json").read_text(encoding="utf-8")
    )
    output = {
        "benchmark_ambiguity": {
            "all": ambiguity_summary(questions, {str(item["qid"]) for item in questions}),
            "held_out_test": ambiguity_summary(questions, test_qids),
        },
        "query_document_shortcuts": shortcut_by_type,
        "performance_by_question_type": by_type,
        "verification_conditioned_on_retrieval": conditioned,
        "oracle_retrieval_headroom": {
            "actual_verified_token_f1": final_summary["verified_token_f1"],
            "oracle_verified_token_f1": oracle["verified_token_f1"],
            "absolute_gap": oracle["verified_token_f1"] - final_summary["verified_token_f1"],
            "oracle_to_actual_ratio": oracle["verified_token_f1"]
            / final_summary["verified_token_f1"],
        },
        "multiple_comparison_check": {
            "comparison": "final_vs_case_bm25_semantic_agent",
            "mean_difference": target_pair["mean_difference"],
            "paired_randomization_p": target_pair["paired_randomization_p"],
            "holm_adjusted_randomization_p": target_pair[
                "holm_adjusted_randomization_p"
            ],
        },
        "image_usage": {
            "local_image_file_count": len(image_files),
            "images_used_by_retrieval": False,
            "images_used_by_generator": False,
            "images_used_by_verifier": False,
            "classification": "linked-image case metadata; text-only modeled pipeline",
        },
        "priority_conclusion": "Improve benchmark identifiability and retrieval before scaling the generator.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
