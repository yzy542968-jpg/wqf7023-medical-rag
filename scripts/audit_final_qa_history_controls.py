"""Audit Calibration history controls and post-hoc answer concordance."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_final_qa_history_policy import negative_transfer
from evaluate_final_qa_qlora_pilot import bootstrap_difference, keyed, metrics, read_jsonl

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.qa.radrestruct import iter_radrestruct_cases  # noqa: E402
from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy  # noqa: E402
from medical_rag.qa.question_vectorizer import RadReStructQuestionVectorizer  # noqa: E402


CONDITIONS = {
    "no_history": "b3_no_history_r2",
    "random_whole_report": "b4_deterministic_random_history",
    "top1_image_whole_report": "b6_top1_image_neighbor_whole_report",
    "top3_image_question_facts": "p1_top3_image_neighbors_question_conditioned_evidence",
    "top3_v12_question_facts": "p1_v12_lambdamart_top3_question_conditioned_evidence",
}


def answer_map(
    root: Path,
) -> tuple[
    dict[str, dict[int, frozenset[str]]],
    dict[tuple[str, int], tuple[int, Any]],
]:
    vectorizer = RadReStructQuestionVectorizer(RadReStructHierarchy(root))
    by_question_id: dict[str, dict[int, frozenset[str]]] = {}
    by_index: dict[tuple[str, int], tuple[int, Any]] = {}
    for case in iter_radrestruct_cases(root):
        question_ids = vectorizer.question_ids(case.questions)
        answers_by_id: dict[int, frozenset[str]] = {}
        for index, (question_id, question) in enumerate(
            zip(question_ids, case.questions, strict=True)
        ):
            by_index[(case.case_id, index)] = (question_id, question)
            answers = frozenset(" ".join(value.lower().split()) for value in question.answers)
            if question_id in answers_by_id:
                raise RuntimeError(f"Duplicate official question ID in {case.case_id}: {question_id}")
            answers_by_id[question_id] = answers
        by_question_id[case.case_id] = answers_by_id
    return by_question_id, by_index


def concordance(
    rows: dict[tuple[str, int], dict[str, Any]],
    by_question_id: dict[str, dict[int, frozenset[str]]],
    by_index: dict[tuple[str, int], tuple[int, Any]],
) -> dict[str, float | int | None]:
    question_count = len(rows)
    with_history = matched_questions = any_exact = 0
    matched_pairs = exact_pairs = 0
    jaccards: list[float] = []
    for key, row in rows.items():
        history_ids = [str(value) for value in row.get("evidence_case_ids", [])]
        if history_ids:
            with_history += 1
        question_id, target = by_index[key]
        target_answers = frozenset(" ".join(value.lower().split()) for value in target.answers)
        question_matched = False
        question_exact = False
        for history_id in history_ids:
            history_answers = by_question_id.get(history_id, {}).get(question_id)
            if history_answers is None:
                continue
            question_matched = True
            matched_pairs += 1
            intersection = len(target_answers & history_answers)
            union = len(target_answers | history_answers)
            jaccards.append(intersection / union if union else 1.0)
            if target_answers == history_answers:
                exact_pairs += 1
                question_exact = True
        matched_questions += int(question_matched)
        any_exact += int(question_exact)
    return {
        "question_count": question_count,
        "questions_with_history": with_history,
        "same_path_coverage_rate": matched_questions / question_count,
        "any_exact_answer_concordance_rate": any_exact / question_count,
        "matched_history_case_question_pairs": matched_pairs,
        "exact_pair_concordance_rate": exact_pairs / matched_pairs if matched_pairs else None,
        "mean_answer_jaccard": float(np.mean(jaccards)) if jaccards else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.rows)
    by_question_id, by_index = answer_map(args.radrestruct_root)
    selected = {label: keyed(rows, condition) for label, condition in CONDITIONS.items()}
    if len({frozenset(value) for value in selected.values()}) != 1:
        raise RuntimeError("History-control conditions use different Calibration rows")
    baseline = selected["no_history"]
    systems: dict[str, Any] = {}
    for label, condition_rows in selected.items():
        record: dict[str, Any] = {"metrics": metrics(condition_rows.values())}
        if label != "no_history":
            record["negative_transfer_from_no_history"] = negative_transfer(
                baseline, condition_rows
            )
            record["case_grouped_bootstrap_vs_no_history"] = bootstrap_difference(
                condition_rows, baseline
            )
            record["posthoc_gold_history_concordance"] = concordance(
                condition_rows, by_question_id, by_index
            )
        systems[label] = record
    random_rows = selected["random_whole_report"]
    comparisons = {
        label + "_minus_random": {
            "option_micro_f1": float(metrics(system.values())["option_micro_f1"])
            - float(metrics(random_rows.values())["option_micro_f1"]),
            "exact_answer_set_accuracy": float(
                metrics(system.values())["exact_answer_set_accuracy"]
            )
            - float(metrics(random_rows.values())["exact_answer_set_accuracy"]),
            "case_grouped_bootstrap": bootstrap_difference(system, random_rows),
        }
        for label, system in selected.items()
        if label not in {"no_history", "random_whole_report"}
    }
    summary = {
        "study": "Final QA Calibration history-control and concordance audit",
        "systems": systems,
        "comparisons": comparisons,
        "interpretation_boundary": (
            "Gold answer concordance is post-hoc analysis only and was unavailable to "
            "retrieval and generation. Results do not establish clinical correctness."
        ),
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "experiments/final_qa_development/final_qa_qlora_384_calibration_rows.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/final_qa_development/final_qa_history_controls_audit.json",
    )
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
