"""Evaluate complete Final-QA role outputs in the official report space."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_final_qa_history_policy import negative_transfer  # noqa: E402
from evaluate_final_qa_qlora_pilot import metrics, read_jsonl  # noqa: E402
from medical_rag.qa.radrestruct import iter_radrestruct_cases  # noqa: E402
from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy  # noqa: E402
from medical_rag.qa.question_vectorizer import RadReStructQuestionVectorizer  # noqa: E402
from medical_rag.qa.structured_metrics import (  # noqa: E402
    bootstrap_supported_macro_f1_difference,
    fit_label_majority,
    repeat_prediction,
    structured_qa_metrics,
)


B3 = "b3_no_history_r2"
B4 = "b4_deterministic_random_history"
B6 = "b6_top1_image_neighbor_whole_report"
P1 = "p1_top3_image_neighbors_question_conditioned_evidence"
CONDITIONS = (B3, B4, B6, P1)


def keyed_full(
    rows: list[dict[str, Any]], condition: str
) -> dict[tuple[str, int], dict[str, Any]]:
    selected = {
        (str(row["case_id"]), int(row["question_index"])): row
        for row in rows
        if row["condition"] == condition
    }
    matching_count = sum(row["condition"] == condition for row in rows)
    if len(selected) != matching_count:
        raise RuntimeError(f"Duplicate full-role keys in condition {condition}")
    return selected


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def cases_for_role(
    root: Path, manifest: dict[str, Any], role: str
) -> list[Any]:
    role_ids = {str(row["case_id"]) for row in manifest["roles"][role]["cases"]}
    cases = [case for case in iter_radrestruct_cases(root) if case.case_id in role_ids]
    cases.sort(key=lambda case: case.case_id)
    if {case.case_id for case in cases} != role_ids:
        raise RuntimeError(f"Rad-ReStruct cases do not cover manifest role {role}")
    return cases


def aggregate_binary_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    target = np.asarray(targets, dtype=bool)
    predicted = np.asarray(predictions, dtype=bool)
    tp = int(np.logical_and(target, predicted).sum())
    tn = int(np.logical_and(~target, ~predicted).sum())
    fp = int(np.logical_and(~target, predicted).sum())
    fn = int(np.logical_and(target, ~predicted).sum())
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "positive_recall": sensitivity,
        "negative_specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
    }


def row_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = metrics(rows)
    result.update(
        {
            "mean_input_tokens": float(np.mean([row["input_tokens"] for row in rows])),
            "mean_output_tokens": float(np.mean([row["output_tokens"] for row in rows])),
            "mean_evidence_units": float(
                np.mean([row.get("evidence_unit_count", 0) for row in rows])
            ),
            "provenance_complete_rate": float(
                np.mean([bool(row.get("provenance_complete", False)) for row in rows])
            ),
        }
    )
    return result


def build_matrices(
    *,
    role_cases: list[Any],
    rows_by_condition: dict[str, dict[tuple[str, int], dict[str, Any]]],
    vectorizer: RadReStructQuestionVectorizer,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    target_rows: list[np.ndarray] = []
    prediction_rows: dict[str, list[np.ndarray]] = {condition: [] for condition in CONDITIONS}
    for case in role_cases:
        target_rows.append(vectorizer.vectorize_answers(case.questions))
        for condition in CONDITIONS:
            answers: list[list[str]] = []
            for question_index, question in enumerate(case.questions):
                row = rows_by_condition[condition][(case.case_id, question_index)]
                indices = [int(value) for value in row["predicted_indices"]]
                answers.append(
                    [question.options[index] for index in indices if 0 <= index < len(question.options)]
                )
            prediction_rows[condition].append(
                vectorizer.vectorize_answers(case.questions, answers)
            )
    return np.stack(target_rows), {
        condition: np.stack(values) for condition, values in prediction_rows.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_json(args.config)
    manifest = load_json(args.manifest)
    rows = read_jsonl(args.rows)
    expected_questions = int(config["expected_question_count"])
    rows_by_condition = {
        condition: keyed_full(rows, condition) for condition in CONDITIONS
    }
    if any(len(value) != expected_questions for value in rows_by_condition.values()):
        counts = {key: len(value) for key, value in rows_by_condition.items()}
        raise RuntimeError(f"Full-role rows are incomplete: {counts}")
    key_sets = {frozenset(value) for value in rows_by_condition.values()}
    if len(key_sets) != 1:
        raise RuntimeError("Full-role conditions do not use identical questions")

    hierarchy = RadReStructHierarchy(args.radrestruct_root)
    vectorizer = RadReStructQuestionVectorizer(hierarchy)
    validation_cases = cases_for_role(
        args.radrestruct_root, manifest, str(config["role"]).lower()
    )
    train_cases = cases_for_role(args.radrestruct_root, manifest, "train")
    targets, predictions = build_matrices(
        role_cases=validation_cases,
        rows_by_condition=rows_by_condition,
        vectorizer=vectorizer,
    )
    train_targets = np.stack(
        [vectorizer.vectorize_answers(case.questions) for case in train_cases]
    )
    majority = repeat_prediction(fit_label_majority(train_targets), len(targets))
    systems: dict[str, Any] = {
        "b0_train_label_majority": {
            "structured": structured_qa_metrics(targets, majority).as_dict(),
            "aggregate_binary": aggregate_binary_metrics(targets, majority),
        }
    }
    for condition in CONDITIONS:
        condition_rows = list(rows_by_condition[condition].values())
        systems[condition] = {
            "structured": structured_qa_metrics(targets, predictions[condition]).as_dict(),
            "aggregate_binary": aggregate_binary_metrics(targets, predictions[condition]),
            "question_level": row_summary(condition_rows),
        }
        if condition != B3:
            systems[condition]["negative_transfer_from_b3"] = negative_transfer(
                rows_by_condition[B3], rows_by_condition[condition]
            )
            systems[condition]["case_bootstrap_structured_vs_b3"] = (
                bootstrap_supported_macro_f1_difference(
                    targets,
                    predictions[condition],
                    predictions[B3],
                    samples=int(config["statistics"]["bootstrap_replicates"]),
                    seed=int(config["statistics"]["seed"]),
                )
            )

    candidate_order = sorted(
        (B6, P1),
        key=lambda condition: (
            -float(systems[condition]["structured"]["supported_label_macro_f1"]),
            float(systems[condition]["question_level"]["mean_input_tokens"]),
            0 if condition == B6 else 1,
        ),
    )
    selected = candidate_order[0]
    selected_system = systems[selected]
    b3_system = systems[B3]
    advanced = (
        float(selected_system["structured"]["supported_label_macro_f1"])
        > float(b3_system["structured"]["supported_label_macro_f1"])
        and float(selected_system["question_level"]["option_micro_f1"])
        > float(b3_system["question_level"]["option_micro_f1"])
        and float(selected_system["question_level"]["contract_valid_rate"])
        >= float(b3_system["question_level"]["contract_valid_rate"]) - 0.010
    )
    summary = {
        "study": config["study"],
        "status": "full_validation_complete_no_test_access",
        "role": config["role"],
        "case_count": len(validation_cases),
        "question_count": expected_questions,
        "label_count": targets.shape[1],
        "systems": systems,
        "selected_meaningful_history_policy": selected,
        "prespecified_validation_advancement_rule_passed": advanced,
        "random_history_control_exceeds_selected": (
            float(systems[B4]["structured"]["supported_label_macro_f1"])
            > float(systems[selected]["structured"]["supported_label_macro_f1"])
        ),
        "boundary": config["boundary"],
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/final_qa_validation.json"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json",
    )
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
