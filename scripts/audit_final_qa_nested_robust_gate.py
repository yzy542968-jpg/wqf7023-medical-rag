"""Run the frozen nested-OOF support/margin audit for real B3/B6 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_final_qa_real_output_gate import (  # noqa: E402
    _matrix_from_sources,
    _option_f1,
    _question_ids,
    _row_exact,
    _selected_rows,
)
from evaluate_final_qa_full_role import (  # noqa: E402
    B3,
    B4,
    B6,
    CONDITIONS,
    build_matrices,
    cases_for_role,
    keyed_full,
)
from evaluate_final_qa_qlora_pilot import metrics, read_jsonl  # noqa: E402
from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy  # noqa: E402
from medical_rag.qa.question_vectorizer import RadReStructQuestionVectorizer  # noqa: E402
from medical_rag.qa.structured_metrics import (  # noqa: E402
    bootstrap_supported_macro_f1_difference,
    structured_qa_metrics,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _assignment(prefix: str, seed: int, case_id: str, folds: int) -> int:
    digest = hashlib.sha256(f"{prefix}|{seed}|{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % folds


def _inner_assignment(seed: int, outer_fold: int, case_id: str, folds: int) -> int:
    digest = hashlib.sha256(
        f"final-qa-robust-inner|{seed}|{outer_fold}|{case_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big") % folds


def _fit_utilities(
    keys: list[tuple[str, int]],
    qids: dict[tuple[str, int], int],
    rows_by_condition: dict[str, dict[tuple[str, int], dict[str, Any]]],
) -> dict[int, tuple[int, float]]:
    by_qid: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for key in keys:
        by_qid[qids[key]].append(key)
    utilities: dict[int, tuple[int, float]] = {}
    for question_id, question_keys in by_qid.items():
        b3_rows = [rows_by_condition[B3][key] for key in question_keys]
        b6_rows = [rows_by_condition[B6][key] for key in question_keys]
        utilities[question_id] = (
            len(question_keys),
            _option_f1(b6_rows) - _option_f1(b3_rows),
        )
    return utilities


def _apply_utility_policy(
    keys: list[tuple[str, int]],
    qids: dict[tuple[str, int], int],
    utilities: dict[int, tuple[int, float]],
    *,
    minimum_support: int,
    minimum_margin: float,
) -> dict[tuple[str, int], str]:
    output: dict[tuple[str, int], str] = {}
    for key in keys:
        support, margin = utilities.get(qids[key], (0, -np.inf))
        output[key] = (
            B6
            if support >= minimum_support and margin >= minimum_margin
            else B3
        )
    return output


def _evaluate_subset(
    *,
    source: dict[tuple[str, int], str],
    subset_keys: list[tuple[str, int]],
    subset_case_mask: np.ndarray,
    rows_by_condition: dict[str, dict[tuple[str, int], dict[str, Any]]],
    cases: list[Any],
    vectorizer: RadReStructQuestionVectorizer,
    targets: np.ndarray,
) -> dict[str, Any]:
    full_source = {key: B3 for key in rows_by_condition[B3]}
    full_source.update(source)
    matrix = _matrix_from_sources(
        cases=cases,
        source_by_key=full_source,
        rows_by_condition=rows_by_condition,
        vectorizer=vectorizer,
    )
    selected_rows = [
        rows_by_condition[full_source[key]][key] for key in subset_keys
    ]
    return {
        "structured": structured_qa_metrics(
            targets[subset_case_mask], matrix[subset_case_mask]
        ).as_dict(),
        "question": metrics(selected_rows),
        "history_row_count": sum(full_source[key] == B6 for key in subset_keys),
        "source": full_source,
        "matrix": matrix,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_json(args.config)
    manifest = _load_json(args.manifest)
    rows = read_jsonl(args.rows)
    rows_by_condition = {
        condition: keyed_full(rows, condition) for condition in CONDITIONS
    }
    if len({frozenset(values) for values in rows_by_condition.values()}) != 1:
        raise RuntimeError("Cached real-output conditions do not align")

    hierarchy = RadReStructHierarchy(args.radrestruct_root)
    vectorizer = RadReStructQuestionVectorizer(hierarchy)
    cases = cases_for_role(args.radrestruct_root, manifest, "validation")
    targets, predictions = build_matrices(
        role_cases=cases,
        rows_by_condition=rows_by_condition,
        vectorizer=vectorizer,
    )
    qids = _question_ids(cases, vectorizer)
    all_keys = sorted(qids)
    seed = int(config["seed"])
    outer_folds = int(config["outer_folds"])
    inner_folds = int(config["inner_folds"])
    outer_by_case = {
        case.case_id: _assignment(
            "final-qa-robust-outer", seed, case.case_id, outer_folds
        )
        for case in cases
    }
    case_index = {case.case_id: index for index, case in enumerate(cases)}
    support_grid = [int(value) for value in config["minimum_support_grid"]]
    margin_grid = [float(value) for value in config["minimum_macro_margin_grid"]]
    exact_floor = float(
        config["inner_selection"]["constraints"][
            "question_exact_delta_minimum"
        ]
    )
    micro_floor = float(
        config["inner_selection"]["constraints"][
            "option_micro_f1_delta_minimum"
        ]
    )

    final_source: dict[tuple[str, int], str] = {}
    outer_records: list[dict[str, Any]] = []
    for outer_fold in range(outer_folds):
        outer_keys = [
            key for key in all_keys if outer_by_case[key[0]] == outer_fold
        ]
        training_keys = [
            key for key in all_keys if outer_by_case[key[0]] != outer_fold
        ]
        training_case_mask = np.asarray(
            [outer_by_case[case.case_id] != outer_fold for case in cases],
            dtype=bool,
        )
        inner_by_case = {
            case.case_id: _inner_assignment(
                seed, outer_fold, case.case_id, inner_folds
            )
            for case in cases
            if outer_by_case[case.case_id] != outer_fold
        }
        inner_utilities: dict[int, dict[int, tuple[int, float]]] = {}
        inner_apply_keys: dict[int, list[tuple[str, int]]] = {}
        for inner_fold in range(inner_folds):
            fit_keys = [
                key
                for key in training_keys
                if inner_by_case[key[0]] != inner_fold
            ]
            apply_keys = [
                key
                for key in training_keys
                if inner_by_case[key[0]] == inner_fold
            ]
            inner_utilities[inner_fold] = _fit_utilities(
                fit_keys, qids, rows_by_condition
            )
            inner_apply_keys[inner_fold] = apply_keys

        b3_training_rows = [rows_by_condition[B3][key] for key in training_keys]
        b3_training_question = metrics(b3_training_rows)
        candidate_rows: list[dict[str, Any]] = []
        for minimum_support in support_grid:
            for minimum_margin in margin_grid:
                inner_source: dict[tuple[str, int], str] = {}
                for inner_fold in range(inner_folds):
                    inner_source.update(
                        _apply_utility_policy(
                            inner_apply_keys[inner_fold],
                            qids,
                            inner_utilities[inner_fold],
                            minimum_support=minimum_support,
                            minimum_margin=minimum_margin,
                        )
                    )
                evaluated = _evaluate_subset(
                    source=inner_source,
                    subset_keys=training_keys,
                    subset_case_mask=training_case_mask,
                    rows_by_condition=rows_by_condition,
                    cases=cases,
                    vectorizer=vectorizer,
                    targets=targets,
                )
                exact_delta = float(
                    evaluated["question"]["exact_answer_set_accuracy"]
                    - b3_training_question["exact_answer_set_accuracy"]
                )
                micro_delta = float(
                    evaluated["question"]["option_micro_f1"]
                    - b3_training_question["option_micro_f1"]
                )
                candidate_rows.append(
                    {
                        "minimum_support": minimum_support,
                        "minimum_margin": minimum_margin,
                        "inner_oof_macro_f1": float(
                            evaluated["structured"][
                                "supported_label_macro_f1"
                            ]
                        ),
                        "inner_oof_exact": float(
                            evaluated["question"]["exact_answer_set_accuracy"]
                        ),
                        "inner_oof_option_micro_f1": float(
                            evaluated["question"]["option_micro_f1"]
                        ),
                        "inner_oof_history_row_count": int(
                            evaluated["history_row_count"]
                        ),
                        "admissible": exact_delta >= exact_floor
                        and micro_delta >= micro_floor,
                    }
                )
        admissible = [row for row in candidate_rows if row["admissible"]]
        pool = admissible or candidate_rows
        selected = min(
            pool,
            key=lambda row: (
                -float(row["inner_oof_macro_f1"]),
                -float(row["inner_oof_exact"]),
                -float(row["inner_oof_option_micro_f1"]),
                -float(row["minimum_margin"]),
                -int(row["minimum_support"]),
            ),
        )
        outer_utilities = _fit_utilities(training_keys, qids, rows_by_condition)
        outer_source = _apply_utility_policy(
            outer_keys,
            qids,
            outer_utilities,
            minimum_support=int(selected["minimum_support"]),
            minimum_margin=float(selected["minimum_margin"]),
        )
        final_source.update(outer_source)
        outer_records.append(
            {
                "outer_fold": outer_fold,
                "training_case_count": int(training_case_mask.sum()),
                "evaluation_case_count": int((~training_case_mask).sum()),
                "selected_inner_policy": selected,
                "admissible_grid_count": len(admissible),
                "grid_count": len(candidate_rows),
                "outer_history_row_count": sum(
                    value == B6 for value in outer_source.values()
                ),
            }
        )

    if set(final_source) != set(all_keys):
        raise RuntimeError("Nested OOF source policy is incomplete")
    final_matrix = _matrix_from_sources(
        cases=cases,
        source_by_key=final_source,
        rows_by_condition=rows_by_condition,
        vectorizer=vectorizer,
    )
    selected_rows = _selected_rows(final_source, rows_by_condition)
    selected_question = metrics(selected_rows)
    selected_structured = structured_qa_metrics(targets, final_matrix).as_dict()
    b3_question = metrics(rows_by_condition[B3].values())
    b3_structured = structured_qa_metrics(targets, predictions[B3]).as_dict()
    b4_structured = structured_qa_metrics(targets, predictions[B4]).as_dict()
    disagreements = [
        key
        for key, source in final_source.items()
        if source == B6
        and set(rows_by_condition[B3][key]["predicted_indices"])
        != set(rows_by_condition[B6][key]["predicted_indices"])
    ]
    recovery = sum(
        _row_exact(rows_by_condition[B6][key])
        and not _row_exact(rows_by_condition[B3][key])
        for key in disagreements
    )
    harm = sum(
        _row_exact(rows_by_condition[B3][key])
        and not _row_exact(rows_by_condition[B6][key])
        for key in disagreements
    )
    macro_delta = float(
        selected_structured["supported_label_macro_f1"]
        - b3_structured["supported_label_macro_f1"]
    )
    exact_delta = float(
        selected_question["exact_answer_set_accuracy"]
        - b3_question["exact_answer_set_accuracy"]
    )
    micro_delta = float(
        selected_question["option_micro_f1"] - b3_question["option_micro_f1"]
    )
    checks = {
        "nested_oof_macro_exceeds_b3": macro_delta > 0,
        "nested_oof_exact_noninferior": exact_delta >= -0.001,
        "nested_oof_option_micro_noninferior": micro_delta >= -0.001,
        "nested_oof_macro_exceeds_random": float(
            selected_structured["supported_label_macro_f1"]
        )
        > float(b4_structured["supported_label_macro_f1"]),
        "history_used": len(disagreements) > 0,
    }
    advanced = all(checks.values())
    summary = {
        "study": config["study"],
        "status": "go_for_confirmation_design" if advanced else "branch_closed",
        "boundary": config["boundary"],
        "test_accessed": False,
        "new_generation_performed": False,
        "case_count": len(cases),
        "question_count": len(all_keys),
        "outer_fold_case_counts": {
            str(fold): sum(value == fold for value in outer_by_case.values())
            for fold in range(outer_folds)
        },
        "outer_fold_selection": outer_records,
        "baselines": {
            "image_only": {
                "structured": b3_structured,
                "question": b3_question,
            },
            "random_history": {
                "structured": b4_structured,
                "question": metrics(rows_by_condition[B4].values()),
            },
        },
        "nested_oof": {
            "structured": selected_structured,
            "question": selected_question,
            "history_selected_row_count": sum(
                value == B6 for value in final_source.values()
            ),
            "history_selected_disagreement_count": len(disagreements),
            "history_only_recovery_count": int(recovery),
            "image_correct_to_history_wrong_count": int(harm),
            "minus_image_only": {
                "supported_label_macro_f1": macro_delta,
                "question_exact": exact_delta,
                "option_micro_f1": micro_delta,
            },
            "case_bootstrap_macro_f1_vs_b3": bootstrap_supported_macro_f1_difference(
                targets,
                final_matrix,
                predictions[B3],
                samples=5000,
                seed=seed + 1,
            ),
        },
        "advancement": {"passed": advanced, **checks},
        "interpretation": (
            "The nested-OOF conservative gate passed all development conditions. "
            "A separate final-fit and confirmation protocol may now be considered; "
            "Test remains locked."
            if advanced
            else "The nested-OOF audit failed at least one fixed condition. The "
            "Final-QA historical-source optimization branch is closed and Test "
            "remains locked."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/final_qa_nested_robust_gate.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json",
    )
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/final_qa_validation_rows.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/final_qa_nested_robust_gate.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "nested_oof_minus_image_only": result["nested_oof"][
                    "minus_image_only"
                ],
                "advancement": result["advancement"],
                "outer_fold_selection": result["outer_fold_selection"],
            },
            indent=2,
            sort_keys=True,
        )
    )
