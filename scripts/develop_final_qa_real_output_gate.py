"""Cross-fit a question-conditional gate over cached B3/B6 real outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_final_qa_v2_selective_gate import GateFeatureEncoder  # noqa: E402
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


def _embedding_map(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        case_ids = [str(value) for value in payload["case_ids"]]
        values = np.asarray(payload["case_image_embeddings"], dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(values).all():
        raise ValueError("Image embeddings are invalid")
    return dict(zip(case_ids, values / norms, strict=True))


def _fold(case_id: str, seed: int, folds: int) -> int:
    digest = hashlib.sha256(
        f"final-qa-real-output-gate|{seed}|{case_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big") % folds


def _option_f1(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    option_count = int(rows[0]["option_count"])
    values: list[float] = []
    for option in range(option_count):
        target = np.asarray(
            [option in set(int(x) for x in row["gold_indices"]) for row in rows]
        )
        if not target.any():
            continue
        predicted = np.asarray(
            [option in set(int(x) for x in row["predicted_indices"]) for row in rows]
        )
        tp = int(np.logical_and(target, predicted).sum())
        fp = int(np.logical_and(~target, predicted).sum())
        fn = int(np.logical_and(target, ~predicted).sum())
        denominator = 2 * tp + fp + fn
        values.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(values)) if values else 0.0


def _row_exact(row: dict[str, Any]) -> bool:
    return set(int(value) for value in row["gold_indices"]) == set(
        int(value) for value in row["predicted_indices"]
    )


def _question_ids(
    cases: list[Any], vectorizer: RadReStructQuestionVectorizer
) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for case in cases:
        for question_index, question_id in enumerate(
            vectorizer.question_ids(case.questions)
        ):
            result[(case.case_id, question_index)] = int(question_id)
    return result


def _matrix_from_sources(
    *,
    cases: list[Any],
    source_by_key: dict[tuple[str, int], str],
    rows_by_condition: dict[str, dict[tuple[str, int], dict[str, Any]]],
    vectorizer: RadReStructQuestionVectorizer,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for case in cases:
        answers: list[list[str]] = []
        for question_index, question in enumerate(case.questions):
            key = (case.case_id, question_index)
            row = rows_by_condition[source_by_key[key]][key]
            answers.append(
                [
                    question.options[int(index)]
                    for index in row["predicted_indices"]
                    if 0 <= int(index) < len(question.options)
                ]
            )
        rows.append(vectorizer.vectorize_answers(case.questions, answers))
    return np.stack(rows)


def _selected_rows(
    source_by_key: dict[tuple[str, int], str],
    rows_by_condition: dict[str, dict[tuple[str, int], dict[str, Any]]],
) -> list[dict[str, Any]]:
    return [
        rows_by_condition[source][key]
        for key, source in sorted(source_by_key.items())
    ]


def _question_policy(
    *,
    qids: dict[tuple[str, int], int],
    rows_by_condition: dict[str, dict[tuple[str, int], dict[str, Any]]],
    case_fold: dict[str, int],
    fold: int,
    objective: str,
) -> dict[tuple[str, int], str]:
    train_keys_by_qid: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for key, question_id in qids.items():
        if case_fold[key[0]] != fold:
            train_keys_by_qid[question_id].append(key)
    source_by_qid: dict[int, str] = {}
    for question_id, keys in train_keys_by_qid.items():
        b3_rows = [rows_by_condition[B3][key] for key in keys]
        b6_rows = [rows_by_condition[B6][key] for key in keys]
        if objective == "exact":
            b3_score = float(np.mean([_row_exact(row) for row in b3_rows]))
            b6_score = float(np.mean([_row_exact(row) for row in b6_rows]))
        elif objective == "macro":
            b3_score = _option_f1(b3_rows)
            b6_score = _option_f1(b6_rows)
        else:
            raise ValueError(f"Unsupported question policy objective: {objective}")
        source_by_qid[question_id] = B6 if b6_score > b3_score else B3
    return {
        key: source_by_qid.get(question_id, B3)
        for key, question_id in qids.items()
        if case_fold[key[0]] == fold
    }


def _gate_numeric(
    keys: list[tuple[str, int]],
    rows_by_condition: dict[str, dict[tuple[str, int], dict[str, Any]]],
    embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    values: list[list[float]] = []
    for key in keys:
        b3 = rows_by_condition[B3][key]
        b6 = rows_by_condition[B6][key]
        history_ids = [str(value) for value in b6.get("evidence_case_ids", [])]
        if not history_ids or history_ids[0] not in embeddings:
            similarity = -1.0
        else:
            similarity = float(embeddings[key[0]] @ embeddings[history_ids[0]])
        image = set(int(value) for value in b3["predicted_indices"])
        history = set(int(value) for value in b6["predicted_indices"])
        union = image | history
        values.append(
            [
                similarity,
                len(image & history) / len(union) if union else 1.0,
                float(len(image)),
                float(len(history)),
                float(b3["option_count"]),
            ]
        )
    return np.asarray(values, dtype=np.float32)


def _logistic_fold_policy(
    *,
    fold: int,
    all_keys: list[tuple[str, int]],
    qids: dict[tuple[str, int], int],
    rows_by_condition: dict[str, dict[tuple[str, int], dict[str, Any]]],
    case_fold: dict[str, int],
    embeddings: dict[str, np.ndarray],
    hierarchy: RadReStructHierarchy,
    cases: list[Any],
    vectorizer: RadReStructQuestionVectorizer,
    targets: np.ndarray,
    predictions: dict[str, np.ndarray],
    threshold_grid: list[float],
    seed: int,
) -> tuple[dict[tuple[str, int], str], dict[str, Any]]:
    train_keys = [key for key in all_keys if case_fold[key[0]] != fold]
    test_keys = [key for key in all_keys if case_fold[key[0]] == fold]
    disagreement_train = [
        key
        for key in train_keys
        if set(rows_by_condition[B3][key]["predicted_indices"])
        != set(rows_by_condition[B6][key]["predicted_indices"])
    ]
    labels = np.asarray(
        [
            int(
                _row_exact(rows_by_condition[B6][key])
                and not _row_exact(rows_by_condition[B3][key])
            )
            for key in disagreement_train
        ],
        dtype=np.uint8,
    )
    answer_types = tuple(
        sorted(
            {
                str(rows_by_condition[B3][key]["answer_type"])
                for key in train_keys
            }
        )
    )
    encoder = GateFeatureEncoder(len(hierarchy.indices_by_question), answer_types)
    numeric_train = _gate_numeric(disagreement_train, rows_by_condition, embeddings)
    encoder.fit(numeric_train)
    train_x = encoder.transform(
        numeric_train,
        np.asarray([qids[key] for key in disagreement_train]),
        np.asarray(
            [rows_by_condition[B3][key]["answer_type"] for key in disagreement_train]
        ),
    )
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed + fold,
        solver="lbfgs",
    ).fit(train_x, labels)
    train_probability = model.predict_proba(train_x)[:, 1]

    case_index = {case.case_id: index for index, case in enumerate(cases)}
    train_case_mask = np.asarray(
        [case_fold[case.case_id] != fold for case in cases], dtype=bool
    )
    threshold_candidates: list[dict[str, Any]] = []
    for threshold in threshold_grid:
        source = {key: B3 for key in all_keys}
        for key, probability in zip(
            disagreement_train, train_probability, strict=True
        ):
            if float(probability) >= float(threshold):
                source[key] = B6
        matrix = _matrix_from_sources(
            cases=cases,
            source_by_key=source,
            rows_by_condition=rows_by_condition,
            vectorizer=vectorizer,
        )
        structured = structured_qa_metrics(
            targets[train_case_mask], matrix[train_case_mask]
        ).as_dict()
        selected = _selected_rows(source, rows_by_condition)
        selected_train = [
            row for row in selected if case_fold[str(row["case_id"])] != fold
        ]
        question = metrics(selected_train)
        b3_train = [rows_by_condition[B3][key] for key in train_keys]
        b3_metrics = metrics(b3_train)
        admissible = (
            float(question["exact_answer_set_accuracy"])
            >= float(b3_metrics["exact_answer_set_accuracy"]) - 0.001
            and float(question["option_micro_f1"])
            >= float(b3_metrics["option_micro_f1"]) - 0.001
        )
        threshold_candidates.append(
            {
                "threshold": float(threshold),
                "admissible": admissible,
                "macro_f1": float(structured["supported_label_macro_f1"]),
                "exact": float(question["exact_answer_set_accuracy"]),
                "option_micro_f1": float(question["option_micro_f1"]),
            }
        )
    admissible_rows = [row for row in threshold_candidates if row["admissible"]]
    pool = admissible_rows or threshold_candidates
    selected_threshold = min(
        pool,
        key=lambda row: (
            -float(row["macro_f1"]),
            -float(row["exact"]),
            -float(row["option_micro_f1"]),
            -float(row["threshold"]),
        ),
    )

    disagreement_test = [
        key
        for key in test_keys
        if set(rows_by_condition[B3][key]["predicted_indices"])
        != set(rows_by_condition[B6][key]["predicted_indices"])
    ]
    test_source = {key: B3 for key in test_keys}
    if disagreement_test:
        numeric_test = _gate_numeric(disagreement_test, rows_by_condition, embeddings)
        test_probability = model.predict_proba(
            encoder.transform(
                numeric_test,
                np.asarray([qids[key] for key in disagreement_test]),
                np.asarray(
                    [
                        rows_by_condition[B3][key]["answer_type"]
                        for key in disagreement_test
                    ]
                ),
            )
        )[:, 1]
        for key, probability in zip(disagreement_test, test_probability, strict=True):
            if float(probability) >= float(selected_threshold["threshold"]):
                test_source[key] = B6
    return test_source, {
        "fold": fold,
        "train_case_count": int(train_case_mask.sum()),
        "evaluation_case_count": int((~train_case_mask).sum()),
        "disagreement_training_count": len(disagreement_train),
        "history_better_training_count": int(labels.sum()),
        "selected_threshold": selected_threshold,
    }


def _policy_summary(
    *,
    name: str,
    source: dict[tuple[str, int], str],
    rows_by_condition: dict[str, dict[tuple[str, int], dict[str, Any]]],
    cases: list[Any],
    vectorizer: RadReStructQuestionVectorizer,
    targets: np.ndarray,
    b3_matrix: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    matrix = _matrix_from_sources(
        cases=cases,
        source_by_key=source,
        rows_by_condition=rows_by_condition,
        vectorizer=vectorizer,
    )
    question_rows = _selected_rows(source, rows_by_condition)
    question = metrics(question_rows)
    structured = structured_qa_metrics(targets, matrix).as_dict()
    history_disagreements = [
        key
        for key, selected in source.items()
        if selected == B6
        and set(rows_by_condition[B3][key]["predicted_indices"])
        != set(rows_by_condition[B6][key]["predicted_indices"])
    ]
    recovery = sum(_row_exact(rows_by_condition[B6][key]) and not _row_exact(rows_by_condition[B3][key]) for key in history_disagreements)
    harm = sum(_row_exact(rows_by_condition[B3][key]) and not _row_exact(rows_by_condition[B6][key]) for key in history_disagreements)
    return {
        "name": name,
        "structured": structured,
        "question": question,
        "history_selected_row_count": sum(value == B6 for value in source.values()),
        "history_selected_disagreement_count": len(history_disagreements),
        "history_only_recovery_count": int(recovery),
        "image_correct_to_history_wrong_count": int(harm),
        "case_bootstrap_macro_f1_vs_b3": bootstrap_supported_macro_f1_difference(
            targets,
            matrix,
            b3_matrix,
            samples=5000,
            seed=7054,
        ),
    }, matrix


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
    targets, base_predictions = build_matrices(
        role_cases=cases,
        rows_by_condition=rows_by_condition,
        vectorizer=vectorizer,
    )
    qids = _question_ids(cases, vectorizer)
    all_keys = sorted(qids)
    folds = int(config["cross_validation"]["folds"])
    seed = int(config["seed"])
    case_fold = {case.case_id: _fold(case.case_id, seed, folds) for case in cases}
    fold_counts = {
        str(fold): sum(value == fold for value in case_fold.values())
        for fold in range(folds)
    }
    embeddings = _embedding_map(args.embeddings)

    sources: dict[str, dict[tuple[str, int], str]] = {
        "question_id_exact_utility": {},
        "question_id_macro_f1_utility": {},
        "logistic_disagreement_gate": {},
    }
    logistic_fold_records: list[dict[str, Any]] = []
    for fold in range(folds):
        sources["question_id_exact_utility"].update(
            _question_policy(
                qids=qids,
                rows_by_condition=rows_by_condition,
                case_fold=case_fold,
                fold=fold,
                objective="exact",
            )
        )
        sources["question_id_macro_f1_utility"].update(
            _question_policy(
                qids=qids,
                rows_by_condition=rows_by_condition,
                case_fold=case_fold,
                fold=fold,
                objective="macro",
            )
        )
        fold_source, fold_record = _logistic_fold_policy(
            fold=fold,
            all_keys=all_keys,
            qids=qids,
            rows_by_condition=rows_by_condition,
            case_fold=case_fold,
            embeddings=embeddings,
            hierarchy=hierarchy,
            cases=cases,
            vectorizer=vectorizer,
            targets=targets,
            predictions=base_predictions,
            threshold_grid=[float(value) for value in config["logistic_features"]["threshold_grid"]],
            seed=seed,
        )
        sources["logistic_disagreement_gate"].update(fold_source)
        logistic_fold_records.append(fold_record)

    candidate_rows: list[dict[str, Any]] = []
    candidate_matrices: dict[str, np.ndarray] = {}
    for name, source in sources.items():
        if set(source) != set(all_keys):
            raise RuntimeError(f"OOF source policy is incomplete: {name}")
        row, matrix = _policy_summary(
            name=name,
            source=source,
            rows_by_condition=rows_by_condition,
            cases=cases,
            vectorizer=vectorizer,
            targets=targets,
            b3_matrix=base_predictions[B3],
        )
        candidate_rows.append(row)
        candidate_matrices[name] = matrix

    tolerance = float(config["selection"]["tie_tolerance"])
    best_macro = max(float(row["structured"]["supported_label_macro_f1"]) for row in candidate_rows)
    eligible = [
        row
        for row in candidate_rows
        if best_macro - float(row["structured"]["supported_label_macro_f1"])
        <= tolerance
    ]
    complexity = {
        "question_id_exact_utility": 0,
        "question_id_macro_f1_utility": 0,
        "logistic_disagreement_gate": 1,
    }
    selected = min(
        eligible,
        key=lambda row: (
            -float(row["question"]["exact_answer_set_accuracy"]),
            -float(row["question"]["option_micro_f1"]),
            complexity[str(row["name"])],
        ),
    )
    b3_structured = structured_qa_metrics(targets, base_predictions[B3]).as_dict()
    b4_structured = structured_qa_metrics(targets, base_predictions[B4]).as_dict()
    b3_question = metrics(rows_by_condition[B3].values())
    selected_macro = float(selected["structured"]["supported_label_macro_f1"])
    checks = {
        "oof_macro_exceeds_image_only": selected_macro
        > float(b3_structured["supported_label_macro_f1"]),
        "oof_exact_noninferior": float(selected["question"]["exact_answer_set_accuracy"])
        >= float(b3_question["exact_answer_set_accuracy"]) - 0.001,
        "oof_option_micro_noninferior": float(selected["question"]["option_micro_f1"])
        >= float(b3_question["option_micro_f1"]) - 0.001,
        "oof_macro_exceeds_random_history": selected_macro
        > float(b4_structured["supported_label_macro_f1"]),
        "history_used": int(selected["history_selected_disagreement_count"]) > 0,
    }
    go = all(checks.values())
    summary = {
        "study": config["study"],
        "status": "go_for_full_development_fit" if go else "stop",
        "boundary": config["boundary"],
        "test_accessed": False,
        "new_generation_performed": False,
        "case_count": len(cases),
        "question_count": len(all_keys),
        "fold_case_counts": fold_counts,
        "baselines": {
            "image_only": {"structured": b3_structured, "question": b3_question},
            "random_history": {
                "structured": b4_structured,
                "question": metrics(rows_by_condition[B4].values()),
            },
            "paired_history": {
                "structured": structured_qa_metrics(
                    targets, base_predictions[B6]
                ).as_dict(),
                "question": metrics(rows_by_condition[B6].values()),
            },
        },
        "oof_candidates": candidate_rows,
        "logistic_fold_training": logistic_fold_records,
        "selected_policy": str(selected["name"]),
        "selected_policy_result": selected,
        "selected_minus_image_only": {
            "supported_label_macro_f1": selected_macro
            - float(b3_structured["supported_label_macro_f1"]),
            "question_exact": float(selected["question"]["exact_answer_set_accuracy"])
            - float(b3_question["exact_answer_set_accuracy"]),
            "option_micro_f1": float(selected["question"]["option_micro_f1"])
            - float(b3_question["option_micro_f1"]),
        },
        "advancement": {"passed": go, **checks},
        "interpretation": (
            "The case-level OOF gate passed the post-hoc development rule. This "
            "justifies a full-development fit and a separate confirmation design, "
            "not a Test or clinical claim."
            if go
            else "No real-output OOF policy satisfied all development requirements. "
            "The Final-QA Test must remain locked."
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
        "--config", type=Path, default=ROOT / "config/final_qa_real_output_gate.json"
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
        "--embeddings",
        type=Path,
        default=ROOT / "data/processed/v10_medsiglip_embeddings.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/final_qa_real_output_gate.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_policy": result["selected_policy"],
                "selected_minus_image_only": result["selected_minus_image_only"],
                "advancement": result["advancement"],
            },
            indent=2,
            sort_keys=True,
        )
    )
