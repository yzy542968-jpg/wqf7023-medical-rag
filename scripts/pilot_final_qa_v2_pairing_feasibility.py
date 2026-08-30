"""Run a cached, offline feasibility gate for paired historical QA.

The pilot intentionally avoids model inference. It combines cached image-only
predictions with report-derived answer vectors belonging to image-retrieved
historical cases, then breaks those ownership links in a deterministic control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.qa.question_vectorizer import (  # noqa: E402
    RadReStructQuestionVectorizer,
)
from medical_rag.qa.radrestruct import iter_radrestruct_cases  # noqa: E402
from medical_rag.qa.radrestruct_hierarchy import (  # noqa: E402
    RadReStructHierarchy,
)
from medical_rag.qa.structured_decoding import (  # noqa: E402
    decode_answer_probabilities,
    knn_answer_probabilities,
)
from medical_rag.qa.structured_metrics import (  # noqa: E402
    bootstrap_supported_macro_f1_difference,
    load_answer_vector,
    load_report_keys,
    stack_answer_vectors,
    structured_qa_metrics,
)


IMAGE_ONLY = "b3_no_history_r2"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _answer_matrix(
    role: dict[str, Any], rad_root: Path, report_keys: tuple[str, ...]
) -> np.ndarray:
    return stack_answer_vectors(
        load_answer_vector(
            rad_root
            / f"{case['official_split']}_vectorized_answers"
            / f"{case['source_report_id']}.json",
            report_keys,
        )
        for case in role["cases"]
    )


def _embedding_map(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as payload:
        case_ids = [str(value) for value in payload["case_ids"]]
        embeddings = np.asarray(payload["case_image_embeddings"], dtype=np.float32)
        signature = str(payload["signature"].item())
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0) or not np.isfinite(embeddings).all():
        raise ValueError("MedSigLIP image embeddings are invalid")
    return dict(zip(case_ids, embeddings / norms, strict=True)), signature


def _role_embeddings(
    role: dict[str, Any], embedding_by_case: dict[str, np.ndarray]
) -> np.ndarray:
    missing = [
        str(case["case_id"])
        for case in role["cases"]
        if str(case["case_id"]) not in embedding_by_case
    ]
    if missing:
        raise ValueError(f"Missing image embeddings for {len(missing)} cases")
    return np.stack(
        [embedding_by_case[str(case["case_id"])] for case in role["cases"]]
    )


def _cached_image_only_predictions(
    rows_path: Path,
    validation_role: dict[str, Any],
    rad_root: Path,
    hierarchy: RadReStructHierarchy,
) -> np.ndarray:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with rows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("condition") != IMAGE_ONLY:
                continue
            key = (str(row["case_id"]), int(row["question_index"]))
            if key in rows:
                raise ValueError(f"Duplicate cached image-only row: {key}")
            rows[key] = row

    cases = {case.case_id: case for case in iter_radrestruct_cases(rad_root)}
    vectorizer = RadReStructQuestionVectorizer(hierarchy)
    predictions: list[np.ndarray] = []
    for case_meta in validation_role["cases"]:
        case_id = str(case_meta["case_id"])
        case = cases[case_id]
        answers: list[list[str]] = []
        for question_index, question in enumerate(case.questions):
            row = rows[(case_id, question_index)]
            answers.append(
                [question.options[int(index)] for index in row["predicted_indices"]]
            )
        predictions.append(vectorizer.vectorize_answers(case.questions, answers))
    return np.stack(predictions)


def _pilot_selection_mask(
    validation_role: dict[str, Any], prefix: str
) -> np.ndarray:
    mask = []
    for case in validation_role["cases"]:
        payload = f"{prefix}{str(case['case_id']).strip()}".encode("utf-8")
        mask.append(int(hashlib.sha256(payload).hexdigest(), 16) % 2 == 0)
    values = np.asarray(mask, dtype=bool)
    if values.sum() < 100 or (~values).sum() < 100:
        raise ValueError("Pilot hash split produced an unexpectedly small partition")
    return values


def _fixed_point_free_payload_permutation(
    case_ids: list[str], *, seed: int, replicate: int
) -> np.ndarray:
    if len(case_ids) < 2:
        raise ValueError("At least two historical cases are required")
    order = np.asarray(
        sorted(
            range(len(case_ids)),
            key=lambda index: hashlib.sha256(
                (
                    f"final-qa-v2-pair-shuffle|{seed}|{replicate}|"
                    f"{case_ids[index]}"
                ).encode("utf-8")
            ).hexdigest(),
        ),
        dtype=int,
    )
    permutation = np.empty(len(case_ids), dtype=int)
    permutation[order] = np.roll(order, -1)
    if np.any(permutation == np.arange(len(case_ids))):
        raise AssertionError("Pairing control must be fixed-point-free")
    return permutation


def _decode(
    probabilities: np.ndarray,
    hierarchy: RadReStructHierarchy,
    config: dict[str, Any],
) -> np.ndarray:
    fusion = config["fusion"]
    return decode_answer_probabilities(
        probabilities,
        hierarchy,
        multi_choice_threshold=float(fusion["multi_choice_threshold"]),
        fixed_choice_threshold=float(fusion["fixed_choice_threshold"]),
    )


def _metric(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    return structured_qa_metrics(targets, predictions).as_dict()


def _question_level_metrics(
    role: dict[str, Any],
    targets: np.ndarray,
    predictions: np.ndarray,
    rad_root: Path,
    hierarchy: RadReStructHierarchy,
    mask: np.ndarray,
) -> dict[str, float | int]:
    cases = {case.case_id: case for case in iter_radrestruct_cases(rad_root)}
    vectorizer = RadReStructQuestionVectorizer(hierarchy)
    exact = true_positive = false_positive = false_negative = 0
    question_count = 0
    for case_index, case_meta in enumerate(role["cases"]):
        if not bool(mask[case_index]):
            continue
        case = cases[str(case_meta["case_id"])]
        for question_id in vectorizer.question_ids(case.questions):
            indices = hierarchy.indices_by_question[question_id]
            target = targets[case_index, indices]
            prediction = predictions[case_index, indices]
            exact += int(np.array_equal(target, prediction))
            true_positive += int(np.logical_and(target, prediction).sum())
            false_positive += int(np.logical_and(1 - target, prediction).sum())
            false_negative += int(np.logical_and(target, 1 - prediction).sum())
            question_count += 1
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "question_count": question_count,
        "exact_answer_set_accuracy": exact / question_count,
        "option_micro_f1": 2 * true_positive / denominator if denominator else 0.0,
    }


def _question_level_complementarity(
    role: dict[str, Any],
    targets: np.ndarray,
    image_predictions: np.ndarray,
    history_predictions: np.ndarray,
    rad_root: Path,
    hierarchy: RadReStructHierarchy,
    mask: np.ndarray,
) -> dict[str, float | int]:
    cases = {case.case_id: case for case in iter_radrestruct_cases(rad_root)}
    vectorizer = RadReStructQuestionVectorizer(hierarchy)
    both = image_only = history_only = neither = 0
    for case_index, case_meta in enumerate(role["cases"]):
        if not bool(mask[case_index]):
            continue
        case = cases[str(case_meta["case_id"])]
        for question_id in vectorizer.question_ids(case.questions):
            indices = hierarchy.indices_by_question[question_id]
            target = targets[case_index, indices]
            image_correct = bool(
                np.array_equal(target, image_predictions[case_index, indices])
            )
            history_correct = bool(
                np.array_equal(target, history_predictions[case_index, indices])
            )
            if image_correct and history_correct:
                both += 1
            elif image_correct:
                image_only += 1
            elif history_correct:
                history_only += 1
            else:
                neither += 1
    total = both + image_only + history_only + neither
    return {
        "question_count": total,
        "both_correct_count": both,
        "image_only_correct_count": image_only,
        "history_only_correct_count": history_only,
        "neither_correct_count": neither,
        "image_only_correct_rate": image_only / total,
        "history_only_correct_rate": history_only / total,
        "oracle_source_selection_exact_accuracy": (both + image_only + history_only)
        / total,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_json(args.config)
    manifest = _load_json(args.manifest)
    hierarchy = RadReStructHierarchy(args.radrestruct_root)
    report_keys = load_report_keys(args.radrestruct_root)
    train_role = manifest["roles"]["train"]
    validation_role = manifest["roles"]["validation"]

    train_targets = _answer_matrix(train_role, args.radrestruct_root, report_keys)
    validation_targets = _answer_matrix(
        validation_role, args.radrestruct_root, report_keys
    )
    image_only = _cached_image_only_predictions(
        args.validation_rows,
        validation_role,
        args.radrestruct_root,
        hierarchy,
    )
    embedding_by_case, embedding_signature = _embedding_map(args.embeddings)
    train_embeddings = _role_embeddings(train_role, embedding_by_case)
    validation_embeddings = _role_embeddings(validation_role, embedding_by_case)
    similarities = validation_embeddings @ train_embeddings.T

    selection = _pilot_selection_mask(
        validation_role, str(config["pilot_split"]["hash_prefix"])
    )
    holdout = ~selection
    history_config = config["historical_policy"]
    history_probabilities = knn_answer_probabilities(
        similarities,
        train_targets,
        top_k=1,
        weighting=str(history_config["weighting"]),
        softmax_temperature=float(history_config["softmax_temperature"]),
    )

    candidates: list[dict[str, Any]] = []
    for alpha_value in config["fusion"]["alpha_grid"]:
        alpha = float(alpha_value)
        predictions = _decode(
            alpha * image_only + (1.0 - alpha) * history_probabilities,
            hierarchy,
            config,
        )
        metrics = _metric(validation_targets[selection], predictions[selection])
        candidates.append(
            {"alpha": alpha, "selection_metrics": metrics, "predictions": predictions}
        )
    selected = min(
        candidates,
        key=lambda row: (
            -float(row["selection_metrics"]["supported_label_macro_f1"]),
            -float(row["alpha"]),
        ),
    )
    alpha = float(selected["alpha"])
    aligned_history_predictions = _decode(history_probabilities, hierarchy, config)
    aligned_fusion_predictions = np.asarray(selected["predictions"], dtype=np.uint8)

    shuffled_history_scores: list[float] = []
    shuffled_fusion_scores: list[float] = []
    train_case_ids = [str(case["case_id"]) for case in train_role["cases"]]
    for replicate in range(int(config["pairing_control"]["replicates"])):
        permutation = _fixed_point_free_payload_permutation(
            train_case_ids,
            seed=int(config["seed"]),
            replicate=replicate,
        )
        shuffled_probabilities = knn_answer_probabilities(
            similarities,
            train_targets[permutation],
            top_k=1,
            weighting=str(history_config["weighting"]),
            softmax_temperature=float(history_config["softmax_temperature"]),
        )
        shuffled_history = _decode(shuffled_probabilities, hierarchy, config)
        shuffled_fusion = _decode(
            alpha * image_only + (1.0 - alpha) * shuffled_probabilities,
            hierarchy,
            config,
        )
        shuffled_history_scores.append(
            structured_qa_metrics(
                validation_targets[holdout], shuffled_history[holdout]
            ).supported_label_macro_f1
        )
        shuffled_fusion_scores.append(
            structured_qa_metrics(
                validation_targets[holdout], shuffled_fusion[holdout]
            ).supported_label_macro_f1
        )

    image_only_metrics = _metric(validation_targets[holdout], image_only[holdout])
    aligned_history_metrics = _metric(
        validation_targets[holdout], aligned_history_predictions[holdout]
    )
    aligned_fusion_metrics = _metric(
        validation_targets[holdout], aligned_fusion_predictions[holdout]
    )
    image_only_question_metrics = _question_level_metrics(
        validation_role,
        validation_targets,
        image_only,
        args.radrestruct_root,
        hierarchy,
        holdout,
    )
    aligned_history_question_metrics = _question_level_metrics(
        validation_role,
        validation_targets,
        aligned_history_predictions,
        args.radrestruct_root,
        hierarchy,
        holdout,
    )
    aligned_fusion_question_metrics = _question_level_metrics(
        validation_role,
        validation_targets,
        aligned_fusion_predictions,
        args.radrestruct_root,
        hierarchy,
        holdout,
    )
    question_complementarity = _question_level_complementarity(
        validation_role,
        validation_targets,
        image_only,
        aligned_history_predictions,
        args.radrestruct_root,
        hierarchy,
        holdout,
    )
    fusion_delta = float(
        aligned_fusion_metrics["supported_label_macro_f1"]
        - image_only_metrics["supported_label_macro_f1"]
    )
    shuffled_fusion_mean = float(np.mean(shuffled_fusion_scores))
    pairing_delta = float(
        aligned_fusion_metrics["supported_label_macro_f1"] - shuffled_fusion_mean
    )
    go = fusion_delta > 0 and pairing_delta > 0 and alpha < 1.0

    summary = {
        "study": config["study"],
        "status": "go_for_deployable_development" if go else "stop_or_redesign",
        "config": str(args.config.relative_to(ROOT)).replace("\\", "/"),
        "boundary": config["boundary"],
        "test_accessed": False,
        "model_inference_performed": False,
        "embedding_signature": embedding_signature,
        "historical_bank_case_count": len(train_role["cases"]),
        "pilot_selection_case_count": int(selection.sum()),
        "pilot_holdout_case_count": int(holdout.sum()),
        "image_only_full_validation_reproduction": _metric(
            validation_targets, image_only
        ),
        "selected_on_pilot_selection": {
            "alpha": alpha,
            "metrics": selected["selection_metrics"],
        },
        "pilot_holdout": {
            "image_only": image_only_metrics,
            "image_only_question_level": image_only_question_metrics,
            "aligned_history_only": aligned_history_metrics,
            "aligned_history_question_level": aligned_history_question_metrics,
            "aligned_fusion": aligned_fusion_metrics,
            "aligned_fusion_question_level": aligned_fusion_question_metrics,
            "image_history_question_complementarity": question_complementarity,
            "aligned_fusion_minus_image_only_question_exact_accuracy": float(
                aligned_fusion_question_metrics["exact_answer_set_accuracy"]
                - image_only_question_metrics["exact_answer_set_accuracy"]
            ),
            "aligned_fusion_minus_image_only_question_option_micro_f1": float(
                aligned_fusion_question_metrics["option_micro_f1"]
                - image_only_question_metrics["option_micro_f1"]
            ),
            "aligned_fusion_minus_image_only_macro_f1": fusion_delta,
            "case_bootstrap_aligned_fusion_vs_image_only": (
                bootstrap_supported_macro_f1_difference(
                    validation_targets[holdout],
                    aligned_fusion_predictions[holdout],
                    image_only[holdout],
                    samples=int(config["bootstrap"]["replicates"]),
                    seed=int(config["bootstrap"]["seed"]),
                )
            ),
            "shuffled_pair_history_macro_f1": {
                "replicates": len(shuffled_history_scores),
                "mean": float(np.mean(shuffled_history_scores)),
                "minimum": float(np.min(shuffled_history_scores)),
                "maximum": float(np.max(shuffled_history_scores)),
                "aligned_minus_mean": float(
                    aligned_history_metrics["supported_label_macro_f1"]
                    - np.mean(shuffled_history_scores)
                ),
            },
            "shuffled_pair_fusion_macro_f1": {
                "replicates": len(shuffled_fusion_scores),
                "mean": shuffled_fusion_mean,
                "minimum": float(np.min(shuffled_fusion_scores)),
                "maximum": float(np.max(shuffled_fusion_scores)),
                "aligned_minus_mean": pairing_delta,
                "aligned_exceeds_all_shuffles": bool(
                    aligned_fusion_metrics["supported_label_macro_f1"]
                    > np.max(shuffled_fusion_scores)
                ),
                "plus_one_monte_carlo_p": float(
                    (
                        1
                        + sum(
                            score
                            >= aligned_fusion_metrics["supported_label_macro_f1"]
                            for score in shuffled_fusion_scores
                        )
                    )
                    / (1 + len(shuffled_fusion_scores))
                ),
            },
        },
        "go_rule": {
            "passed": go,
            "fusion_improves_image_only": fusion_delta > 0,
            "aligned_fusion_exceeds_shuffled_mean": pairing_delta > 0,
            "history_used": alpha < 1.0,
        },
        "interpretation": (
            "The narrow prespecified macro-F1 GO rule passed, but fixed fusion did "
            "not improve question-level exact accuracy or option micro-F1. This is "
            "conditional feasibility evidence for selective paired-history gating, "
            "not evidence that unconditional fusion is suitable."
            if go
            else "The cached upper-bound pilot did not satisfy the predefined GO rule."
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
        default=ROOT / "config/final_qa_v2_feasibility.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json",
    )
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "data/processed/v10_medsiglip_embeddings.npz",
    )
    parser.add_argument(
        "--validation-rows",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/final_qa_validation_rows.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/final_qa_v2_feasibility.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected": result["selected_on_pilot_selection"],
                "pilot_holdout": result["pilot_holdout"],
                "go_rule": result["go_rule"],
            },
            indent=2,
            sort_keys=True,
        )
    )
