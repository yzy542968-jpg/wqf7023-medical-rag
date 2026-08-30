"""Develop a question-conditional paired-history gate without Test access.

The experiment reuses cached image-only Validation predictions and an oracle
report-derived historical answer payload. It is a bounded development study,
not a deployable or confirmatory evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.qa.radrestruct import (  # noqa: E402
    RadReStructCase,
    iter_radrestruct_cases,
)
from medical_rag.qa.radrestruct_hierarchy import (  # noqa: E402
    RadReStructHierarchy,
)
from medical_rag.qa.question_vectorizer import (  # noqa: E402
    RadReStructQuestionVectorizer,
)
from medical_rag.qa.structured_metrics import (  # noqa: E402
    load_answer_vector,
    load_report_keys,
    stack_answer_vectors,
    structured_qa_metrics,
)

from pilot_final_qa_v2_pairing_feasibility import (  # noqa: E402
    _cached_image_only_predictions,
    _fixed_point_free_payload_permutation,
    _pilot_selection_mask,
)


NUMERIC_FEATURE_NAMES = (
    "selected_image_similarity",
    "selected_target_image_report_similarity",
    "selected_question_report_similarity",
    "selected_pair_consistency",
    "candidate_relevance_probability",
    "candidate_probability_margin",
    "top3_answer_agreement",
    "image_history_option_overlap",
    "image_answer_count",
    "history_answer_count",
)


@dataclass(frozen=True)
class QuestionRecord:
    case_index: int
    question_index: int
    question_id: int
    answer_type: str
    text: str


@dataclass
class CandidateBlock:
    records: list[QuestionRecord]
    bank_indices: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    payload_answers: np.ndarray


@dataclass
class SelectorOutput:
    history_predictions: np.ndarray
    gate_numeric: np.ndarray
    question_ids: np.ndarray
    answer_types: np.ndarray
    records: list[QuestionRecord]


class GateFeatureEncoder:
    def __init__(self, question_count: int, answer_types: tuple[str, ...]) -> None:
        self.question_count = int(question_count)
        self.answer_types = answer_types
        self.type_to_index = {value: index for index, value in enumerate(answer_types)}
        self.scaler = StandardScaler()

    def fit(self, numeric: np.ndarray) -> "GateFeatureEncoder":
        self.scaler.fit(np.asarray(numeric, dtype=np.float64))
        return self

    def transform(
        self,
        numeric: np.ndarray,
        question_ids: np.ndarray,
        answer_types: np.ndarray,
    ) -> np.ndarray:
        scaled = self.scaler.transform(np.asarray(numeric, dtype=np.float64))
        qids = np.asarray(question_ids, dtype=int)
        if np.any(qids < 0) or np.any(qids >= self.question_count):
            raise ValueError("Question IDs fall outside the frozen hierarchy")
        q_one_hot = np.zeros((len(qids), self.question_count), dtype=np.float32)
        q_one_hot[np.arange(len(qids)), qids] = 1.0
        type_one_hot = np.zeros(
            (len(qids), len(self.answer_types)), dtype=np.float32
        )
        for row_index, answer_type in enumerate(answer_types):
            type_index = self.type_to_index.get(str(answer_type))
            if type_index is not None:
                type_one_hot[row_index, type_index] = 1.0
        return np.concatenate(
            [scaled.astype(np.float32), q_one_hot, type_one_hot], axis=1
        )


class TorchMLPGate:
    def __init__(self, input_dimension: int, seed: int) -> None:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        ).to(self.device)
        self.best_epoch = 0
        self.best_validation_loss = math.inf

    def fit(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        validation_x: np.ndarray,
        validation_y: np.ndarray,
        *,
        max_epochs: int = 300,
        patience: int = 25,
    ) -> "TorchMLPGate":
        torch = self.torch
        x_train = torch.as_tensor(train_x, dtype=torch.float32, device=self.device)
        y_train = torch.as_tensor(train_y, dtype=torch.float32, device=self.device)
        x_validation = torch.as_tensor(
            validation_x, dtype=torch.float32, device=self.device
        )
        y_validation = torch.as_tensor(
            validation_y, dtype=torch.float32, device=self.device
        )
        positive = float(np.asarray(train_y).sum())
        negative = float(len(train_y) - positive)
        pos_weight = torch.tensor(
            negative / max(positive, 1.0), dtype=torch.float32, device=self.device
        )
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=1e-4
        )
        best_state: dict[str, Any] | None = None
        stale = 0
        for epoch in range(max_epochs):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(self.model(x_train).squeeze(1), y_train)
            loss.backward()
            optimizer.step()
            self.model.eval()
            with torch.inference_mode():
                validation_loss = float(
                    criterion(
                        self.model(x_validation).squeeze(1), y_validation
                    ).item()
                )
            if validation_loss < self.best_validation_loss - 1e-6:
                self.best_validation_loss = validation_loss
                self.best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("MLP early stopping did not retain a checkpoint")
        self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(
                torch.as_tensor(values, dtype=torch.float32, device=self.device)
            ).squeeze(1)
            return torch.sigmoid(logits).float().cpu().numpy()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalized_rows(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError(f"{name} contains zero-norm rows")
    return matrix / norms


def _load_embedding_state(
    path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as payload:
        image_ids = [str(value) for value in payload["case_ids"]]
        report_ids = [str(value) for value in payload["report_ids"]]
        image_embeddings = _normalized_rows(
            payload["case_image_embeddings"], "image embeddings"
        )
        report_embeddings = _normalized_rows(
            payload["report_embeddings"], "report embeddings"
        )
        signature = str(payload["signature"].item())
    return (
        dict(zip(image_ids, image_embeddings, strict=True)),
        dict(zip(report_ids, report_embeddings, strict=True)),
        signature,
    )


def _load_question_embeddings(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as payload:
        questions = [str(value) for value in payload["questions"]]
        embeddings = _normalized_rows(payload["embeddings"], "question embeddings")
        fingerprint = str(payload["text_sha256"].item())
    return dict(zip(questions, embeddings, strict=True)), fingerprint


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


def _hash_parity(prefix: str, case_id: str) -> bool:
    digest = hashlib.sha256(f"{prefix}{case_id.strip()}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % 2 == 0


def _question_records(
    validation_role: dict[str, Any],
    cases: dict[str, RadReStructCase],
    vectorizer: RadReStructQuestionVectorizer,
) -> tuple[list[QuestionRecord], dict[int, list[QuestionRecord]]]:
    records: list[QuestionRecord] = []
    by_case: dict[int, list[QuestionRecord]] = {}
    for case_index, case_meta in enumerate(validation_role["cases"]):
        case = cases[str(case_meta["case_id"])]
        qids = vectorizer.question_ids(case.questions)
        case_records = [
            QuestionRecord(
                case_index=case_index,
                question_index=question_index,
                question_id=int(question_id),
                answer_type=question.answer_type or "unknown",
                text=question.question,
            )
            for question_index, (question_id, question) in enumerate(
                zip(qids, case.questions, strict=True)
            )
        ]
        records.extend(case_records)
        by_case[case_index] = case_records
    return records, by_case


def _build_blocks(
    *,
    validation_role: dict[str, Any],
    records_by_case: dict[int, list[QuestionRecord]],
    validation_targets: np.ndarray,
    validation_images: np.ndarray,
    train_images: np.ndarray,
    train_reports: np.ndarray,
    train_answers: np.ndarray,
    train_clusters: np.ndarray,
    question_embeddings: dict[str, np.ndarray],
    hierarchy: RadReStructHierarchy,
    shortlist_k: int,
) -> list[CandidateBlock]:
    blocks: list[CandidateBlock] = []
    pair_consistency = np.sum(train_images * train_reports, axis=1)
    for case_index, case_meta in enumerate(validation_role["cases"]):
        target_image = validation_images[case_index]
        image_scores = train_images @ target_image
        eligible = train_clusters != str(case_meta["cluster_id"])
        eligible_indices = np.flatnonzero(eligible)
        ranked = eligible_indices[
            np.argsort(-image_scores[eligible_indices], kind="stable")[:shortlist_k]
        ]
        if len(ranked) != shortlist_k:
            raise ValueError("Historical bank is smaller than the frozen shortlist")
        records = records_by_case[case_index]
        reports = train_reports[ranked]
        question_matrix = np.stack(
            [question_embeddings[record.text] for record in records]
        )
        q_report = question_matrix @ reports.T
        target_report = reports @ target_image
        top_score = float(image_scores[ranked[0]])
        features = np.empty((len(records), shortlist_k, 6), dtype=np.float32)
        features[:, :, 0] = image_scores[ranked][None, :]
        features[:, :, 1] = target_report[None, :]
        features[:, :, 2] = q_report
        features[:, :, 3] = pair_consistency[ranked][None, :]
        features[:, :, 4] = 1.0 / (1.0 + np.arange(shortlist_k))[None, :]
        features[:, :, 5] = top_score - image_scores[ranked][None, :]

        labels = np.empty((len(records), shortlist_k), dtype=np.uint8)
        for row_index, record in enumerate(records):
            indices = hierarchy.indices_by_question[record.question_id]
            labels[row_index] = np.all(
                train_answers[ranked][:, indices]
                == validation_targets[case_index, indices][None, :],
                axis=1,
            )
        blocks.append(
            CandidateBlock(
                records=records,
                bank_indices=ranked,
                features=features,
                labels=labels,
                payload_answers=train_answers[ranked],
            )
        )
    return blocks


def _fit_candidate_model(
    blocks: list[CandidateBlock], case_mask: np.ndarray, seed: int
) -> tuple[StandardScaler, LogisticRegression, dict[str, int | float]]:
    selected = [
        block for case_index, block in enumerate(blocks) if bool(case_mask[case_index])
    ]
    x = np.concatenate([block.features.reshape(-1, 6) for block in selected])
    y = np.concatenate([block.labels.reshape(-1) for block in selected])
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    ).fit(scaler.transform(x), y)
    return scaler, model, {
        "candidate_row_count": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_rate": float(y.mean()),
    }


def _candidate_scores(
    block: CandidateBlock,
    policy: str,
    scaler: StandardScaler,
    model: LogisticRegression,
) -> tuple[np.ndarray, np.ndarray]:
    relevance = model.predict_proba(
        scaler.transform(block.features.reshape(-1, 6))
    )[:, 1].reshape(block.labels.shape)
    if policy == "image_top1":
        return block.features[:, :, 0], relevance
    if policy == "logistic_mcr_lite":
        return relevance, relevance
    raise ValueError(f"Unknown candidate policy: {policy}")


def _top3_agreement(answer_rows: np.ndarray) -> float:
    keys = [row.tobytes() for row in np.asarray(answer_rows, dtype=np.uint8)]
    return max(keys.count(key) for key in set(keys)) / len(keys)


def _selector_output(
    *,
    blocks: list[CandidateBlock],
    policy: str,
    scaler: StandardScaler,
    model: LogisticRegression,
    image_only: np.ndarray,
    hierarchy: RadReStructHierarchy,
) -> SelectorOutput:
    history = np.zeros_like(image_only, dtype=np.uint8)
    raw_features: list[np.ndarray] = []
    records: list[QuestionRecord] = []
    for block in blocks:
        scores, relevance = _candidate_scores(block, policy, scaler, model)
        for row_index, record in enumerate(block.records):
            order = np.argsort(-scores[row_index], kind="stable")
            selected_position = int(order[0])
            selected_answers = block.payload_answers[selected_position]
            indices = hierarchy.indices_by_question[record.question_id]
            history[record.case_index, indices] = selected_answers[indices]
            selected_feature = block.features[row_index, selected_position]
            image_bits = image_only[record.case_index, indices]
            history_bits = selected_answers[indices]
            union = int(np.logical_or(image_bits, history_bits).sum())
            overlap = (
                float(np.logical_and(image_bits, history_bits).sum() / union)
                if union
                else 1.0
            )
            top3 = order[:3]
            probability_order = np.argsort(-relevance[row_index], kind="stable")
            probability_margin = float(
                relevance[row_index, probability_order[0]]
                - relevance[row_index, probability_order[1]]
            )
            raw_features.append(
                np.asarray(
                    [
                        selected_feature[0],
                        selected_feature[1],
                        selected_feature[2],
                        selected_feature[3],
                        relevance[row_index, selected_position],
                        probability_margin,
                        _top3_agreement(block.payload_answers[top3][:, indices]),
                        overlap,
                        float(image_bits.sum()),
                        float(history_bits.sum()),
                    ],
                    dtype=np.float32,
                )
            )
            records.append(record)
    history = hierarchy.clean(history)
    return SelectorOutput(
        history_predictions=history,
        gate_numeric=np.stack(raw_features),
        question_ids=np.asarray([record.question_id for record in records]),
        answer_types=np.asarray([record.answer_type for record in records]),
        records=records,
    )


def _record_masks(
    records: list[QuestionRecord], case_mask: np.ndarray
) -> np.ndarray:
    return np.asarray([bool(case_mask[record.case_index]) for record in records])


def _disagreement_and_labels(
    output: SelectorOutput,
    image_only: np.ndarray,
    targets: np.ndarray,
    hierarchy: RadReStructHierarchy,
) -> tuple[np.ndarray, np.ndarray]:
    disagreement = np.zeros(len(output.records), dtype=bool)
    labels = np.zeros(len(output.records), dtype=np.uint8)
    for row_index, record in enumerate(output.records):
        indices = hierarchy.indices_by_question[record.question_id]
        image = image_only[record.case_index, indices]
        history = output.history_predictions[record.case_index, indices]
        target = targets[record.case_index, indices]
        disagreement[row_index] = not np.array_equal(image, history)
        labels[row_index] = int(
            np.array_equal(history, target) and not np.array_equal(image, target)
        )
    return disagreement, labels


def _apply_gate(
    output: SelectorOutput,
    image_only: np.ndarray,
    hierarchy: RadReStructHierarchy,
    history_probability: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, int]:
    predictions = np.asarray(image_only, dtype=np.uint8).copy()
    used = 0
    for row_index, record in enumerate(output.records):
        indices = hierarchy.indices_by_question[record.question_id]
        image = image_only[record.case_index, indices]
        history = output.history_predictions[record.case_index, indices]
        if np.array_equal(image, history):
            continue
        if float(history_probability[row_index]) >= threshold:
            predictions[record.case_index, indices] = history
            used += 1
    return hierarchy.clean(predictions), used


def _exact_case_values(
    role: dict[str, Any],
    targets: np.ndarray,
    predictions: np.ndarray,
    cases: dict[str, RadReStructCase],
    vectorizer: RadReStructQuestionVectorizer,
    hierarchy: RadReStructHierarchy,
) -> np.ndarray:
    values = np.zeros(len(role["cases"]), dtype=float)
    for case_index, case_meta in enumerate(role["cases"]):
        case = cases[str(case_meta["case_id"])]
        exact: list[float] = []
        for qid in vectorizer.question_ids(case.questions):
            indices = hierarchy.indices_by_question[qid]
            exact.append(
                float(np.array_equal(targets[case_index, indices], predictions[case_index, indices]))
            )
        values[case_index] = float(np.mean(exact))
    return values


def _case_bootstrap_exact_difference(
    case_a: np.ndarray,
    case_b: np.ndarray,
    mask: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    differences = np.asarray(case_a[mask] - case_b[mask], dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    estimates = differences[indices].mean(axis=1)
    return {
        "samples": samples,
        "seed": seed,
        "case_count": int(len(differences)),
        "observed_difference": float(differences.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "probability_difference_greater_than_zero": float((estimates > 0).mean()),
    }


def _metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    hierarchy: RadReStructHierarchy,
    mask: np.ndarray,
    records: list[QuestionRecord],
) -> dict[str, Any]:
    exact = true_positive = false_positive = false_negative = question_count = 0
    for record in records:
        if not bool(mask[record.case_index]):
            continue
        indices = hierarchy.indices_by_question[record.question_id]
        target = targets[record.case_index, indices]
        prediction = predictions[record.case_index, indices]
        exact += int(np.array_equal(target, prediction))
        true_positive += int(np.logical_and(target, prediction).sum())
        false_positive += int(np.logical_and(1 - target, prediction).sum())
        false_negative += int(np.logical_and(target, 1 - prediction).sum())
        question_count += 1
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "structured": structured_qa_metrics(
            targets[mask], predictions[mask]
        ).as_dict(),
        "question": {
            "question_count": question_count,
            "exact_answer_set_accuracy": exact / question_count,
            "option_micro_f1": (
                2 * true_positive / denominator if denominator else 0.0
            ),
        },
    }


def _threshold_search(
    *,
    probabilities: np.ndarray,
    output: SelectorOutput,
    image_only: np.ndarray,
    targets: np.ndarray,
    hierarchy: RadReStructHierarchy,
    calibration_mask: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        predictions, used = _apply_gate(
            output, image_only, hierarchy, probabilities, float(threshold)
        )
        calibration_use = sum(
            int(
                bool(calibration_mask[record.case_index])
                and not np.array_equal(
                    image_only[record.case_index, hierarchy.indices_by_question[record.question_id]],
                    output.history_predictions[
                        record.case_index, hierarchy.indices_by_question[record.question_id]
                    ],
                )
                and float(probabilities[row_index]) >= float(threshold)
            )
            for row_index, record in enumerate(output.records)
        )
        metrics = _metrics(
            targets,
            predictions,
            hierarchy,
            calibration_mask,
            output.records,
        )
        candidates.append(
            {
                "threshold": float(threshold),
                "history_use_count": int(calibration_use),
                "full_development_history_use_count": int(used),
                "metrics": metrics,
                "predictions": predictions,
            }
        )
    return min(
        candidates,
        key=lambda row: (
            -float(row["metrics"]["question"]["exact_answer_set_accuracy"]),
            -float(row["metrics"]["question"]["option_micro_f1"]),
            int(row["history_use_count"]),
            float(row["threshold"]),
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_json(args.config)
    manifest = _load_json(args.manifest)
    hierarchy = RadReStructHierarchy(args.radrestruct_root)
    vectorizer = RadReStructQuestionVectorizer(hierarchy)
    report_keys = load_report_keys(args.radrestruct_root)
    train_role = manifest["roles"]["train"]
    validation_role = manifest["roles"]["validation"]
    cases = {case.case_id: case for case in iter_radrestruct_cases(args.radrestruct_root)}

    train_targets_full = _answer_matrix(train_role, args.radrestruct_root, report_keys)
    validation_targets = _answer_matrix(
        validation_role, args.radrestruct_root, report_keys
    )
    image_only = _cached_image_only_predictions(
        args.validation_rows,
        validation_role,
        args.radrestruct_root,
        hierarchy,
    )
    image_map, report_map, embedding_signature = _load_embedding_state(
        args.embeddings
    )
    question_map, question_fingerprint = _load_question_embeddings(
        args.question_embeddings
    )

    eligible_train_indices = [
        index
        for index, case in enumerate(train_role["cases"])
        if str(case["case_id"]) in image_map and str(case["case_id"]) in report_map
    ]
    train_meta = [train_role["cases"][index] for index in eligible_train_indices]
    train_targets = train_targets_full[eligible_train_indices]
    train_images = np.stack([image_map[str(case["case_id"])] for case in train_meta])
    train_reports = np.stack([report_map[str(case["case_id"])] for case in train_meta])
    train_clusters = np.asarray([str(case["cluster_id"]) for case in train_meta])
    validation_images = np.stack(
        [image_map[str(case["case_id"])] for case in validation_role["cases"]]
    )

    all_records, records_by_case = _question_records(
        validation_role, cases, vectorizer
    )
    missing_questions = sorted({record.text for record in all_records} - set(question_map))
    if missing_questions:
        raise ValueError(f"Question cache is missing {len(missing_questions)} texts")

    selection = _pilot_selection_mask(
        validation_role, "final-qa-v2-pilot-fit|case_id"
    )
    development_holdout = ~selection
    gate_train = np.asarray(
        [
            bool(selection[index])
            and _hash_parity("final-qa-v2-gate-train|", str(case["case_id"]))
            for index, case in enumerate(validation_role["cases"])
        ],
        dtype=bool,
    )
    gate_calibration = np.logical_and(selection, ~gate_train)
    mlp_train = np.asarray(
        [
            bool(gate_train[index])
            and _hash_parity("final-qa-v2-mlp-train|", str(case["case_id"]))
            for index, case in enumerate(validation_role["cases"])
        ],
        dtype=bool,
    )
    mlp_early = np.logical_and(gate_train, ~mlp_train)

    blocks = _build_blocks(
        validation_role=validation_role,
        records_by_case=records_by_case,
        validation_targets=validation_targets,
        validation_images=validation_images,
        train_images=train_images,
        train_reports=train_reports,
        train_answers=train_targets,
        train_clusters=train_clusters,
        question_embeddings=question_map,
        hierarchy=hierarchy,
        shortlist_k=int(config["candidate_frame"]["image_shortlist_k"]),
    )
    candidate_scaler, candidate_model, candidate_training = _fit_candidate_model(
        blocks, gate_train, int(config["seed"])
    )
    selector_candidates: list[dict[str, Any]] = []
    selector_outputs: dict[str, SelectorOutput] = {}
    for policy in config["candidate_models"]:
        output = _selector_output(
            blocks=blocks,
            policy=str(policy),
            scaler=candidate_scaler,
            model=candidate_model,
            image_only=image_only,
            hierarchy=hierarchy,
        )
        selector_outputs[str(policy)] = output
        selector_candidates.append(
            {
                "policy": str(policy),
                "gate_calibration_history_metrics": _metrics(
                    validation_targets,
                    output.history_predictions,
                    hierarchy,
                    gate_calibration,
                    output.records,
                ),
            }
        )
    candidate_tolerance = float(config["selection"]["tie_tolerance"])
    image_row = next(row for row in selector_candidates if row["policy"] == "image_top1")
    logistic_row = next(
        row for row in selector_candidates if row["policy"] == "logistic_mcr_lite"
    )
    image_exact = float(
        image_row["gate_calibration_history_metrics"]["question"][
            "exact_answer_set_accuracy"
        ]
    )
    logistic_exact = float(
        logistic_row["gate_calibration_history_metrics"]["question"][
            "exact_answer_set_accuracy"
        ]
    )
    selected_policy = (
        "logistic_mcr_lite"
        if logistic_exact > image_exact + candidate_tolerance
        else "image_top1"
    )
    output = selector_outputs[selected_policy]
    disagreement, history_better = _disagreement_and_labels(
        output, image_only, validation_targets, hierarchy
    )
    record_gate_train = _record_masks(output.records, gate_train)
    record_gate_calibration = _record_masks(output.records, gate_calibration)
    training_rows = np.logical_and(record_gate_train, disagreement)
    calibration_rows = np.logical_and(record_gate_calibration, disagreement)
    if history_better[training_rows].sum() == 0:
        raise RuntimeError("Gate training contains no history-only recoveries")

    answer_types = tuple(sorted(set(str(value) for value in output.answer_types)))
    logistic_encoder = GateFeatureEncoder(
        len(hierarchy.indices_by_question), answer_types
    ).fit(output.gate_numeric[training_rows])
    logistic_x = logistic_encoder.transform(
        output.gate_numeric[training_rows],
        output.question_ids[training_rows],
        output.answer_types[training_rows],
    )
    gate_logistic = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=int(config["seed"]),
        solver="lbfgs",
    ).fit(logistic_x, history_better[training_rows])
    all_logistic_probability = gate_logistic.predict_proba(
        logistic_encoder.transform(
            output.gate_numeric, output.question_ids, output.answer_types
        )
    )[:, 1]

    record_mlp_train = _record_masks(output.records, mlp_train)
    record_mlp_early = _record_masks(output.records, mlp_early)
    mlp_training_rows = np.logical_and(record_mlp_train, disagreement)
    mlp_early_rows = np.logical_and(record_mlp_early, disagreement)
    mlp_encoder = GateFeatureEncoder(
        len(hierarchy.indices_by_question), answer_types
    ).fit(output.gate_numeric[mlp_training_rows])
    mlp_gate = TorchMLPGate(
        mlp_encoder.transform(
            output.gate_numeric[mlp_training_rows],
            output.question_ids[mlp_training_rows],
            output.answer_types[mlp_training_rows],
        ).shape[1],
        int(config["seed"]),
    ).fit(
        mlp_encoder.transform(
            output.gate_numeric[mlp_training_rows],
            output.question_ids[mlp_training_rows],
            output.answer_types[mlp_training_rows],
        ),
        history_better[mlp_training_rows],
        mlp_encoder.transform(
            output.gate_numeric[mlp_early_rows],
            output.question_ids[mlp_early_rows],
            output.answer_types[mlp_early_rows],
        ),
        history_better[mlp_early_rows],
    )
    all_mlp_probability = mlp_gate.predict_proba(
        mlp_encoder.transform(
            output.gate_numeric, output.question_ids, output.answer_types
        )
    )

    similarity = output.gate_numeric[:, 0]
    calibration_similarity = similarity[calibration_rows]
    similarity_thresholds = np.unique(
        np.concatenate(
            [
                np.asarray([np.inf], dtype=float),
                np.quantile(calibration_similarity, np.linspace(0.0, 1.0, 101)),
            ]
        )
    )
    probability_thresholds = np.linspace(0.0, 1.01, 102)
    gate_rows = []
    for name, probability, thresholds, complexity in (
        ("image_similarity_threshold", similarity, similarity_thresholds, 0),
        ("logistic_regression", all_logistic_probability, probability_thresholds, 1),
        ("two_layer_mlp", all_mlp_probability, probability_thresholds, 2),
    ):
        best = _threshold_search(
            probabilities=probability,
            output=output,
            image_only=image_only,
            targets=validation_targets,
            hierarchy=hierarchy,
            calibration_mask=gate_calibration,
            thresholds=thresholds,
        )
        gate_rows.append(
            {
                "name": name,
                "complexity": complexity,
                "probability": probability,
                "calibration": best,
            }
        )
    best_calibration_exact = max(
        float(row["calibration"]["metrics"]["question"]["exact_answer_set_accuracy"])
        for row in gate_rows
    )
    tie_tolerance = float(config["selection"]["tie_tolerance"])
    eligible_gate_rows = [
        row
        for row in gate_rows
        if best_calibration_exact
        - float(row["calibration"]["metrics"]["question"]["exact_answer_set_accuracy"])
        <= tie_tolerance
    ]
    selected_gate = min(
        eligible_gate_rows,
        key=lambda row: (
            int(row["complexity"]),
            int(row["calibration"]["history_use_count"]),
        ),
    )
    selected_threshold = float(selected_gate["calibration"]["threshold"])
    selected_probability = np.asarray(selected_gate["probability"])
    holdout_predictions, total_history_use = _apply_gate(
        output,
        image_only,
        hierarchy,
        selected_probability,
        selected_threshold,
    )
    image_holdout_metrics = _metrics(
        validation_targets,
        image_only,
        hierarchy,
        development_holdout,
        output.records,
    )
    selected_holdout_metrics = _metrics(
        validation_targets,
        holdout_predictions,
        hierarchy,
        development_holdout,
        output.records,
    )

    holdout_record_mask = _record_masks(output.records, development_holdout)
    chosen_history = np.logical_and(
        holdout_record_mask,
        np.logical_and(disagreement, selected_probability >= selected_threshold),
    )
    image_correct_to_history_wrong = 0
    history_only_recovery = 0
    for row_index in np.flatnonzero(chosen_history):
        record = output.records[int(row_index)]
        indices = hierarchy.indices_by_question[record.question_id]
        target = validation_targets[record.case_index, indices]
        image_correct = np.array_equal(image_only[record.case_index, indices], target)
        history_correct = np.array_equal(
            output.history_predictions[record.case_index, indices], target
        )
        image_correct_to_history_wrong += int(image_correct and not history_correct)
        history_only_recovery += int(history_correct and not image_correct)

    shuffled_exact_scores: list[float] = []
    shuffled_macro_scores: list[float] = []
    train_case_ids = [str(case["case_id"]) for case in train_meta]
    for replicate in range(int(config["controls"]["shuffled_pair_replicates"])):
        permutation = _fixed_point_free_payload_permutation(
            train_case_ids, seed=int(config["seed"]), replicate=replicate
        )
        if (
            selected_policy == "image_top1"
            and selected_gate["name"] == "image_similarity_threshold"
        ):
            shuffled_blocks = [
                CandidateBlock(
                    records=block.records,
                    bank_indices=block.bank_indices,
                    features=block.features,
                    labels=block.labels,
                    payload_answers=train_targets[permutation][block.bank_indices],
                )
                for block in blocks
            ]
        else:
            shuffled_blocks = _build_blocks(
                validation_role=validation_role,
                records_by_case=records_by_case,
                validation_targets=validation_targets,
                validation_images=validation_images,
                train_images=train_images,
                train_reports=train_reports[permutation],
                train_answers=train_targets[permutation],
                train_clusters=train_clusters,
                question_embeddings=question_map,
                hierarchy=hierarchy,
                shortlist_k=int(config["candidate_frame"]["image_shortlist_k"]),
            )
        shuffled_output = _selector_output(
            blocks=shuffled_blocks,
            policy=selected_policy,
            scaler=candidate_scaler,
            model=candidate_model,
            image_only=image_only,
            hierarchy=hierarchy,
        )
        if selected_gate["name"] == "image_similarity_threshold":
            shuffled_probability = shuffled_output.gate_numeric[:, 0]
        elif selected_gate["name"] == "logistic_regression":
            shuffled_probability = gate_logistic.predict_proba(
                logistic_encoder.transform(
                    shuffled_output.gate_numeric,
                    shuffled_output.question_ids,
                    shuffled_output.answer_types,
                )
            )[:, 1]
        else:
            shuffled_probability = mlp_gate.predict_proba(
                mlp_encoder.transform(
                    shuffled_output.gate_numeric,
                    shuffled_output.question_ids,
                    shuffled_output.answer_types,
                )
            )
        shuffled_predictions, _ = _apply_gate(
            shuffled_output,
            image_only,
            hierarchy,
            shuffled_probability,
            selected_threshold,
        )
        shuffled_metrics = _metrics(
            validation_targets,
            shuffled_predictions,
            hierarchy,
            development_holdout,
            shuffled_output.records,
        )
        shuffled_exact_scores.append(
            float(shuffled_metrics["question"]["exact_answer_set_accuracy"])
        )
        shuffled_macro_scores.append(
            float(shuffled_metrics["structured"]["supported_label_macro_f1"])
        )

    exact_delta = float(
        selected_holdout_metrics["question"]["exact_answer_set_accuracy"]
        - image_holdout_metrics["question"]["exact_answer_set_accuracy"]
    )
    option_delta = float(
        selected_holdout_metrics["question"]["option_micro_f1"]
        - image_holdout_metrics["question"]["option_micro_f1"]
    )
    macro_delta = float(
        selected_holdout_metrics["structured"]["supported_label_macro_f1"]
        - image_holdout_metrics["structured"]["supported_label_macro_f1"]
    )
    holdout_history_use = int(chosen_history.sum())
    shuffled_max = float(np.max(shuffled_exact_scores))
    go_checks = {
        "question_exact_improves": exact_delta > 0,
        "option_micro_f1_noninferior": option_delta
        >= float(config["selection"]["option_micro_f1_noninferiority_margin"]),
        "macro_f1_noninferior": macro_delta
        >= float(
            config["selection"]["supported_label_macro_f1_noninferiority_margin"]
        ),
        "history_used": holdout_history_use > 0,
        "aligned_exceeds_all_shuffled_exact": float(
            selected_holdout_metrics["question"]["exact_answer_set_accuracy"]
        )
        > shuffled_max,
    }
    go = all(go_checks.values())

    image_case_values = _exact_case_values(
        validation_role, validation_targets, image_only, cases, vectorizer, hierarchy
    )
    selected_case_values = _exact_case_values(
        validation_role,
        validation_targets,
        holdout_predictions,
        cases,
        vectorizer,
        hierarchy,
    )
    summary = {
        "study": config["study"],
        "status": "go_for_report_derived_fact_development" if go else "stop_or_redesign",
        "boundary": config["boundary"],
        "test_accessed": False,
        "new_medgemma_inference_performed": False,
        "embedding_signature": embedding_signature,
        "question_text_sha256": question_fingerprint,
        "historical_bank": {
            "manifest_train_cases": len(train_role["cases"]),
            "eligible_image_report_pairs": len(train_meta),
            "excluded_missing_embedding_pairs": len(train_role["cases"])
            - len(train_meta),
        },
        "case_partitions": {
            "selection": int(selection.sum()),
            "gate_train": int(gate_train.sum()),
            "gate_calibration": int(gate_calibration.sum()),
            "mlp_internal_train": int(mlp_train.sum()),
            "mlp_internal_early_stopping": int(mlp_early.sum()),
            "development_holdout": int(development_holdout.sum()),
        },
        "candidate_training": candidate_training,
        "candidate_selection": {
            "candidates": selector_candidates,
            "selected_policy": selected_policy,
            "calibration_exact_delta_logistic_minus_image_top1": logistic_exact
            - image_exact,
        },
        "gate_training": {
            "disagreement_rows": int(training_rows.sum()),
            "history_better_rows": int(history_better[training_rows].sum()),
            "history_better_rate": float(history_better[training_rows].mean()),
            "mlp_device": str(mlp_gate.device),
            "mlp_best_epoch": int(mlp_gate.best_epoch),
            "mlp_best_validation_loss": float(mlp_gate.best_validation_loss),
        },
        "gate_selection": {
            "candidates": [
                {
                    "name": row["name"],
                    "complexity": row["complexity"],
                    "threshold": row["calibration"]["threshold"],
                    "history_use_count": row["calibration"]["history_use_count"],
                    "metrics": row["calibration"]["metrics"],
                }
                for row in gate_rows
            ],
            "selected_model": selected_gate["name"],
            "selected_threshold": selected_threshold,
        },
        "development_holdout": {
            "image_only": image_holdout_metrics,
            "selected_gate": selected_holdout_metrics,
            "selected_gate_minus_image_only_exact": exact_delta,
            "selected_gate_minus_image_only_option_micro_f1": option_delta,
            "selected_gate_minus_image_only_macro_f1": macro_delta,
            "history_use_count": holdout_history_use,
            "history_use_rate": holdout_history_use
            / int(holdout_record_mask.sum()),
            "image_correct_to_history_wrong_count": image_correct_to_history_wrong,
            "history_only_recovery_count": history_only_recovery,
            "net_exact_question_gain_count_before_hierarchy_side_effects": history_only_recovery
            - image_correct_to_history_wrong,
            "case_bootstrap_selected_vs_image_only_exact": _case_bootstrap_exact_difference(
                selected_case_values,
                image_case_values,
                development_holdout,
                samples=5000,
                seed=int(config["seed"]) + 1,
            ),
            "shuffled_pair_control": {
                "replicates": len(shuffled_exact_scores),
                "exact_mean": float(np.mean(shuffled_exact_scores)),
                "exact_minimum": float(np.min(shuffled_exact_scores)),
                "exact_maximum": shuffled_max,
                "aligned_minus_shuffled_mean_exact": float(
                    selected_holdout_metrics["question"]["exact_answer_set_accuracy"]
                    - np.mean(shuffled_exact_scores)
                ),
                "aligned_exceeds_all_shuffles": bool(
                    selected_holdout_metrics["question"]["exact_answer_set_accuracy"]
                    > shuffled_max
                ),
                "plus_one_monte_carlo_p": float(
                    (
                        1
                        + sum(
                            score
                            >= selected_holdout_metrics["question"][
                                "exact_answer_set_accuracy"
                            ]
                            for score in shuffled_exact_scores
                        )
                    )
                    / (1 + len(shuffled_exact_scores))
                ),
                "macro_f1_mean": float(np.mean(shuffled_macro_scores)),
            },
        },
        "go_rule": {"passed": go, **go_checks},
        "interpretation": (
            "The selective gate passed the fixed development advancement rule. "
            "The next permitted step is to replace oracle answer payloads with "
            "deterministic report-derived facts; Test remains locked."
            if go
            else "The selective gate did not pass every fixed development advancement "
            "criterion. It must not be promoted or run on Test."
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
        default=ROOT / "config/final_qa_v2_selective_gate.json",
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
        "--question-embeddings",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/final_qa_v2_question_embeddings.npz",
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
        / "experiments/final_qa_development/final_qa_v2_selective_gate.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_selection": result["candidate_selection"],
                "gate_selection": result["gate_selection"],
                "development_holdout": result["development_holdout"],
                "go_rule": result["go_rule"],
            },
            indent=2,
            sort_keys=True,
        )
    )
