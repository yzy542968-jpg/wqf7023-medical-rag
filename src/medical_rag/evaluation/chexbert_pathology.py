from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


CHEXBERT_LABELS = (
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
)
CHEXBERT_FIVE = (
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
)
CHEXBERT_FIVE_INDICES = np.asarray(
    [CHEXBERT_LABELS.index(label) for label in CHEXBERT_FIVE], dtype=np.int64
)
METRIC_NAMES = (
    "micro_f1_14",
    "macro_f1_14",
    "micro_f1_5",
    "macro_f1_5",
    "exact_set_accuracy_5",
    "mean_reference_positive_recall",
    "mean_predicted_positive_precision",
    "positive_label_hamming_agreement",
)


def logits_to_rrg_binary(class_ids: np.ndarray) -> np.ndarray:
    """Apply the binary conversion used by f1chexbert's `rrg` mode."""

    values = np.asarray(class_ids)
    if values.ndim != 2 or values.shape[1] != len(CHEXBERT_LABELS):
        raise ValueError("CheXbert class IDs must have shape (n, 14)")
    if np.any((values < 0) | (values > 3)):
        raise ValueError("CheXbert class IDs must be in [0, 3]")
    return np.isin(values, (1, 3)).astype(np.int8)


def label_texts_batched(
    labeler: object,
    texts: Sequence[str],
    *,
    batch_size: int = 64,
) -> np.ndarray:
    """Run the official F1CheXbert model in batches without changing its labels."""

    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    outputs: list[np.ndarray] = []
    tokenizer = getattr(labeler, "tokenizer")
    model = getattr(labeler, "model")
    device = getattr(labeler, "device")
    for start in range(0, len(texts), batch_size):
        batch = [str(text or "") for text in texts[start : start + batch_size]]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.inference_mode():
            heads = model(
                encoded["input_ids"].to(device),
                encoded["attention_mask"].to(device),
            )
        class_ids = torch.stack([head.argmax(dim=1) for head in heads], dim=1)
        outputs.append(logits_to_rrg_binary(class_ids.cpu().numpy()))
    if not outputs:
        return np.empty((0, len(CHEXBERT_LABELS)), dtype=np.int8)
    return np.concatenate(outputs, axis=0)


@dataclass(frozen=True)
class CaseStatistics:
    case_ids: tuple[str, ...]
    true_positive: np.ndarray
    false_positive: np.ndarray
    false_negative: np.ndarray
    row_count: np.ndarray
    exact_five_count: np.ndarray
    reference_recall_sum: np.ndarray
    reference_recall_count: np.ndarray
    prediction_precision_sum: np.ndarray
    prediction_precision_count: np.ndarray

    def __post_init__(self) -> None:
        case_count = len(self.case_ids)
        if self.true_positive.shape != (case_count, len(CHEXBERT_LABELS)):
            raise ValueError("Invalid case statistic shape")
        for values in (self.false_positive, self.false_negative):
            if values.shape != self.true_positive.shape:
                raise ValueError("Case label statistic shapes differ")
        for values in (
            self.row_count,
            self.exact_five_count,
            self.reference_recall_sum,
            self.reference_recall_count,
            self.prediction_precision_sum,
            self.prediction_precision_count,
        ):
            if values.shape != (case_count,):
                raise ValueError("Invalid per-case statistic shape")


def build_case_statistics(
    case_ids: Sequence[str],
    references: np.ndarray,
    predictions: np.ndarray,
) -> CaseStatistics:
    references = np.asarray(references, dtype=np.int8)
    predictions = np.asarray(predictions, dtype=np.int8)
    if references.shape != predictions.shape:
        raise ValueError("Reference and prediction label arrays differ")
    if references.ndim != 2 or references.shape[1] != len(CHEXBERT_LABELS):
        raise ValueError("Label arrays must have shape (n, 14)")
    if len(case_ids) != references.shape[0]:
        raise ValueError("Case IDs and label arrays differ in length")
    if np.any((references < 0) | (references > 1)) or np.any(
        (predictions < 0) | (predictions > 1)
    ):
        raise ValueError("Binary labels must contain only 0 and 1")

    unique_case_ids = tuple(sorted({str(case_id) for case_id in case_ids}))
    position = {case_id: index for index, case_id in enumerate(unique_case_ids)}
    shape = (len(unique_case_ids), len(CHEXBERT_LABELS))
    tp = np.zeros(shape, dtype=np.int64)
    fp = np.zeros(shape, dtype=np.int64)
    fn = np.zeros(shape, dtype=np.int64)
    row_count = np.zeros(len(unique_case_ids), dtype=np.int64)
    exact_five_count = np.zeros(len(unique_case_ids), dtype=np.int64)
    reference_recall_sum = np.zeros(len(unique_case_ids), dtype=np.float64)
    reference_recall_count = np.zeros(len(unique_case_ids), dtype=np.int64)
    prediction_precision_sum = np.zeros(len(unique_case_ids), dtype=np.float64)
    prediction_precision_count = np.zeros(len(unique_case_ids), dtype=np.int64)

    for row_index, raw_case_id in enumerate(case_ids):
        case_index = position[str(raw_case_id)]
        reference = references[row_index]
        prediction = predictions[row_index]
        intersection = int(np.logical_and(reference == 1, prediction == 1).sum())
        reference_count = int(reference.sum())
        prediction_count = int(prediction.sum())
        tp[case_index] += np.logical_and(reference == 1, prediction == 1)
        fp[case_index] += np.logical_and(reference == 0, prediction == 1)
        fn[case_index] += np.logical_and(reference == 1, prediction == 0)
        row_count[case_index] += 1
        exact_five_count[case_index] += int(
            np.array_equal(
                reference[CHEXBERT_FIVE_INDICES],
                prediction[CHEXBERT_FIVE_INDICES],
            )
        )
        if reference_count:
            reference_recall_sum[case_index] += intersection / reference_count
            reference_recall_count[case_index] += 1
        if prediction_count:
            prediction_precision_sum[case_index] += intersection / prediction_count
            prediction_precision_count[case_index] += 1

    return CaseStatistics(
        case_ids=unique_case_ids,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
        row_count=row_count,
        exact_five_count=exact_five_count,
        reference_recall_sum=reference_recall_sum,
        reference_recall_count=reference_recall_count,
        prediction_precision_sum=prediction_precision_sum,
        prediction_precision_count=prediction_precision_count,
    )


def _safe_divide(numerator: np.ndarray | float, denominator: np.ndarray | float):
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    result = np.zeros(np.broadcast_shapes(numerator_array.shape, denominator_array.shape))
    np.divide(numerator_array, denominator_array, out=result, where=denominator_array != 0)
    return result


def metrics_from_case_statistics(
    statistics: CaseStatistics,
    selected_indices: Iterable[int] | None = None,
) -> dict[str, object]:
    indices = (
        np.arange(len(statistics.case_ids), dtype=np.int64)
        if selected_indices is None
        else np.asarray(list(selected_indices), dtype=np.int64)
    )
    tp = statistics.true_positive[indices].sum(axis=0)
    fp = statistics.false_positive[indices].sum(axis=0)
    fn = statistics.false_negative[indices].sum(axis=0)
    f1 = _safe_divide(2.0 * tp, 2.0 * tp + fp + fn)
    five_f1 = f1[CHEXBERT_FIVE_INDICES]
    row_count = int(statistics.row_count[indices].sum())
    micro_14 = float(_safe_divide(2.0 * tp.sum(), 2.0 * tp.sum() + fp.sum() + fn.sum()))
    five_tp = tp[CHEXBERT_FIVE_INDICES]
    five_fp = fp[CHEXBERT_FIVE_INDICES]
    five_fn = fn[CHEXBERT_FIVE_INDICES]
    micro_5 = float(
        _safe_divide(
            2.0 * five_tp.sum(),
            2.0 * five_tp.sum() + five_fp.sum() + five_fn.sum(),
        )
    )
    per_condition = {
        label: {
            "precision": float(_safe_divide(tp[index], tp[index] + fp[index])),
            "recall": float(_safe_divide(tp[index], tp[index] + fn[index])),
            "f1": float(f1[index]),
            "reference_positive_support": int(tp[index] + fn[index]),
            "predicted_positive_count": int(tp[index] + fp[index]),
            "true_positive_count": int(tp[index]),
            "omission_count": int(fn[index]),
            "addition_count": int(fp[index]),
        }
        for index, label in enumerate(CHEXBERT_LABELS)
    }
    return {
        "case_count": int(len(indices)),
        "row_count": row_count,
        "micro_f1_14": micro_14,
        "macro_f1_14": float(f1.mean()),
        "micro_f1_5": micro_5,
        "macro_f1_5": float(five_f1.mean()),
        "exact_set_accuracy_5": float(
            _safe_divide(statistics.exact_five_count[indices].sum(), row_count)
        ),
        "mean_reference_positive_recall": float(
            _safe_divide(
                statistics.reference_recall_sum[indices].sum(),
                statistics.reference_recall_count[indices].sum(),
            )
        ),
        "reference_positive_recall_row_count": int(
            statistics.reference_recall_count[indices].sum()
        ),
        "mean_predicted_positive_precision": float(
            _safe_divide(
                statistics.prediction_precision_sum[indices].sum(),
                statistics.prediction_precision_count[indices].sum(),
            )
        ),
        "predicted_positive_precision_row_count": int(
            statistics.prediction_precision_count[indices].sum()
        ),
        "positive_label_hamming_agreement": float(
            1.0 - _safe_divide(fp.sum() + fn.sum(), row_count * len(CHEXBERT_LABELS))
        ),
        "reference_positive_omission_count": int(fn.sum()),
        "predicted_positive_addition_count": int(fp.sum()),
        "per_condition": per_condition,
    }


def paired_case_bootstrap(
    left: CaseStatistics,
    right: CaseStatistics,
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if left.case_ids != right.case_ids:
        raise ValueError("Paired bootstrap requires identical ordered case IDs")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(seed)
    differences = {metric: np.empty(iterations, dtype=np.float64) for metric in METRIC_NAMES}
    case_count = len(left.case_ids)
    for iteration in range(iterations):
        sample = rng.integers(0, case_count, size=case_count)
        left_metrics = metrics_from_case_statistics(left, sample)
        right_metrics = metrics_from_case_statistics(right, sample)
        for metric in METRIC_NAMES:
            differences[metric][iteration] = float(left_metrics[metric]) - float(
                right_metrics[metric]
            )
    observed_left = metrics_from_case_statistics(left)
    observed_right = metrics_from_case_statistics(right)
    return {
        metric: {
            "mean_difference": float(observed_left[metric])
            - float(observed_right[metric]),
            "ci_95_low": float(np.quantile(values, 0.025)),
            "ci_95_high": float(np.quantile(values, 0.975)),
        }
        for metric, values in differences.items()
    }


def random_control_case_bootstrap(
    selected_history: CaseStatistics,
    random_history_assignments: Sequence[CaseStatistics],
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if not random_history_assignments:
        raise ValueError("At least one random-history assignment is required")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if any(
        assignment.case_ids != selected_history.case_ids
        for assignment in random_history_assignments
    ):
        raise ValueError("Random-control bootstrap requires identical ordered case IDs")
    rng = np.random.default_rng(seed)
    differences = {metric: np.empty(iterations, dtype=np.float64) for metric in METRIC_NAMES}
    case_count = len(selected_history.case_ids)
    for iteration in range(iterations):
        sample = rng.integers(0, case_count, size=case_count)
        selected_metrics = metrics_from_case_statistics(selected_history, sample)
        random_metrics = [
            metrics_from_case_statistics(assignment, sample)
            for assignment in random_history_assignments
        ]
        for metric in METRIC_NAMES:
            differences[metric][iteration] = float(selected_metrics[metric]) - float(
                np.mean([float(values[metric]) for values in random_metrics])
            )
    selected_point = metrics_from_case_statistics(selected_history)
    random_points = [
        metrics_from_case_statistics(assignment) for assignment in random_history_assignments
    ]
    return {
        metric: {
            "mean_difference": float(selected_point[metric])
            - float(np.mean([float(values[metric]) for values in random_points])),
            "ci_95_low": float(np.quantile(samples, 0.025)),
            "ci_95_high": float(np.quantile(samples, 0.975)),
        }
        for metric, samples in differences.items()
    }


def case_bootstrap_intervals(
    statistics: CaseStatistics,
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rng = np.random.default_rng(seed)
    values = {metric: np.empty(iterations, dtype=np.float64) for metric in METRIC_NAMES}
    case_count = len(statistics.case_ids)
    for iteration in range(iterations):
        sample = rng.integers(0, case_count, size=case_count)
        metrics = metrics_from_case_statistics(statistics, sample)
        for metric in METRIC_NAMES:
            values[metric][iteration] = float(metrics[metric])
    observed = metrics_from_case_statistics(statistics)
    return {
        metric: {
            "mean": float(observed[metric]),
            "ci_95_low": float(np.quantile(samples, 0.025)),
            "ci_95_high": float(np.quantile(samples, 0.975)),
        }
        for metric, samples in values.items()
    }
