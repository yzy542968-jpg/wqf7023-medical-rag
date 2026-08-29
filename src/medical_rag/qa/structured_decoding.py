from __future__ import annotations

import numpy as np

from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy


def decode_answer_probabilities(
    probabilities: np.ndarray,
    hierarchy: RadReStructHierarchy,
    *,
    multi_choice_threshold: float,
    fixed_choice_threshold: float,
) -> np.ndarray:
    """Decode answer probabilities without using target-answer history."""

    values = np.asarray(probabilities, dtype=np.float32)
    expected_labels = len(hierarchy.report_keys)
    if values.ndim != 2 or values.shape[1] != expected_labels:
        raise ValueError(
            f"Probabilities must have shape (cases, {expected_labels})"
        )
    if not np.isfinite(values).all() or (
        (values < -1e-6) | (values > 1 + 1e-6)
    ).any():
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    values = np.clip(values, 0.0, 1.0)
    for threshold in (multi_choice_threshold, fixed_choice_threshold):
        if not 0 <= threshold <= 1:
            raise ValueError("Decoding thresholds must lie in [0, 1]")

    decoded = np.zeros(values.shape, dtype=np.uint8)
    for question_id, choice in hierarchy.choice_options.items():
        indices = hierarchy.indices_by_question[question_id]
        group = values[:, indices]
        if choice == "single_choice":
            winners = np.argmax(group, axis=1)
            decoded[np.arange(len(decoded)), indices[winners]] = 1
        elif choice == "multi_choice":
            decoded[:, indices] = group >= multi_choice_threshold
        elif choice == "fixed_choice":
            decoded[:, indices] = group >= fixed_choice_threshold
        else:
            raise ValueError(
                f"Unsupported Rad-ReStruct choice type for {question_id}: {choice}"
            )
    return hierarchy.clean(decoded)


def knn_answer_probabilities(
    similarities: np.ndarray,
    bank_targets: np.ndarray,
    *,
    top_k: int,
    weighting: str,
    softmax_temperature: float,
) -> np.ndarray:
    scores = np.asarray(similarities, dtype=np.float32)
    targets = np.asarray(bank_targets, dtype=np.float32)
    if scores.ndim != 2 or targets.ndim != 2:
        raise ValueError("Similarities and bank targets must be matrices")
    if scores.shape[1] != targets.shape[0]:
        raise ValueError("Similarity columns must align with bank-target rows")
    if not 1 <= top_k <= targets.shape[0]:
        raise ValueError("top_k is outside the historical bank")
    if softmax_temperature <= 0:
        raise ValueError("softmax_temperature must be positive")

    order = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
    neighbor_scores = np.take_along_axis(scores, order, axis=1)
    neighbor_targets = targets[order]
    if weighting == "uniform":
        weights = np.full(neighbor_scores.shape, 1.0 / top_k, dtype=np.float32)
    elif weighting == "cosine_softmax":
        logits = neighbor_scores / softmax_temperature
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown historical-neighbor weighting: {weighting}")
    return np.einsum("nk,nkl->nl", weights, neighbor_targets, optimize=True)


__all__ = ["decode_answer_probabilities", "knn_answer_probabilities"]
