from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    return np.divide(array, norms, out=np.zeros_like(array), where=norms > 1e-12)


def minmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(array)
    result = np.zeros_like(array)
    if not np.any(finite):
        return result
    lower = float(np.min(array[finite]))
    upper = float(np.max(array[finite]))
    if upper - lower > 1e-12:
        result[finite] = (array[finite] - lower) / (upper - lower)
    return result


def fused_component_score(image_image: np.ndarray, image_report: np.ndarray) -> np.ndarray:
    return 0.5 * minmax(image_image) + 0.5 * minmax(image_report)


def mean_view_scores(
    query_views: np.ndarray,
    candidate_images: np.ndarray,
    candidate_reports: np.ndarray,
) -> np.ndarray:
    query = l2_normalize(np.mean(l2_normalize(query_views), axis=0))
    return fused_component_score(candidate_images @ query, candidate_reports @ query)


def max_view_scores(
    query_views: np.ndarray,
    candidate_images: np.ndarray,
    candidate_reports: np.ndarray,
) -> np.ndarray:
    views = l2_normalize(query_views)
    image_image = np.max(candidate_images @ views.T, axis=1)
    image_report = np.max(candidate_reports @ views.T, axis=1)
    return fused_component_score(image_image, image_report)


class ViewAttention(nn.Module):
    def __init__(self, width: int = 1152) -> None:
        super().__init__()
        self.projection = nn.Linear(width, 1)

    def aggregate(self, views: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(views, dim=-1)
        weights = torch.softmax(self.projection(normalized).squeeze(-1), dim=0)
        return F.normalize(torch.sum(weights[:, None] * normalized, dim=0), dim=0)

    def forward(self, views: torch.Tensor, candidate_vectors: torch.Tensor) -> torch.Tensor:
        return candidate_vectors @ self.aggregate(views)


def attention_view_scores(
    models: Sequence[ViewAttention],
    query_views: np.ndarray,
    candidate_images: np.ndarray,
    candidate_reports: np.ndarray,
) -> np.ndarray:
    views = torch.from_numpy(np.asarray(query_views, dtype=np.float32))
    image_tensor = torch.from_numpy(np.asarray(candidate_images, dtype=np.float32))
    report_tensor = torch.from_numpy(np.asarray(candidate_reports, dtype=np.float32))
    image_scores = []
    report_scores = []
    with torch.inference_mode():
        for model in models:
            model.eval()
            query = model.aggregate(views)
            image_scores.append((image_tensor @ query).numpy())
            report_scores.append((report_tensor @ query).numpy())
    return fused_component_score(
        np.mean(np.stack(image_scores), axis=0),
        np.mean(np.stack(report_scores), axis=0),
    )


def attention_query_embedding(models: Sequence[ViewAttention], query_views: np.ndarray) -> np.ndarray:
    views = torch.from_numpy(np.asarray(query_views, dtype=np.float32))
    with torch.inference_mode():
        embeddings = [model.aggregate(views).numpy() for model in models]
    return l2_normalize(np.mean(np.stack(embeddings), axis=0))


@dataclass(frozen=True)
class AttentionTrainingRecord:
    views: np.ndarray
    pair_differences: np.ndarray
    weights: np.ndarray


def make_attention_record(
    query_views: np.ndarray,
    candidate_vectors: np.ndarray,
    gains: np.ndarray,
    *,
    high_candidates: int,
    low_candidates: int,
    minimum_gain_difference: float,
) -> AttentionTrainingRecord | None:
    valid = np.isfinite(gains)
    indices = np.flatnonzero(valid).tolist()
    high = sorted(indices, key=lambda index: (-float(gains[index]), index))[:high_candidates]
    low = sorted(indices, key=lambda index: (float(gains[index]), index))[:low_candidates]
    differences = []
    weights = []
    for high_index in high:
        for low_index in low:
            difference = float(gains[high_index] - gains[low_index])
            if difference + 1e-12 < minimum_gain_difference:
                continue
            differences.append(candidate_vectors[high_index] - candidate_vectors[low_index])
            weights.append(difference)
    if not differences:
        return None
    return AttentionTrainingRecord(
        views=np.asarray(query_views, dtype=np.float32),
        pair_differences=np.asarray(differences, dtype=np.float32),
        weights=np.asarray(weights, dtype=np.float32),
    )


def attention_record_loss(model: ViewAttention, record: AttentionTrainingRecord) -> torch.Tensor:
    query = model.aggregate(torch.from_numpy(record.views))
    differences = torch.from_numpy(record.pair_differences)
    weights = torch.from_numpy(record.weights)
    margins = differences @ query
    return (F.softplus(-margins) * weights).mean()


__all__ = [
    "AttentionTrainingRecord",
    "ViewAttention",
    "attention_record_loss",
    "attention_query_embedding",
    "attention_view_scores",
    "fused_component_score",
    "l2_normalize",
    "make_attention_record",
    "max_view_scores",
    "mean_view_scores",
    "minmax",
]
