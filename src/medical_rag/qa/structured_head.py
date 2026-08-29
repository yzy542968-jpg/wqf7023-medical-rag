from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class StructuredHistoryHead(nn.Module):
    def __init__(
        self,
        input_features: int,
        output_labels: int,
        *,
        hidden_features: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, output_labels),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


@dataclass(frozen=True)
class FeatureBlocks:
    target: np.ndarray
    history: np.ndarray

    def combined(self, history_present: bool = True) -> np.ndarray:
        if history_present:
            history = self.history
        else:
            history = np.zeros_like(self.history)
        return np.concatenate([self.target, history], axis=1).astype(
            np.float32, copy=False
        )


def retrieve_top1_history(
    query_embeddings: np.ndarray,
    query_cluster_ids: list[str],
    bank_embeddings: np.ndarray,
    bank_cluster_ids: list[str],
    bank_report_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(query_embeddings, dtype=np.float32)
    bank = np.asarray(bank_embeddings, dtype=np.float32)
    reports = np.asarray(bank_report_embeddings, dtype=np.float32)
    if query.ndim != 2 or bank.ndim != 2 or reports.ndim != 2:
        raise ValueError("Query, bank and report embeddings must be matrices")
    if bank.shape[0] != reports.shape[0] or len(bank_cluster_ids) != bank.shape[0]:
        raise ValueError("Historical bank arrays are not aligned")
    if len(query_cluster_ids) != query.shape[0] or query.shape[1] != bank.shape[1]:
        raise ValueError("Query metadata or embedding dimensions do not align")
    similarities = query @ bank.T
    for row, cluster_id in enumerate(query_cluster_ids):
        blocked = np.asarray(
            [candidate == cluster_id for candidate in bank_cluster_ids], dtype=bool
        )
        similarities[row, blocked] = -np.inf
    if np.any(np.all(~np.isfinite(similarities), axis=1)):
        raise ValueError("At least one query has no eligible historical candidate")
    indices = np.argmax(similarities, axis=1)
    scores = similarities[np.arange(len(similarities)), indices]
    return indices, scores.astype(np.float32), reports[indices]


def history_feature_block(
    report_embeddings: np.ndarray, similarities: np.ndarray
) -> np.ndarray:
    reports = np.asarray(report_embeddings, dtype=np.float32)
    scores = np.asarray(similarities, dtype=np.float32)
    if reports.ndim != 2 or scores.shape != (reports.shape[0],):
        raise ValueError("History reports and similarities do not align")
    presence = np.ones((len(reports), 1), dtype=np.float32)
    return np.concatenate([reports, scores[:, None], presence], axis=1)


__all__ = [
    "FeatureBlocks",
    "StructuredHistoryHead",
    "history_feature_block",
    "retrieve_top1_history",
]
