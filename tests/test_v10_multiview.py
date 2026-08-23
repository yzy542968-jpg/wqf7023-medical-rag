from __future__ import annotations

import numpy as np
import torch

from medical_rag.similar_case.v10_multiview import (
    ViewAttention,
    attention_query_embedding,
    attention_record_loss,
    l2_normalize,
    make_attention_record,
    max_view_scores,
    mean_view_scores,
)


def test_l2_normalize_handles_zero_rows() -> None:
    result = l2_normalize(np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.allclose(result[0], [0.6, 0.8])
    assert np.allclose(result[1], [0.0, 0.0])


def test_mean_and_max_view_scores_are_deterministic_and_bounded() -> None:
    views = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    candidates = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    mean = mean_view_scores(views, candidates, candidates)
    maximum = max_view_scores(views, candidates, candidates)
    assert np.all((0.0 <= mean) & (mean <= 1.0))
    assert np.all((0.0 <= maximum) & (maximum <= 1.0))
    assert maximum[0] == maximum[1]


def test_attention_record_uses_only_valid_gain_pairs() -> None:
    views = np.asarray([[1.0, 0.0]], dtype=np.float32)
    candidates = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    gains = np.asarray([1.0, 0.5, -np.inf], dtype=np.float32)
    record = make_attention_record(
        views,
        candidates,
        gains,
        high_candidates=1,
        low_candidates=1,
        minimum_gain_difference=0.05,
    )
    assert record is not None
    assert record.pair_differences.shape == (1, 2)
    assert np.allclose(record.weights, [0.5])


def test_attention_loss_backpropagates() -> None:
    record = make_attention_record(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32),
        np.asarray([1.0, 0.0], dtype=np.float32),
        high_candidates=1,
        low_candidates=1,
        minimum_gain_difference=0.05,
    )
    assert record is not None
    model = ViewAttention(width=2)
    loss = attention_record_loss(model, record)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.projection.weight.grad is not None


def test_attention_query_embedding_is_normalized() -> None:
    models = [ViewAttention(width=2), ViewAttention(width=2)]
    embedding = attention_query_embedding(
        models,
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    assert np.isclose(np.linalg.norm(embedding), 1.0)
