from __future__ import annotations

import numpy as np
import pytest

from medical_rag.similar_case.v10_loader import _unique_embedding_map


def test_embedding_map_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        _unique_embedding_map(
            ["a", "b"],
            np.zeros((1, 4), dtype=np.float32),
            label="test",
        )


def test_embedding_map_rejects_duplicate_case_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _unique_embedding_map(
            ["a", "a"],
            np.zeros((2, 4), dtype=np.float32),
            label="test",
        )


@pytest.mark.parametrize(
    ("identifiers", "embeddings", "message"),
    [
        ([""], np.zeros((1, 4), dtype=np.float32), "blank"),
        (["a"], np.asarray([[np.nan, 0.0]], dtype=np.float32), "non-finite"),
        (["a"], np.zeros((1, 0), dtype=np.float32), "feature column"),
    ],
)
def test_embedding_map_rejects_invalid_payloads(
    identifiers: list[str], embeddings: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _unique_embedding_map(identifiers, embeddings, label="test")
