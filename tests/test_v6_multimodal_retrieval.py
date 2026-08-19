from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v6_development_multimodal_retrieval.py"
SPEC = importlib.util.spec_from_file_location("v6_multimodal", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_aggregate_chunk_embeddings_means_then_normalizes() -> None:
    chunks = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
    actual = MODULE.aggregate_chunk_embeddings(chunks, ["a", "a", "b"], ["a", "b"])

    assert np.allclose(actual[0], [2**-0.5, 2**-0.5])
    assert np.allclose(actual[1], [0.0, 1.0])


def test_maximum_chunk_scores_uses_best_chunk_per_case() -> None:
    chunks = np.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=np.float32)
    scores = MODULE.maximum_chunk_scores(
        np.asarray([1.0, 0.0], dtype=np.float32),
        chunks,
        ["a", "a", "b"],
        ["a", "b"],
    )

    assert scores == {"a": 1.0, "b": 0.0}


def test_chunk_policy_selection_requires_frozen_margin() -> None:
    tied = MODULE.select_chunk_policy(0.50, 0.5049, 0.005)
    improved = MODULE.select_chunk_policy(0.50, 0.505, 0.005)

    assert tied["selected_chunk_policy"] == "normalized_mean_chunk_embedding"
    assert improved["selected_chunk_policy"] == "maximum_image_chunk_cosine"
