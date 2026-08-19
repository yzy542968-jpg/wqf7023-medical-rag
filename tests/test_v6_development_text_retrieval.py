from __future__ import annotations

import numpy as np

from scripts.run_v6_development_text_retrieval import (
    dense_rankings,
    detailed_instruction,
    select_text_retriever,
)


def test_qwen3_detailed_instruction_matches_official_format() -> None:
    assert detailed_instruction("Retrieve reports.", "Clinical indication: cough") == (
        "Instruct: Retrieve reports.\nQuery:Clinical indication: cough"
    )


def test_t_star_requires_material_qwen3_mrr_advantage() -> None:
    tied = select_text_retriever(0.500, 0.5049, 0.005)
    assert tied["selected_text_retriever"] == "bm25"

    material = select_text_retriever(0.500, 0.505, 0.005)
    assert material["selected_text_retriever"] == "qwen3_embedding"

    worse = select_text_retriever(0.500, 0.450, 0.005)
    assert worse["selected_text_retriever"] == "bm25"


def test_dense_rankings_use_stable_case_id_tie_break() -> None:
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    documents = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    rankings = dense_rankings(
        ["q1"], ["CXR2", "CXR1", "CXR3"], queries, documents
    )
    assert rankings["q1"] == ["CXR1", "CXR2", "CXR3"]
