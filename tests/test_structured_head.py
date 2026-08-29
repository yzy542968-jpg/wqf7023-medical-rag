from __future__ import annotations

import numpy as np
import pytest

from medical_rag.qa.structured_head import (
    FeatureBlocks,
    history_feature_block,
    retrieve_top1_history,
)


def test_retrieval_excludes_same_cluster() -> None:
    query = np.asarray([[1.0, 0.0]], dtype=np.float32)
    bank = np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
    reports = np.asarray([[1.0], [2.0]], dtype=np.float32)
    indices, scores, selected = retrieve_top1_history(
        query,
        ["duplicate"],
        bank,
        ["duplicate", "other"],
        reports,
    )
    assert indices.tolist() == [1]
    assert scores.tolist() == pytest.approx([0.8])
    assert selected.tolist() == [[2.0]]


def test_feature_blocks_zero_complete_history_channel() -> None:
    target = np.asarray([[1.0, 2.0]], dtype=np.float32)
    history = history_feature_block(
        np.asarray([[3.0, 4.0]], dtype=np.float32),
        np.asarray([0.5], dtype=np.float32),
    )
    blocks = FeatureBlocks(target, history)
    assert blocks.combined(True).tolist() == [[1.0, 2.0, 3.0, 4.0, 0.5, 1.0]]
    assert blocks.combined(False).tolist() == [[1.0, 2.0, 0.0, 0.0, 0.0, 0.0]]
