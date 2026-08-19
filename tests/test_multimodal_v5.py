from __future__ import annotations

import numpy as np
import pytest

from scripts.build_v5_artifact_manifest import portable_sha256
from scripts.run_multimodal_v5_retrieval import derangement_indices


def test_v5_derangement_has_no_fixed_points_and_is_reproducible() -> None:
    first = derangement_indices(20, 7023)
    second = derangement_indices(20, 7023)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(20))
    assert not np.any(first == np.arange(20))


def test_v5_derangement_requires_two_cases() -> None:
    with pytest.raises(ValueError):
        derangement_indices(1, 7023)


def test_v5_manifest_hash_is_line_ending_independent(tmp_path) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"alpha\nbeta\n")
    crlf.write_bytes(b"alpha\r\nbeta\r\n")
    assert portable_sha256(lf) == portable_sha256(crlf)
