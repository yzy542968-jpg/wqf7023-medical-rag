from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_v13_concept_qa_manifest import (  # noqa: E402
    case_id_fingerprint,
    selection_digest,
    spectrum,
)


def test_manifest_hashes_are_deterministic() -> None:
    assert selection_digest(" CXR1 ") == selection_digest("CXR1")
    assert selection_digest("CXR1") != selection_digest("CXR2")
    assert case_id_fingerprint(["B", "A", "A"]) == case_id_fingerprint(["A", "B"])


def test_manifest_spectrum_uses_report_index_terms() -> None:
    assert spectrum({"problems": "Normal"}) == "normal"
    assert spectrum({"problems": "No Indexing"}) == "indeterminate"
    assert spectrum({"problems": "Cardiomegaly"}) == "abnormal"
