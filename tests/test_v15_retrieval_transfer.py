from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_v15_retrieval_transfer_generation import paired_bootstrap, summarize


def row(case_id: str, question_type: str, score: float, *, proxy: bool = False) -> dict:
    return {
        "case_id": case_id,
        "question_type": question_type,
        "reference_is_proxy": proxy,
        "token_f1": score,
        "answer_only_contract_valid": 1.0,
        "evidence_provenance_valid": 1.0,
        "hit_token_ceiling": 0.0,
        "input_tokens": 10,
        "output_tokens": 3,
        "latency_seconds": 0.1,
    }


def test_v15_primary_scope_excludes_proxy_rows() -> None:
    rows = [
        row("A", "findings", 0.5),
        row("A", "impression", 0.7),
        row("A", "acute", 1.0, proxy=True),
    ]
    assert summarize(rows, primary_only=False)["row_count"] == 3
    primary = summarize(rows, primary_only=True)
    assert primary["row_count"] == 2
    assert primary["token_f1"] == pytest.approx(0.6)


def test_v15_bootstrap_is_paired_by_case() -> None:
    baseline = [row("A", "findings", 0.1), row("B", "findings", 0.2)]
    deeper = [row("A", "findings", 0.3), row("B", "findings", 0.4)]
    result = paired_bootstrap(baseline, deeper, primary_only=True, iterations=100, seed=2)
    assert result["case_count"] == 2
    assert result["mean_difference"] == pytest.approx(0.2)
    assert result["ci_95_low"] > 0.0


def test_v15_bootstrap_rejects_unpaired_rows() -> None:
    with pytest.raises(RuntimeError):
        paired_bootstrap(
            [row("A", "findings", 0.1)],
            [row("B", "findings", 0.2)],
            primary_only=True,
        )
