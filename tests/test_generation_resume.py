from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_hf_generation import (
    _acquire_output_lock,
    _deduplicate_existing_output,
)


def test_resume_deduplicates_qids_before_appending(tmp_path: Path) -> None:
    output = tmp_path / "generations.jsonl"
    rows = [{"qid": "q1", "answer": "first"}, {"qid": "q1", "answer": "duplicate"}, {"qid": "q2", "answer": "second"}]
    output.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    assert _deduplicate_existing_output(output) == 2
    saved = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["qid"] for row in saved] == ["q1", "q2"]
    assert saved[0]["answer"] == "first"


def test_output_lock_rejects_concurrent_writer(tmp_path: Path) -> None:
    output = tmp_path / "generations.jsonl"
    lock = _acquire_output_lock(output)
    try:
        with pytest.raises(RuntimeError):
            _acquire_output_lock(output)
    finally:
        lock.unlink(missing_ok=True)
