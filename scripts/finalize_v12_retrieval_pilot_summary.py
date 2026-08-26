"""Finalize the V12 retrieval summary from the completed Validation audit.

The ranking audit already stores the same saved LambdaMART model's flattened
Validation rankings plus qrel sensitivity results.  This utility converts
that completed artifact into the compact pilot summary schema without
retraining or reopening any partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_validation_rankings.json")
    parser.add_argument("--ranking-rows", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_validation_ranking_rows.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_retrieval_pilot.json")
    parser.add_argument("--rows-output", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_retrieval_pilot_rows.jsonl")
    args = parser.parse_args()

    audit = json.loads(args.rankings.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.ranking_rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    qrel = "qrel_v2"
    systems = ("r5_full_bank", "rrf_candidate", "rrf_r5_rerank", "rrf_lambdamart")
    compact_rows: list[dict[str, Any]] = []
    for row in rows:
        compact = {
            "case_id": row["case_id"],
            "question_type": row["question_type"],
            "spectrum": row["spectrum"],
        }
        for system in systems:
            compact[system] = float(row["metrics"][system][qrel])
            compact[f"{system}_top200"] = row["rankings"][system][:200]
        compact["target_in_rrf_top200"] = 0.0
        compact_rows.append(compact)
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        "".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in compact_rows),
        encoding="utf-8",
    )
    metrics = {
        system: audit["metrics"][system][qrel]
        for system in systems
    }
    output = {
        "study": "V12 retrieval pilot",
        "status": "validation_only_development",
        "no_test_evaluation": True,
        "source_audit": str(args.rankings.resolve().relative_to(ROOT)),
        "inputs": {
            **audit["inputs"],
            "ranking_rows_sha256": file_sha256(args.ranking_rows),
        },
        "ranker": {
            "type": "LightGBM LambdaMART",
            "version": "4.7.0",
            "features": 17,
            "candidate_pool": "RRF Top-200",
            "qrel": "qrel-v2 full report-derived proxy",
            "model_sha256": audit["inputs"]["model_sha256"],
            "model_path": "experiments/v12_optimization/retrieval/v12_lambdamart.txt",
        },
        "metrics": metrics,
        "bootstrap_vs_r5": audit["bootstrap_vs_r5"],
        "rows_path": str(args.rows_output.resolve().relative_to(ROOT)),
        "rows_sha256": file_sha256(args.rows_output),
        "claim_boundary": (
            "This is a Validation-only development pilot. qrel-v2, label-only and fact-only scores are "
            "report-derived proxies; none is physician-adjudicated clinical correctness. The learned result "
            "must not be promoted to confirmation evidence without a newly frozen protocol and cohort."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
