"""Add case-grouped uncertainty estimates to the frozen V10 2x2 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.grouped_bootstrap import grouped_bootstrap_ci  # noqa: E402


DEFAULT_ROWS = ROOT / "experiments/v10_publication/v10_fact_attention_2x2_rows.jsonl"
DEFAULT_OUTPUT = ROOT / "data/splits/v10/v10_fact_attention_2x2_bootstrap_summary.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rows = read_rows(args.rows)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[str(row["case_id"])].append(row)
    expected = {"r4_mean", "r4_attention", "r5_mean", "r5_attention"}
    if any(set(row["ndcg@10"]) != expected for row in rows):
        raise ValueError("Every row must contain the four frozen 2x2 cells")

    case_values: dict[str, dict[str, float]] = {}
    for case_id, case_rows in sorted(by_case.items()):
        values = {
            condition: statistics.fmean(float(row["ndcg@10"][condition]) for row in case_rows)
            for condition in expected
        }
        case_values[case_id] = {
            "fact_main_effect": ((values["r5_mean"] - values["r4_mean"]) + (values["r5_attention"] - values["r4_attention"])) / 2.0,
            "attention_main_effect": ((values["r4_attention"] - values["r4_mean"]) + (values["r5_attention"] - values["r5_mean"])) / 2.0,
            "fact_attention_interaction": (values["r5_attention"] - values["r5_mean"]) - (values["r4_attention"] - values["r4_mean"]),
        }

    effects = {
        name: grouped_bootstrap_ci(
            {case_id: values[name] for case_id, values in case_values.items()},
            repetitions=args.repetitions,
            seed=args.seed,
        )
        for name in ("fact_main_effect", "attention_main_effect", "fact_attention_interaction")
    }
    summary = {
        "study": "V10 facts x attention 2x2 case-grouped bootstrap",
        "status": "validation_only_test_not_run",
        "input_rows_sha256": sha256(args.rows),
        "row_count": len(rows),
        "case_count": len(case_values),
        "unit_of_resampling": "case-level mean over the three fixed question rows",
        "effects": effects,
        "claim_boundary": "Uncertainty estimates describe the fixed Validation checkpoint comparison; they do not establish causal effects, clinical correctness, safety or external validity.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
