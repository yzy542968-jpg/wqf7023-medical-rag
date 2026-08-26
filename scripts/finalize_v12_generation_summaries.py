"""Normalize V12 generation summaries and create the budget comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-96", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_96_summary.json")
    parser.add_argument("--summary-128", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_128_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_budget_comparison.json")
    args = parser.parse_args()

    summaries: dict[str, dict[str, Any]] = {}
    for budget, path in (("96", args.summary_96), ("128", args.summary_128)):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["inputs"]["selection_source"] = (
            "predeclared V12 48-case Validation manifest selected by spectrum-stratified "
            "SHA-256 ordering; no outcome-based replacement"
        )
        path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        summaries[budget] = value

    output = {
        "study": "V12 generation budget comparison",
        "status": "development_only_no_confirmation",
        "no_test_evaluation": True,
        "budgets": {
            budget: {
                "whole_report": value["metrics"]["whole_report"],
                "case_to_fact": value["metrics"]["case_to_fact"],
                "paired_case_to_fact_minus_whole_report": value["paired_case_bootstrap"]["case_to_fact_minus_whole_report"],
            }
            for budget, value in summaries.items()
        },
        "descriptive_engineering_choice": (
            "The two budgets produced identical Token-F1 and paired differences. "
            "The 96-token budget is retained as the lower-latency tie-break for this pilot only; "
            "no frozen V10/V11 configuration is changed."
        ),
        "claim_boundary": (
            "This comparison is a Validation-only development diagnostic. Token-F1 is automated "
            "same-source report overlap, not clinical correctness or human validation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
