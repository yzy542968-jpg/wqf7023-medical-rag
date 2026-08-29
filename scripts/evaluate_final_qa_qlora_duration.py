"""Apply the frozen sequential duration rule to two Final-QA adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_final_qa_qlora_pilot import (
    BASE_CONDITIONS,
    bootstrap_difference,
    keyed,
    metrics,
    read_jsonl,
)


ROOT = Path(__file__).resolve().parents[1]


def run(args: argparse.Namespace) -> dict[str, Any]:
    previous_rows = read_jsonl(args.previous_rows)
    candidate_rows = read_jsonl(args.candidate_rows)
    comparisons: dict[str, Any] = {}
    micro_gains: list[float] = []
    condition_passes: list[bool] = []
    for label, condition in BASE_CONDITIONS.items():
        previous = keyed(previous_rows, condition)
        candidate = keyed(candidate_rows, condition)
        if set(previous) != set(candidate):
            raise RuntimeError(f"Duration candidates use different rows for {label}")
        previous_metrics = metrics(previous.values())
        candidate_metrics = metrics(candidate.values())
        micro_gain = float(candidate_metrics["option_micro_f1"]) - float(
            previous_metrics["option_micro_f1"]
        )
        validity_gain = float(candidate_metrics["contract_valid_rate"]) - float(
            previous_metrics["contract_valid_rate"]
        )
        micro_gains.append(micro_gain)
        condition_passes.append(micro_gain >= -0.005 and validity_gain >= -0.010)
        comparisons[label] = {
            "previous": previous_metrics,
            "candidate": candidate_metrics,
            "candidate_minus_previous": {
                "option_micro_f1": micro_gain,
                "exact_answer_set_accuracy": float(
                    candidate_metrics["exact_answer_set_accuracy"]
                )
                - float(previous_metrics["exact_answer_set_accuracy"]),
                "contract_valid_rate": validity_gain,
            },
            "case_grouped_bootstrap": bootstrap_difference(candidate, previous),
        }
    mean_micro_gain = sum(micro_gains) / len(micro_gains)
    advanced = mean_micro_gain >= 0.010 and all(condition_passes)
    summary = {
        "study": "Final QA sequential QLoRA duration selection",
        "previous_forward_steps": args.previous_steps,
        "candidate_forward_steps": args.candidate_steps,
        "conditions": comparisons,
        "mean_option_micro_f1_gain": mean_micro_gain,
        "prespecified_advancement_rule_passed": advanced,
        "next_action": "train_next_prespecified_duration" if advanced else "retain_previous_duration",
        "boundary": "Calibration duration selection only; no Validation or Test access.",
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous-rows", type=Path, required=True)
    parser.add_argument("--candidate-rows", type=Path, required=True)
    parser.add_argument("--previous-steps", type=int, required=True)
    parser.add_argument("--candidate-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
