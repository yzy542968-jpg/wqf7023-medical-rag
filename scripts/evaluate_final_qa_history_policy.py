"""Apply the frozen Calibration rule to Final-QA history policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_final_qa_qlora_pilot import bootstrap_difference, keyed, metrics, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
B3 = "b3_no_history_r2"
IMAGE_P1 = "p1_top3_image_neighbors_question_conditioned_evidence"
V12_P1 = "p1_v12_lambdamart_top3_question_conditioned_evidence"


def negative_transfer(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, float | int | None]:
    eligible = sum(
        set(row["predicted_indices"]) == set(row["gold_indices"])
        for row in baseline.values()
    )
    count = sum(
        set(baseline[key]["predicted_indices"]) == set(baseline[key]["gold_indices"])
        and set(candidate[key]["predicted_indices"]) != set(candidate[key]["gold_indices"])
        for key in baseline
    )
    return {
        "count": count,
        "baseline_correct_denominator": eligible,
        "rate": count / eligible if eligible else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(args.rows)
    b3 = keyed(rows, B3)
    image = keyed(rows, IMAGE_P1)
    v12 = keyed(rows, V12_P1)
    if not (set(b3) == set(image) == set(v12)):
        raise RuntimeError("History policies do not use identical Calibration rows")
    b3_metrics = metrics(b3.values())
    image_metrics = metrics(image.values())
    v12_metrics = metrics(v12.values())
    image_negative = negative_transfer(b3, image)
    v12_negative = negative_transfer(b3, v12)
    micro_gain = float(v12_metrics["option_micro_f1"]) - float(
        image_metrics["option_micro_f1"]
    )
    validity_gain = float(v12_metrics["contract_valid_rate"]) - float(
        image_metrics["contract_valid_rate"]
    )
    advanced = (
        micro_gain > 0
        and validity_gain >= -0.010
        and float(v12_negative["rate"]) <= float(image_negative["rate"])
    )
    summary = {
        "study": "Final QA Calibration history-policy selection",
        "b3_no_history": b3_metrics,
        "image_only_top3_p1": {
            "metrics": image_metrics,
            "negative_transfer_from_b3": image_negative,
            "case_grouped_bootstrap_vs_b3": bootstrap_difference(image, b3),
        },
        "v12_lambdamart_top3_p1": {
            "metrics": v12_metrics,
            "negative_transfer_from_b3": v12_negative,
            "case_grouped_bootstrap_vs_b3": bootstrap_difference(v12, b3),
        },
        "v12_minus_image_only": {
            "option_micro_f1": micro_gain,
            "exact_answer_set_accuracy": float(v12_metrics["exact_answer_set_accuracy"])
            - float(image_metrics["exact_answer_set_accuracy"]),
            "contract_valid_rate": validity_gain,
            "case_grouped_bootstrap": bootstrap_difference(v12, image),
        },
        "prespecified_v12_advancement_rule_passed": advanced,
        "selected_history_policy": "v12_lambdamart_top3" if advanced else "medsiglip_image_only_top3",
        "boundary": "Calibration retrieval-policy selection only; no Validation or Test access.",
    }
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "experiments/final_qa_development/final_qa_qlora_384_calibration_rows.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/final_qa_development/final_qa_history_policy_selection.json",
    )
    print(json.dumps(run(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
