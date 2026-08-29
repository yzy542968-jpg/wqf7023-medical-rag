from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.qa.medgemma_contract import (  # noqa: E402
    parse_option_indices_with_wrapper_repair,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, float | int]:
    tp = fp = fn = 0
    exact: list[float] = []
    valid: list[float] = []
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        gold = set(row["gold_indices"])
        predicted = set(row[prediction_key])
        value = float(gold == predicted)
        exact.append(value)
        by_type[row["answer_type"]].append(value)
        valid.append(float(row[f"{prediction_key}_valid"]))
        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    denominator = 2 * tp + fp + fn
    return {
        "row_count": len(rows),
        "exact_answer_set_accuracy": sum(exact) / len(exact),
        "option_micro_f1": 2 * tp / denominator if denominator else 0.0,
        "contract_valid_rate": sum(valid) / len(valid),
        "single_choice_accuracy": sum(by_type["single_choice"]) / len(by_type["single_choice"]),
        "multi_choice_exact_accuracy": sum(by_type["multi_choice"]) / len(by_type["multi_choice"]),
        "fixed_choice_accuracy": sum(by_type["fixed_choice"]) / len(by_type["fixed_choice"]),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = _read_jsonl(args.rows)
    categories: Counter[str] = Counter()
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        repaired = parse_option_indices_with_wrapper_repair(
            row["raw_output"],
            option_count=int(row["option_count"]),
            answer_type=row["answer_type"],
        )
        if repaired["contract_valid"]:
            category = "+".join(repaired["repairs"]) or "strict_no_repair"
        elif "[" not in repaired["normalized_output"] or "]" not in repaired["normalized_output"]:
            category = "incomplete_or_missing_array"
        else:
            category = "remaining_invalid_payload"
        categories[category] += 1
        audit_rows.append(
            {
                **row,
                "strict_prediction": row["predicted_indices"],
                "strict_prediction_valid": bool(row["contract_valid"]),
                "repaired_prediction": repaired["indices"],
                "repaired_prediction_valid": bool(repaired["contract_valid"]),
            }
        )

    conditions: dict[str, Any] = {}
    for condition in sorted({row["condition"] for row in audit_rows}):
        selected = [row for row in audit_rows if row["condition"] == condition]
        conditions[condition] = {
            "strict": _metrics(selected, "strict_prediction"),
            "repaired": _metrics(selected, "repaired_prediction"),
        }
    result = {
        "study": "Final QA post-hoc MedGemma wrapper-repair audit",
        "status": "calibration_only_post_hoc_parser_development",
        "source_rows": str(args.rows.relative_to(ROOT)),
        "row_count": len(rows),
        "repair_categories": dict(sorted(categories.items())),
        "conditions": conditions,
        "strict_results_unchanged": True,
        "boundary": (
            "Wrapper repair was designed after Calibration formatting inspection. It is "
            "not outcome-blind validation and cannot overwrite the strict r1 result."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/medgemma_contract_pilot_rows.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/medgemma_parser_audit_summary.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
