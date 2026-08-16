from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POSITIVE = {"1", "true", "yes", "y", "supported"}
NEGATIVE = {"0", "false", "no", "n", "unsupported"}


def parse_label(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in POSITIVE:
        return True
    if normalized in NEGATIVE:
        return False
    return None


def safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate(rows: list[dict], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        human = parse_label(row.get("human_supported", ""))
        if human is None:
            continue
        predicted = (
            float(row["support_score"]) >= threshold
            and str(row["negation_consistent"]).strip().lower() == "true"
        )
        if predicted and human:
            tp += 1
        elif predicted and not human:
            fp += 1
        elif not predicted and human:
            fn += 1
        else:
            tn += 1

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    return {
        "threshold": threshold,
        "n": tp + fp + tn + fn,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": safe_divide(2 * precision * recall, precision + recall),
        "accuracy": safe_divide(tp + tn, tp + fp + tn + fn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score evidence thresholds against manual labels.")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "experiments" / "final_p2" / "evidence_calibration_50.csv",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.40, 0.50, 0.65])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "final_p2" / "evidence_calibration_metrics.json",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labelled = sum(parse_label(row.get("human_supported", "")) is not None for row in rows)
    if labelled == 0:
        raise SystemExit(
            f"No manual labels found in {args.input}. Fill human_supported with 1 or 0 first."
        )

    overall = [evaluate(rows, threshold) for threshold in args.thresholds]
    systems = sorted({row["system"] for row in rows})
    by_system = {
        system: [
            evaluate([row for row in rows if row["system"] == system], threshold)
            for threshold in args.thresholds
        ]
        for system in systems
    }
    best = max(overall, key=lambda result: (result["f1"], result["recall"], -result["threshold"]))
    result = {
        "input": str(args.input),
        "labelled": labelled,
        "unlabelled": len(rows) - labelled,
        "recommended_threshold": best["threshold"],
        "overall": overall,
        "by_system": by_system,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
