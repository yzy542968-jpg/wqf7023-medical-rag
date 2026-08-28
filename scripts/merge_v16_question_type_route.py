#!/usr/bin/env python3
"""Build an auditable question-type-routed V16 row file.

The route is deliberately explicit: rows for selected question types come
from the specialist file and all remaining rows come from the fallback file.
The script validates that both files describe the same case/condition matrix
before writing the routed JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


KEY_FIELDS = ("case_id", "question_type", "condition")


def load_rows(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [field for field in KEY_FIELDS if field not in row]
            if missing:
                raise ValueError(f"{path}:{line_number}: missing key fields {missing}")
            key = tuple(str(row[field]) for field in KEY_FIELDS)
            if key in rows:
                raise ValueError(f"{path}:{line_number}: duplicate row key {key}")
            rows[key] = row
    return rows


def build_routed_rows(
    fallback: dict[tuple[str, str, str], dict[str, Any]],
    specialist: dict[tuple[str, str, str], dict[str, Any]],
    *,
    specialist_question_type: str,
    specialist_condition: str | None = None,
) -> tuple[list[dict[str, Any]], int]:
    if set(fallback) != set(specialist):
        only_fallback = sorted(set(fallback) - set(specialist))
        only_specialist = sorted(set(specialist) - set(fallback))
        raise ValueError(
            "Fallback and specialist matrices differ: "
            f"only_fallback={only_fallback[:5]}, only_specialist={only_specialist[:5]}"
        )

    routed: list[dict[str, Any]] = []
    specialist_count = 0
    for key in sorted(fallback):
        question_type = key[1]
        use_specialist = question_type == specialist_question_type
        if specialist_condition is not None:
            use_specialist = use_specialist and key[2] == specialist_condition
        source = specialist if use_specialist else fallback
        row = dict(source[key])
        row["question_type_route"] = (
            f"specialist:{specialist_question_type}" if use_specialist else "fallback"
        )
        routed.append(row)
        specialist_count += use_specialist
    return routed, specialist_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fallback-rows", type=Path, required=True)
    parser.add_argument("--specialist-rows", type=Path, required=True)
    parser.add_argument("--specialist-question-type", required=True)
    parser.add_argument(
        "--specialist-condition",
        help="Optional condition gate; use the specialist only for this condition.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fallback = load_rows(args.fallback_rows)
    specialist = load_rows(args.specialist_rows)
    routed, specialist_count = build_routed_rows(
        fallback,
        specialist,
        specialist_question_type=args.specialist_question_type,
        specialist_condition=args.specialist_condition,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in routed:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "status": "validation_exploratory_route_created",
        "fallback_rows": str(args.fallback_rows),
        "specialist_rows": str(args.specialist_rows),
        "specialist_question_type": args.specialist_question_type,
        "specialist_condition": args.specialist_condition,
        "row_count": len(routed),
        "specialist_row_count": specialist_count,
        "fallback_row_count": len(routed) - specialist_count,
        "claim_boundary": "Exploratory validation routing; no test or clinical claim.",
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
