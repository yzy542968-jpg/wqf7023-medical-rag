from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = {
    "report_bm25": ROOT / "experiments" / "final_p2" / "report_bm25_checked_t0.50.jsonl",
    "case_bm25_top1": ROOT / "experiments" / "final_p2" / "case_bm25_top1_checked_t0.50.jsonl",
    "case_hybrid_top1": ROOT / "experiments" / "final_p2" / "case_hybrid_top1_checked_t0.50.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def classify_stratum(score: float, negation_consistent: bool) -> str:
    if not negation_consistent:
        return "negation_conflict"
    if 0.40 <= score < 0.65:
        return "threshold_disagreement"
    if score >= 0.65:
        return "high_support"
    return "low_support"


def expand_sentence_checks() -> list[dict]:
    expanded: list[dict] = []
    for system, path in DEFAULT_INPUTS.items():
        for row in read_jsonl(path):
            for sentence_index, check in enumerate(row.get("sentence_checks", []), start=1):
                score = float(check.get("support_score", 0.0))
                negation_consistent = bool(check.get("negation_consistent", True))
                expanded.append(
                    {
                        "system": system,
                        "qid": row.get("qid", ""),
                        "case_id": row.get("case_id", ""),
                        "question_type": row.get("question_type", ""),
                        "question": row.get("question", ""),
                        "reference_answer": row.get("reference_answer", ""),
                        "evidence_case_ids": "|".join(row.get("evidence_case_ids", [])),
                        "draft_answer": row.get("draft_answer", ""),
                        "sentence_index": sentence_index,
                        "answer_sentence": check.get("sentence", ""),
                        "matched_evidence": check.get("matched_evidence", ""),
                        "support_score": score,
                        "negation_consistent": negation_consistent,
                        "stratum": classify_stratum(score, negation_consistent),
                    }
                )
    return expanded


def round_robin_sample(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    by_system: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)
    for values in by_system.values():
        rng.shuffle(values)

    selected: list[dict] = []
    systems = sorted(by_system)
    while len(selected) < count and any(by_system.values()):
        for system in systems:
            if by_system[system] and len(selected) < count:
                selected.append(by_system[system].pop())
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a stratified sentence-level manual calibration sample."
    )
    parser.add_argument("--size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "final_p2" / "evidence_calibration_50.csv",
    )
    args = parser.parse_args()

    rows = expand_sentence_checks()
    rng = random.Random(args.seed)
    desired = {
        "negation_conflict": 12,
        "threshold_disagreement": 18,
        "high_support": 10,
        "low_support": 10,
    }
    scale = args.size / sum(desired.values())
    targets = {name: round(count * scale) for name, count in desired.items()}
    targets["threshold_disagreement"] += args.size - sum(targets.values())

    selected: list[dict] = []
    selected_keys: set[tuple[str, str, int]] = set()
    for stratum, target in targets.items():
        pool = [row for row in rows if row["stratum"] == stratum]
        for row in round_robin_sample(pool, target, rng):
            key = (row["system"], row["qid"], row["sentence_index"])
            if key not in selected_keys:
                selected.append(row)
                selected_keys.add(key)

    if len(selected) < args.size:
        remainder = [
            row
            for row in rows
            if (row["system"], row["qid"], row["sentence_index"]) not in selected_keys
        ]
        for row in round_robin_sample(remainder, args.size - len(selected), rng):
            selected.append(row)

    selected.sort(key=lambda row: (row["stratum"], row["system"], row["qid"]))
    for sample_id, row in enumerate(selected, start=1):
        row["sample_id"] = f"CAL{sample_id:03d}"
        row["auto_supported_t040"] = int(
            row["support_score"] >= 0.40 and row["negation_consistent"]
        )
        row["auto_supported_t050"] = int(
            row["support_score"] >= 0.50 and row["negation_consistent"]
        )
        row["auto_supported_t065"] = int(
            row["support_score"] >= 0.65 and row["negation_consistent"]
        )
        row["human_supported"] = ""
        row["human_negation_correct"] = ""
        row["notes"] = ""

    fieldnames = [
        "sample_id",
        "stratum",
        "system",
        "qid",
        "case_id",
        "question_type",
        "question",
        "reference_answer",
        "evidence_case_ids",
        "draft_answer",
        "sentence_index",
        "answer_sentence",
        "matched_evidence",
        "support_score",
        "negation_consistent",
        "auto_supported_t040",
        "auto_supported_t050",
        "auto_supported_t065",
        "human_supported",
        "human_negation_correct",
        "notes",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    stratum_counts: dict[str, int] = defaultdict(int)
    system_counts: dict[str, int] = defaultdict(int)
    for row in selected:
        stratum_counts[row["stratum"]] += 1
        system_counts[row["system"]] += 1
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": len(selected),
                "strata": dict(sorted(stratum_counts.items())),
                "systems": dict(sorted(system_counts.items())),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
