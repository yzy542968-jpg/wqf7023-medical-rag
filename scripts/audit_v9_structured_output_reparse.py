from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import file_sha256, read_json  # noqa: E402
from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.multimodal.v6_generation import normalized_text  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v9_supplemental_validity.json"
DEFAULT_ROWS = ROOT / "experiments" / "post_submission_v9" / "v9_qa_raw_rows.jsonl"
DEFAULT_AUDIT = ROOT / "experiments" / "post_submission_v9" / "v9_structured_reparse_rows.csv"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v9" / "v9_structured_reparse_summary.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, character in enumerate(text):
        if start is None:
            if character == "{":
                start = index
                depth = 1
                in_string = False
                escaped = False
            continue
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                objects.append(text[start : index + 1])
                start = None
    return objects


def remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


def validate_payload(
    parsed: Any, *, allowed_case_ids: Sequence[str]
) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    answer = normalized_text(parsed.get("answer", ""))
    uncertainty = normalized_text(parsed.get("uncertainty", "")).lower()
    if not answer or uncertainty not in {"low", "medium", "high"}:
        return None
    allowed = set(map(str, allowed_case_ids))
    ids = parsed.get("supporting_case_ids", [])
    if not isinstance(ids, list):
        ids = []
    findings = parsed.get("target_image_findings", [])
    if not isinstance(findings, list):
        findings = []
    return {
        "answer": answer,
        "target_image_findings": [
            normalized_text(value) for value in findings if normalized_text(value)
        ],
        "supporting_case_ids": [
            str(value) for value in ids if str(value) in allowed
        ],
        "historical_support": normalized_text(parsed.get("historical_support", "")),
        "uncertainty": uncertainty,
        "abstain": bool(parsed.get("abstain", False)),
    }


def reparse_v9_output(
    text: str, *, allowed_case_ids: Sequence[str]
) -> tuple[dict[str, Any] | None, str]:
    cleaned = str(text or "").replace("```json", "").replace("```JSON", "")
    cleaned = cleaned.replace("```", "")
    for candidate in balanced_json_objects(cleaned):
        attempts = ((candidate, "balanced_json"), (remove_trailing_commas(candidate), "trailing_comma"))
        seen: set[str] = set()
        for value, repair in attempts:
            if value in seen:
                continue
            seen.add(value)
            try:
                parsed = json.loads(value, strict=False)
            except json.JSONDecodeError:
                continue
            validated = validate_payload(parsed, allowed_case_ids=allowed_case_ids)
            if validated is not None:
                return validated, repair
    return None, "unrecoverable"


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    original_valid = sum(bool(row["original_valid"]) for row in rows)
    recovered_valid = sum(bool(row["reparsed_valid"]) for row in rows)
    newly_recovered = sum(bool(row["newly_recovered"]) for row in rows)
    changed = sum(bool(row["answer_changed"]) for row in rows)
    newly_recovered_rows = [row for row in rows if bool(row["newly_recovered"])]
    return {
        "row_count": count,
        "original_valid_count": original_valid,
        "original_valid_rate": original_valid / count if count else 0.0,
        "reparsed_valid_count": recovered_valid,
        "reparsed_valid_rate": recovered_valid / count if count else 0.0,
        "newly_recovered_count": newly_recovered,
        "newly_recovered_rate": newly_recovered / count if count else 0.0,
        "unrecoverable_count": count - recovered_valid,
        "unrecoverable_rate": (count - recovered_valid) / count if count else 0.0,
        "answer_change_count": changed,
        "answer_change_rate": changed / count if count else 0.0,
        "original_token_f1": statistics.fmean(
            float(row["original_token_f1"]) for row in rows
        ),
        "reparsed_or_original_token_f1": statistics.fmean(
            float(row["reparsed_or_original_token_f1"]) for row in rows
        ),
        "newly_recovered_token_f1": (
            statistics.fmean(float(row["reparsed_token_f1"]) for row in newly_recovered_rows)
            if newly_recovered_rows
            else None
        ),
        "repair_counts": {
            repair: sum(str(row["repair"]) == repair for row in rows)
            for repair in sorted({str(row["repair"]) for row in rows})
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reparse frozen V9 generations without regenerating answers."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    if config["structured_output_reparse"]["generation_rerun_allowed"]:
        raise RuntimeError("Supplemental protocol unexpectedly permits generation reruns.")
    source_rows = read_jsonl(args.rows)
    audit_rows: list[dict[str, Any]] = []
    for row in source_rows:
        reparsed, repair = reparse_v9_output(
            str(row.get("raw_output") or ""),
            allowed_case_ids=row.get("retrieved_case_ids") or [],
        )
        original_answer = normalized_text(row.get("answer") or "")
        reparsed_answer = "" if reparsed is None else str(reparsed["answer"])
        final_answer = reparsed_answer or original_answer
        reference = str(row.get("reference_answer") or "")
        original_valid = bool(row.get("structured_output_valid"))
        reparsed_valid = reparsed is not None
        audit_rows.append(
            {
                "system": str(row["system"]),
                "case_id": str(row["case_id"]),
                "qid": str(row["qid"]),
                "question_type": str(row["question_type"]),
                "original_valid": original_valid,
                "reparsed_valid": reparsed_valid,
                "newly_recovered": reparsed_valid and not original_valid,
                "repair": repair,
                "answer_changed": bool(reparsed_answer and reparsed_answer != original_answer),
                "original_token_f1": float(row["token_f1"]),
                "reparsed_token_f1": (
                    token_f1(reparsed_answer, reference) if reparsed_answer else 0.0
                ),
                "reparsed_or_original_token_f1": token_f1(final_answer, reference),
            }
        )

    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    with args.audit_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        grouped[str(row["system"])].append(row)
    output = {
        "study": "V9 structured-output reparse audit",
        "status": "post_hoc_engineering_sensitivity_complete",
        "source_rows_sha256": file_sha256(args.rows),
        "config_sha256": file_sha256(args.config),
        "overall": summarize(audit_rows),
        "systems": {system: summarize(rows) for system, rows in sorted(grouped.items())},
        "audit_output": {
            "path": args.audit_output.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(args.audit_output),
            "committed_to_public_repository": False,
        },
        "claim_boundary": (
            "The audit repairs parseable formatting only, does not fabricate "
            "truncated objects, and does not replace frozen primary QA metrics."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
