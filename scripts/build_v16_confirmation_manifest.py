"""Build the all-eligible V16 confirmation manifest from frozen Test rankings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_ids(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(values))).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=568)
    args = parser.parse_args()

    rows = read_jsonl(args.ranking_rows)
    by_case: dict[str, set[str]] = {}
    spectrum: dict[str, str] = {}
    for row in rows:
        case_id = str(row["case_id"]).strip()
        question_type = str(row["question_type"]).strip()
        by_case.setdefault(case_id, set()).add(question_type)
        spectrum[case_id] = str(row.get("spectrum", "indeterminate"))
        ranking = row.get("rankings", {}).get("rrf_lambdamart", [])
        if len(ranking) < 3 or len(set(ranking[:3])) != 3:
            raise RuntimeError(f"Incomplete Top-3 ranking for {case_id}:{question_type}")

    expected_types = {"findings", "impression"}
    invalid = {case_id: sorted(types) for case_id, types in by_case.items() if types != expected_types}
    if invalid:
        raise RuntimeError(f"Incomplete question matrix: {invalid}")
    if len(by_case) != args.expected_cases:
        raise RuntimeError(f"Confirmation frame has {len(by_case)} cases; expected {args.expected_cases}")

    manifest_rows = []
    for case_id in sorted(by_case):
        digest = hashlib.sha256(f"v16-confirmation|{case_id}".encode("utf-8")).hexdigest()
        manifest_rows.append({
            "case_id": case_id,
            "selection_digest": digest,
            "spectrum": spectrum[case_id],
            "selection_policy": "all_technically_eligible_v10_test_cases",
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    summary = {
        "study": "V16 generation adaptation confirmation",
        "status": "confirmation_manifest_instantiated_after_protocol_freeze",
        "case_count": len(manifest_rows),
        "case_ids_sha256": sha256_ids([row["case_id"] for row in manifest_rows]),
        "ranking_rows_sha256": file_sha256(args.ranking_rows),
        "manifest_sha256": file_sha256(args.output),
        "spectrum_counts": {
            name: sum(row["spectrum"] == name for row in manifest_rows)
            for name in sorted({row["spectrum"] for row in manifest_rows})
        },
        "replacement_policy": "no replacement after instantiation",
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
