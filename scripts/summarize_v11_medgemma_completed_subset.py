"""Summarize complete rows from an interrupted V11 generation diagnostic.

This intentionally labels the result as a completed-subset diagnostic. It is
not a replacement for a prospectively selected generation study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_v11_medgemma_development import POLICIES, QUESTIONS, read_jsonl, summarize  # noqa: E402
from medical_rag.similar_case.v11_qrel import report_index_spectrum  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=ROOT / "experiments/v11_development/v11_medgemma_generation_48_rows.jsonl")
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--output-rows", type=Path, default=ROOT / "experiments/v11_development/v11_medgemma_generation_completed_rows.jsonl")
    parser.add_argument("--output-summary", type=Path, default=ROOT / "data/splits/v11/v11_medgemma_generation_completed_summary.json")
    args = parser.parse_args()

    rows = read_jsonl(args.rows)
    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    expected_per_case = len(QUESTIONS) * len(POLICIES)
    counts = Counter(str(row["case_id"]) for row in rows)
    complete_ids = sorted(case_id for case_id, count in counts.items() if count == expected_per_case)
    complete_rows = [row for row in rows if str(row["case_id"]) in set(complete_ids)]
    if len(complete_rows) != len(complete_ids) * expected_per_case:
        raise RuntimeError("Completed-case filtering did not produce a complete matrix")
    args.output_rows.parent.mkdir(parents=True, exist_ok=True)
    args.output_rows.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in complete_rows), encoding="utf-8")
    spectrum_counts = Counter(report_index_spectrum(cases[case_id]) for case_id in complete_ids)
    summary = {
        "study": "V11 MedGemma completed-generation diagnostic",
        "status": "development_only_completed_subset_not_confirmation",
        "source_rows_sha256": sha256(args.rows),
        "completed_rows_sha256": sha256(args.output_rows),
        "complete_case_count": len(complete_ids),
        "complete_row_count": len(complete_rows),
        "incomplete_case_ids": sorted(case_id for case_id, count in counts.items() if count != expected_per_case),
        "case_selection": {
            "rule": "retain cases with a complete 12-row matrix from an interrupted resumable development run",
            "report_indexed_spectrum_counts": dict(sorted(spectrum_counts.items())),
            "not_prospectively_selected": True,
        },
        "metrics": summarize(complete_rows),
        "claim_boundary": "This completed-subset diagnostic reports output-contract, evidence-budget and same-source reference metrics only; it is not a clinical accuracy estimate, confirmation result, human review or external validation.",
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
