"""Create the deterministic V12 generation-pilot case manifest.

The pilot needs RadGraph-complete Validation cases because the saved V12
ranking rows use that same executable bank.  This is a data-integrity filter,
not an outcome filter.  The final 48 cases are selected within spectrum
strata by a predeclared SHA-256 order and are written before generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def spectrum(case: dict[str, Any]) -> str:
    value = " ".join(str(case.get("problems", "")).lower().split())
    if value == "normal":
        return "report_indexed_normal"
    if value in {"", "no indexing"}:
        return "report_indexed_indeterminate"
    return "report_indexed_abnormal"


def digest(case_id: str) -> str:
    return hashlib.sha256(f"v12-generation|{case_id}".encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--ranking-rows", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_validation_ranking_rows.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_selection_rows.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_manifest.json")
    parser.add_argument("--per-stratum", type=int, default=24)
    args = parser.parse_args()

    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    split = read_json(args.split)
    validation_ids = {str(case_id) for case_id in split["partitions"]["validation"]["case_ids"]}
    ranking_ids = {str(row["case_id"]) for row in read_jsonl(args.ranking_rows)}
    eligible = sorted(validation_ids & ranking_ids & set(cases))
    strata: dict[str, list[str]] = {}
    for case_id in eligible:
        strata.setdefault(spectrum(cases[case_id]), []).append(case_id)
    chosen: list[str] = []
    quotas = {"report_indexed_normal": args.per_stratum, "report_indexed_abnormal": args.per_stratum}
    for label, quota in quotas.items():
        ordered = sorted(strata.get(label, []), key=lambda case_id: (digest(case_id), case_id))
        if len(ordered) < quota:
            raise RuntimeError(f"not enough eligible {label} cases: {len(ordered)} < {quota}")
        chosen.extend(ordered[:quota])
    chosen = sorted(chosen)
    rows = [
        {"case_id": case_id, "spectrum": spectrum(cases[case_id]), "selection_digest": digest(case_id)}
        for case_id in chosen
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "study": "V12 generation pilot case manifest",
        "status": "selection_frozen_before_generation",
        "selection_rule": "Within report-indexed normal/abnormal strata, sort by SHA256(v12-generation|canonical_case_id) and take the first 24 per stratum.",
        "source_filter": "V10 Validation IDs intersected with case IDs covered by the saved V12 ranking rows; this is a data-integrity filter, not an outcome filter.",
        "no_test_evaluation": True,
        "eligible_validation_case_count": len(eligible),
        "eligible_spectrum_counts": {label: len(strata.get(label, [])) for label in ("report_indexed_normal", "report_indexed_abnormal", "report_indexed_indeterminate")},
        "selected_case_count": len(chosen),
        "selected_spectrum_counts": {label: sum(row["spectrum"] == label for row in rows) for label in quotas},
        "selected_case_ids_sha256": hashlib.sha256("\n".join(chosen).encode("utf-8")).hexdigest(),
        "selection_rows_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "selection_rows": str(args.output.resolve().relative_to(ROOT)),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
