from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.multimodal.v9_generation import select_primary_image  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from run_v10_evidence_generation_development import read_json, read_jsonl  # noqa: E402


DEFAULT_CASES = ROOT / "data/processed/openi_cases.jsonl"
DEFAULT_SPLIT = ROOT / "data/splits/v10/v10_cluster_disjoint_split.json"
DEFAULT_RANKINGS = (
    ROOT / "experiments/v12_optimization/retrieval/v12_validation_ranking_rows.jsonl"
)
DEFAULT_VALIDATION = ROOT / "data/splits/v13/v13_target_concept_validation_summary.json"
DEFAULT_IMAGE_ROOT = ROOT / "data/raw/openi_official_images"
DEFAULT_ROWS = ROOT / "data/splits/v13/v13_concept_qa_manifest.jsonl"
DEFAULT_SUMMARY = ROOT / "data/splits/v13/v13_concept_qa_manifest_summary.json"
SEED = 7143
QUOTA_PER_STRATUM = 48
QUESTION_TYPES = ("findings", "impression")


def spectrum(case: Mapping[str, Any]) -> str:
    value = " ".join(str(case.get("problems") or "").lower().split())
    if value == "normal":
        return "normal"
    if value in {"", "no indexing"}:
        return "indeterminate"
    return "abnormal"


def selection_digest(case_id: object) -> str:
    canonical = str(case_id).strip()
    if not canonical:
        raise ValueError("case_id must be non-empty")
    return hashlib.sha256(
        f"v13-concept-qa|{SEED}|{canonical}".encode("utf-8")
    ).hexdigest()


def case_id_fingerprint(case_ids: Sequence[object]) -> str:
    canonical = sorted({str(case_id).strip() for case_id in case_ids})
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen V13 concept-QA manifest.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    validation = read_json(args.validation)
    if validation.get("status") != "validation_development_complete_test_not_evaluated":
        raise RuntimeError("V13 concept Validation is not complete")
    interval = validation["macro_auprc_difference_vs_prevalence"]
    if float(interval["ci_95_low"]) <= 0:
        raise RuntimeError("V13 concept model did not pass its Validation promotion gate")

    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    split = read_json(args.split)
    validation_ids = {
        str(case_id) for case_id in split["partitions"]["validation"]["case_ids"]
    }
    ranking_rows = read_jsonl(args.rankings)
    by_case: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in ranking_rows:
        case_id = str(row["case_id"])
        question_type = str(row["question_type"])
        if question_type in QUESTION_TYPES:
            by_case[case_id][question_type] = row

    eligible: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for case_id in sorted(validation_ids & set(cases) & set(by_case)):
        if set(by_case[case_id]) != set(QUESTION_TYPES):
            continue
        valid_rankings = all(
            len(by_case[case_id][question]["rankings"]["rrf_lambdamart"][:3]) == 3
            and all(
                str(candidate) in cases
                for candidate in by_case[case_id][question]["rankings"][
                    "rrf_lambdamart"
                ][:3]
            )
            for question in QUESTION_TYPES
        )
        if not valid_rankings:
            continue
        image_path = select_primary_image(cases[case_id], args.image_root)
        if not image_path.is_file():
            continue
        eligible[spectrum(cases[case_id])].append((case_id, image_path))

    selected: list[tuple[str, str, Path]] = []
    for label in ("normal", "abnormal"):
        ordered = sorted(
            eligible[label], key=lambda value: (selection_digest(value[0]), value[0])
        )
        if len(ordered) < QUOTA_PER_STRATUM:
            raise RuntimeError(f"Insufficient eligible {label} cases")
        selected.extend((case_id, label, path) for case_id, path in ordered[:QUOTA_PER_STRATUM])
    selected.sort(key=lambda value: value[0])

    rows = [
        {
            "case_id": case_id,
            "spectrum": label,
            "selection_digest": selection_digest(case_id),
            "target_image_path": str(path),
        }
        for case_id, label, path in selected
    ]
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    selected_ids = [row["case_id"] for row in rows]
    summary = {
        "study": "V13 concept-on/off QA manifest",
        "status": "selection_frozen_before_generation",
        "seed": SEED,
        "selection_domain": "v13-concept-qa",
        "eligibility": (
            "V10 Validation cases with both formal V12 ranking rows, three available "
            "rrf_lambdamart historical cases per question, and a readable primary image"
        ),
        "eligible_counts": {key: len(value) for key, value in sorted(eligible.items())},
        "selected_counts": {
            "normal": sum(row["spectrum"] == "normal" for row in rows),
            "abnormal": sum(row["spectrum"] == "abnormal" for row in rows),
            "total": len(rows),
        },
        "selected_case_ids_sha256": case_id_fingerprint(selected_ids),
        "artifacts": {
            "cases_sha256": file_sha256(args.cases),
            "split_sha256": file_sha256(args.split),
            "ranking_rows_sha256": file_sha256(args.rankings),
            "v13_validation_sha256": file_sha256(args.validation),
            "manifest_rows_sha256": file_sha256(args.rows_output),
            "script_sha256": file_sha256(Path(__file__)),
        },
        "test_outcomes_inspected": False,
        "claim_boundary": "Validation-only automated QA pilot; no V10 Test evaluation.",
    }
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

