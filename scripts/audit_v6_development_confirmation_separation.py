from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_V5_CONFIG = ROOT / "config" / "multimodal_v5.json"
DEFAULT_V5_COHORT = ROOT / "data" / "processed" / "openi_multimodal_v5_cohort.json"
DEFAULT_DEVELOPMENT_MANIFEST = (
    ROOT / "data" / "splits" / "v6" / "v6_development_case_ids.txt"
)
DEFAULT_AUDIT_OUTPUT = (
    ROOT
    / "data"
    / "splits"
    / "v6"
    / "v6_development_confirmation_overlap_audit.json"
)

KNOWN_INDETERMINATE_PROBLEM_VALUES = {"no indexing"}
UNEXPECTED_ADMINISTRATIVE_PROBLEM_VALUES = {
    "",
    "missing",
    "n/a",
    "na",
    "no index",
    "none",
    "not available",
    "not indexed",
    "not reported",
    "not specified",
    "null",
    "other",
    "unknown",
    "unspecified",
}
UNEXPECTED_ADMINISTRATIVE_PROBLEM_PATTERNS = (
    re.compile(r"^(?:no|not)\s+index(?:ed|ing)?$"),
    re.compile(r"^(?:data|information)\s+not\s+available$"),
)


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def placeholder_ratio(value: Any) -> float:
    tokens = clean_text(value).split()
    if not tokens:
        return 1.0
    return sum("XXXX" in token.upper() for token in tokens) / len(tokens)


def canonical_case_id(value: Any) -> str:
    case_id = str(value).strip()
    if not case_id:
        raise ValueError("Case IDs must be non-empty after str(value).strip().")
    return case_id


def canonical_ids(values: Iterable[Any]) -> list[str]:
    ids = [canonical_case_id(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError("Case-ID collections must not contain duplicates.")
    return sorted(ids)


def case_id_payload(values: Iterable[Any]) -> str:
    return "\n".join(canonical_ids(values))


def case_id_fingerprint(values: Iterable[Any]) -> str:
    return hashlib.sha256(case_id_payload(values).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def case_ids_from_payload(payload: dict[str, Any]) -> set[str]:
    if "questions" in payload:
        return {canonical_case_id(row["case_id"]) for row in payload["questions"]}
    if "case_ids" in payload:
        return {canonical_case_id(value) for value in payload["case_ids"]}
    if "cases" in payload:
        return {canonical_case_id(row["case_id"]) for row in payload["cases"]}
    raise ValueError("Cannot find case IDs in source manifest.")


def is_v6_eligible_case(case: dict[str, Any]) -> bool:
    """Apply the broadened V6 eligibility rule before problems-field stratification."""
    return (
        bool(case.get("images"))
        and len(clean_text(case.get("findings", ""))) >= 40
        and len(clean_text(case.get("impression", ""))) >= 8
        and placeholder_ratio(case.get("indication", "")) <= 0.5
    )


def report_index_class(case: dict[str, Any]) -> str:
    value = clean_text(case.get("problems", "")).lower()
    if value == "normal":
        return "report_indexed_normal"
    if value in KNOWN_INDETERMINATE_PROBLEM_VALUES:
        return "report_index_indeterminate"
    if value in UNEXPECTED_ADMINISTRATIVE_PROBLEM_VALUES or any(
        pattern.fullmatch(value)
        for pattern in UNEXPECTED_ADMINISTRATIVE_PROBLEM_PATTERNS
    ):
        raise ValueError(
            f"Eligible case {canonical_case_id(case['case_id'])} has an unclassified "
            f"administrative problems value: {value!r}."
        )
    return "report_indexed_abnormal"


def build_audit(
    *,
    all_cases: list[dict[str, Any]],
    prior_excluded_ids: set[str],
    v5_case_ids: set[str],
    development_ids: set[str],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    source_ids = [canonical_case_id(case["case_id"]) for case in all_cases]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Source cases contain duplicate case IDs.")

    source_by_id = {
        canonical_case_id(case["case_id"]): case
        for case in all_cases
    }
    missing_development = sorted(development_ids - set(source_by_id))
    if missing_development:
        raise ValueError(f"Development IDs missing from source: {missing_development[:5]}")

    confirmation_excluded = set(prior_excluded_ids) | set(v5_case_ids)
    eligible_cases = [
        case
        for case_id, case in source_by_id.items()
        if case_id not in confirmation_excluded and is_v6_eligible_case(case)
    ]
    normalized_problem_counts = Counter(
        clean_text(case.get("problems", "")).lower()
        for case in eligible_cases
    )

    strata: dict[str, list[str]] = {
        "report_indexed_normal": [],
        "report_indexed_abnormal": [],
        "report_index_indeterminate": [],
    }
    for case in eligible_cases:
        strata[report_index_class(case)].append(canonical_case_id(case["case_id"]))
    strata = {name: canonical_ids(ids) for name, ids in strata.items()}

    eligible_ids = canonical_ids(
        case_id for ids in strata.values() for case_id in ids
    )
    stratifiable_ids = canonical_ids(
        strata["report_indexed_normal"] + strata["report_indexed_abnormal"]
    )
    development_manifest = canonical_ids(development_ids)
    development_eligible_overlap = canonical_ids(
        set(development_manifest) & set(eligible_ids)
    )
    development_stratifiable_overlap = canonical_ids(
        set(development_manifest) & set(stratifiable_ids)
    )

    development_spectrum = {
        "report_indexed_normal": 0,
        "report_indexed_abnormal": 0,
        "report_index_indeterminate": 0,
    }
    for case_id in development_manifest:
        development_spectrum[report_index_class(source_by_id[case_id])] += 1

    audit = {
        "audit": "V6 development-confirmation case-ID separation",
        "version": "1.0",
        "status": "selection_frame_audit_only_no_confirmation_case_ids_instantiated",
        "canonicalization": {
            "case_id": "str(value).strip()",
            "ordering": "ascending Unicode code-point order of unique canonical case IDs",
            "serialization": "newline joined UTF-8 with no trailing newline",
            "fingerprint": "SHA-256 lowercase hexadecimal digest",
        },
        "development_source": {
            "definition": "development split of the frozen V5 multimodal cohort",
            "case_count": len(development_manifest),
            "case_ids_sha256": case_id_fingerprint(development_manifest),
            "report_index_spectrum": development_spectrum,
        },
        "confirmation_selection_frame": {
            "source_case_count": len(source_ids),
            "source_case_ids_sha256": case_id_fingerprint(source_ids),
            "prior_excluded_case_count": len(prior_excluded_ids),
            "prior_excluded_case_ids_sha256": case_id_fingerprint(prior_excluded_ids),
            "v5_excluded_case_count": len(v5_case_ids),
            "v5_excluded_case_ids_sha256": case_id_fingerprint(v5_case_ids),
            "all_excluded_case_count": len(confirmation_excluded),
            "all_excluded_case_ids_sha256": case_id_fingerprint(confirmation_excluded),
            "v6_eligible_case_count": len(eligible_ids),
            "v6_eligible_case_ids_sha256": case_id_fingerprint(eligible_ids),
            "v6_stratifiable_case_count": len(stratifiable_ids),
            "v6_stratifiable_case_ids_sha256": case_id_fingerprint(stratifiable_ids),
            "report_indexed_normal_case_count": len(strata["report_indexed_normal"]),
            "report_indexed_normal_case_ids_sha256": case_id_fingerprint(
                strata["report_indexed_normal"]
            ),
            "report_indexed_abnormal_case_count": len(strata["report_indexed_abnormal"]),
            "report_indexed_abnormal_case_ids_sha256": case_id_fingerprint(
                strata["report_indexed_abnormal"]
            ),
            "report_index_indeterminate_case_count": len(
                strata["report_index_indeterminate"]
            ),
            "report_index_indeterminate_case_ids_sha256": case_id_fingerprint(
                strata["report_index_indeterminate"]
            ),
            "problems_field_audit": {
                "normalized_unique_value_count": len(normalized_problem_counts),
                "known_indeterminate_values": sorted(
                    KNOWN_INDETERMINATE_PROBLEM_VALUES
                ),
                "known_indeterminate_value_counts": {
                    value: normalized_problem_counts[value]
                    for value in sorted(KNOWN_INDETERMINATE_PROBLEM_VALUES)
                },
                "unexpected_administrative_values": [],
            },
        },
        "separation_check": {
            "development_confirmation_eligible_overlap_count": len(
                development_eligible_overlap
            ),
            "development_confirmation_eligible_overlap_case_ids": (
                development_eligible_overlap
            ),
            "development_confirmation_stratifiable_overlap_count": len(
                development_stratifiable_overlap
            ),
            "development_confirmation_stratifiable_overlap_case_ids": (
                development_stratifiable_overlap
            ),
            "case_id_disjointness_verified": not development_eligible_overlap,
            "patient_level_independence_verified": False,
            "patient_level_boundary": (
                "Reliable patient or subject identifiers are unavailable in the processed "
                "source data; this audit establishes case-ID disjointness only."
            ),
        },
        "predefined_confirmation_composition": {
            "selected_case_count": 240,
            "report_indexed_normal": 172,
            "report_indexed_abnormal": 68,
            "target": {
                "case_count": 120,
                "report_indexed_normal": 86,
                "report_indexed_abnormal": 34,
            },
            "distractor": {
                "case_count": 120,
                "report_indexed_normal": 86,
                "report_indexed_abnormal": 34,
            },
            "instantiated_case_ids_present": False,
        },
    }
    collections = {
        "development": development_manifest,
        "eligible": eligible_ids,
        "stratifiable": stratifiable_ids,
        **strata,
    }
    return audit, collections


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit V6 development-confirmation case-ID separation without selecting "
            "or instantiating the final confirmation cohort."
        )
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--v5-config", type=Path, default=DEFAULT_V5_CONFIG)
    parser.add_argument("--v5-cohort", type=Path, default=DEFAULT_V5_COHORT)
    parser.add_argument(
        "--development-output", type=Path, default=DEFAULT_DEVELOPMENT_MANIFEST
    )
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    args = parser.parse_args()

    v5_config = read_json(args.v5_config)
    v5_cohort = read_json(args.v5_cohort)
    all_cases = read_jsonl(args.cases)

    prior_excluded_ids: set[str] = set()
    prior_manifest_hashes: dict[str, str] = {}
    for relative in v5_config["cohort"]["excluded_source_manifests"]:
        path = ROOT / relative
        prior_excluded_ids.update(case_ids_from_payload(read_json(path)))
        prior_manifest_hashes[portable_path(path)] = file_sha256(path)

    v5_case_ids = {canonical_case_id(value) for value in v5_cohort["case_ids"]}
    development_ids = {
        canonical_case_id(value)
        for value in v5_cohort["split"]["development"]["case_ids"]
    }
    audit, collections = build_audit(
        all_cases=all_cases,
        prior_excluded_ids=prior_excluded_ids,
        v5_case_ids=v5_case_ids,
        development_ids=development_ids,
    )
    audit["inputs"] = {
        "source_cases": portable_path(args.cases),
        "source_jsonl_sha256": file_sha256(args.cases),
        "v5_config": portable_path(args.v5_config),
        "v5_config_sha256": file_sha256(args.v5_config),
        "v5_cohort": portable_path(args.v5_cohort),
        "v5_cohort_sha256": file_sha256(args.v5_cohort),
        "prior_source_manifest_sha256": prior_manifest_hashes,
        "audit_script": portable_path(Path(__file__)),
        "audit_script_sha256": file_sha256(Path(__file__)),
    }

    expected_counts = {
        "development": 120,
        "eligible": 1479,
        "stratifiable": 1462,
        "report_indexed_normal": 1045,
        "report_indexed_abnormal": 417,
        "report_index_indeterminate": 17,
    }
    observed_counts = {name: len(collections[name]) for name in expected_counts}
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"V6 selection-frame counts changed: expected {expected_counts}, "
            f"observed {observed_counts}."
        )
    problems_audit = audit["confirmation_selection_frame"]["problems_field_audit"]
    if problems_audit["normalized_unique_value_count"] != 322:
        raise RuntimeError(
            "The V6 eligible-frame normalized problems vocabulary changed: "
            f"expected 322 unique values, observed "
            f"{problems_audit['normalized_unique_value_count']}."
        )
    separation = audit["separation_check"]
    if separation["development_confirmation_eligible_overlap_count"] != 0:
        raise RuntimeError("V6 development overlaps the confirmation-eligible frame.")
    if separation["development_confirmation_stratifiable_overlap_count"] != 0:
        raise RuntimeError("V6 development overlaps the confirmation-stratifiable frame.")

    args.development_output.parent.mkdir(parents=True, exist_ok=True)
    args.development_output.write_text(
        case_id_payload(collections["development"]), encoding="utf-8", newline=""
    )
    audit["outputs"] = {
        "development_manifest": portable_path(args.development_output),
        "development_manifest_sha256": file_sha256(args.development_output),
        "audit_output": portable_path(args.audit_output),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
