from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import (  # noqa: E402
    canonical_case_id,
    case_id_fingerprint,
    case_ids_from_payload,
    clean_text,
    file_sha256,
    is_v6_eligible_case,
    read_json,
    read_jsonl,
    report_index_class,
)


DEFAULT_PROTOCOL_COMMIT = "afd7ef7"
DEFAULT_CONFIG = ROOT / "config" / "v9_full_source_split_protocol.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_V5_CONFIG = ROOT / "config" / "multimodal_v5.json"
DEFAULT_V5_COHORT = ROOT / "data" / "processed" / "openi_multimodal_v5_cohort.json"
DEFAULT_V6_DEVELOPMENT = ROOT / "data" / "splits" / "v6" / "v6_development_case_ids.txt"
DEFAULT_V6_CONFIRMATION = ROOT / "data" / "splits" / "v6" / "v6_confirmation_cohort.json"
DEFAULT_V7_DEVELOPMENT = ROOT / "data" / "splits" / "v7" / "v7_development_manifest.json"
DEFAULT_V7_CONFIRMATION = ROOT / "data" / "splits" / "v7" / "v7_confirmation_cohort.json"
DEFAULT_MANIFEST = ROOT / "data" / "splits" / "v9" / "v9_full_source_split.json"
DEFAULT_FREEZE = ROOT / "data" / "splits" / "v9" / "v9_full_source_split_freeze.json"


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_id_file(path: Path) -> set[str]:
    return {
        canonical_case_id(value)
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    }


def hash_order_key(domain: str, seed: str, case_id: str) -> tuple[str, str]:
    canonical = canonical_case_id(case_id)
    digest = hashlib.sha256(f"{domain}|{seed}|{canonical}".encode("utf-8")).hexdigest()
    return digest, canonical


def committed_json(commit: str, path: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout.decode("utf-8"))


def historical_untouched_frame(
    *,
    cases_by_id: dict[str, dict[str, Any]],
    v5_config: dict[str, Any],
    v5_cohort: dict[str, Any],
    v6_development_ids: set[str],
    v6_confirmation_ids: set[str],
    v7_development_ids: set[str],
    v7_confirmation_ids: set[str],
) -> set[str]:
    legacy_ids: set[str] = set()
    for relative in v5_config["cohort"]["excluded_source_manifests"]:
        legacy_ids.update(
            canonical_case_id(value)
            for value in case_ids_from_payload(read_json(ROOT / relative))
        )
    v5_ids = {
        canonical_case_id(value) for value in v5_cohort["case_ids"]
    }
    v6_frame = {
        case_id
        for case_id, case in cases_by_id.items()
        if case_id not in legacy_ids | v5_ids and is_v6_eligible_case(case)
    }
    return (
        v6_frame
        - v6_development_ids
        - v6_confirmation_ids
        - v7_development_ids
        - v7_confirmation_ids
    )


def select_by_hash(
    values: Iterable[str],
    *,
    count: int,
    domain: str,
    seed: str,
) -> set[str]:
    ordered = sorted(values, key=lambda case_id: hash_order_key(domain, seed, case_id))
    if len(ordered) < count:
        raise RuntimeError(
            f"Cannot select {count} cases from a frame containing {len(ordered)} cases."
        )
    return set(ordered[:count])


def complete_qa_reference(case: dict[str, Any]) -> bool:
    return bool(clean_text(case.get("findings"))) and bool(
        clean_text(case.get("impression"))
    )


def partition_summary(
    case_ids: set[str], cases_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    normal = {
        case_id
        for case_id in case_ids
        if report_index_class(cases_by_id[case_id]) == "report_indexed_normal"
    }
    abnormal = case_ids - normal
    complete_qa = {
        case_id for case_id in case_ids if complete_qa_reference(cases_by_id[case_id])
    }
    return {
        "case_count": len(case_ids),
        "report_indexed_normal": len(normal),
        "report_indexed_abnormal": len(abnormal),
        "complete_findings_and_impression_reference": len(complete_qa),
        "case_ids_sha256": case_id_fingerprint(case_ids),
        "complete_reference_case_ids_sha256": case_id_fingerprint(complete_qa),
        "case_ids": sorted(case_ids),
    }


def write_id_file(path: Path, values: set[str]) -> None:
    path.write_text("\n".join(sorted(values)), encoding="utf-8", newline="")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instantiate the frozen V9 OpenI full-source split."
    )
    parser.add_argument("--protocol-commit", default=DEFAULT_PROTOCOL_COMMIT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--v5-config", type=Path, default=DEFAULT_V5_CONFIG)
    parser.add_argument("--v5-cohort", type=Path, default=DEFAULT_V5_COHORT)
    parser.add_argument("--v6-development", type=Path, default=DEFAULT_V6_DEVELOPMENT)
    parser.add_argument("--v6-confirmation", type=Path, default=DEFAULT_V6_CONFIRMATION)
    parser.add_argument("--v7-development", type=Path, default=DEFAULT_V7_DEVELOPMENT)
    parser.add_argument("--v7-confirmation", type=Path, default=DEFAULT_V7_CONFIRMATION)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--freeze-output", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()

    config = read_json(args.config)
    if config != committed_json(args.protocol_commit, args.config):
        raise RuntimeError("Current split config differs from the committed protocol config.")
    if config["split_ids_instantiated"] is not False:
        raise RuntimeError("Protocol config unexpectedly contains instantiated split IDs.")
    if file_sha256(args.cases) != config["source"]["sha256"]:
        raise RuntimeError("OpenI source differs from the frozen split protocol.")

    cases = read_jsonl(args.cases)
    cases_by_id = {
        canonical_case_id(case["case_id"]): case for case in cases
    }
    if len(cases_by_id) != len(cases) or len(cases) != 3851:
        raise RuntimeError("OpenI source case count or case-ID uniqueness changed.")

    strata = {
        name: {
            case_id
            for case_id, case in cases_by_id.items()
            if report_index_class(case) == name
        }
        for name in (
            "report_indexed_normal",
            "report_indexed_abnormal",
            "report_index_indeterminate",
        )
    }
    normal = strata["report_indexed_normal"]
    abnormal = strata["report_indexed_abnormal"]
    indeterminate = strata["report_index_indeterminate"]
    stratifiable = normal | abnormal
    expected = config["eligibility"]
    observed_counts = (len(normal), len(abnormal), len(indeterminate), len(stratifiable))
    expected_counts = (
        expected["report_indexed_normal"],
        expected["report_indexed_abnormal"],
        expected["report_index_indeterminate"],
        expected["stratifiable_total"],
    )
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"V9 source strata changed: expected {expected_counts}, found {observed_counts}."
        )

    v7_development = read_json(args.v7_development)
    strict_frame = historical_untouched_frame(
        cases_by_id=cases_by_id,
        v5_config=read_json(args.v5_config),
        v5_cohort=read_json(args.v5_cohort),
        v6_development_ids=read_id_file(args.v6_development),
        v6_confirmation_ids={
            canonical_case_id(value)
            for value in read_json(args.v6_confirmation)["case_ids"]
        },
        v7_development_ids={
            canonical_case_id(value)
            for block in v7_development["blocks"].values()
            for value in block["case_ids"]
        },
        v7_confirmation_ids={
            canonical_case_id(value)
            for value in read_json(args.v7_confirmation)["case_ids"]
        },
    )
    strict_stratifiable = strict_frame & stratifiable
    strict_normal = strict_stratifiable & normal
    strict_abnormal = strict_stratifiable & abnormal
    strict_config = config["strict_project_history_untouched_subset"]
    if (
        len(strict_frame) != 279
        or len(strict_normal) != strict_config["normal"]
        or len(strict_abnormal) != strict_config["abnormal"]
        or case_id_fingerprint(strict_stratifiable) != strict_config["case_ids_sha256"]
    ):
        raise RuntimeError("Strict project-history-untouched frame failed its frozen audit.")

    seed = config["deterministic_selection"]["seed"]
    test_domain = config["deterministic_selection"]["test_supplement_domain"]
    validation_domain = config["deterministic_selection"]["validation_domain"]
    supplement_config = config["test_supplement"]
    supplement_normal = select_by_hash(
        normal - strict_stratifiable,
        count=supplement_config["normal"],
        domain=test_domain,
        seed=seed,
    )
    supplement_abnormal = select_by_hash(
        abnormal - strict_stratifiable,
        count=supplement_config["abnormal"],
        domain=test_domain,
        seed=seed,
    )
    test_ids = strict_stratifiable | supplement_normal | supplement_abnormal

    remaining = stratifiable - test_ids
    validation_config = config["partitions"]["validation"]
    validation_normal = select_by_hash(
        remaining & normal,
        count=validation_config["normal"],
        domain=validation_domain,
        seed=seed,
    )
    validation_abnormal = select_by_hash(
        remaining & abnormal,
        count=validation_config["abnormal"],
        domain=validation_domain,
        seed=seed,
    )
    validation_ids = validation_normal | validation_abnormal
    train_ids = remaining - validation_ids

    partitions = {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids,
    }
    if set.union(*partitions.values()) != stratifiable:
        raise RuntimeError("V9 partitions do not cover the stratifiable universe.")
    if any(
        left != right and partitions[left] & partitions[right]
        for left in partitions
        for right in partitions
    ):
        raise RuntimeError("V9 partitions overlap.")
    summaries = {
        name: partition_summary(case_ids, cases_by_id)
        for name, case_ids in partitions.items()
    }
    for name, summary in summaries.items():
        expected_partition = config["partitions"][name]
        if (
            summary["case_count"] != expected_partition["total"]
            or summary["report_indexed_normal"] != expected_partition["normal"]
            or summary["report_indexed_abnormal"] != expected_partition["abnormal"]
        ):
            raise RuntimeError(f"V9 {name} composition differs from the protocol.")

    complete_source = {
        case_id for case_id in stratifiable if complete_qa_reference(cases_by_id[case_id])
    }
    if len(complete_source) != config["qa_complete_reference_source_count"]:
        raise RuntimeError("Complete-reference QA source count changed.")

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "benchmark": "OpenI V9 full-source similar-case RAG split",
        "version": "9.0-full-source",
        "status": "instantiated_after_split_protocol_commit_before_v9_outcomes",
        "protocol_commit": args.protocol_commit,
        "config_path": portable_path(args.config),
        "config_sha256": file_sha256(args.config),
        "source_path": portable_path(args.cases),
        "source_sha256": file_sha256(args.cases),
        "patient_identity_claim": "source-design patient uniqueness",
        "partitions": summaries,
        "strict_project_history_untouched_test_subset": {
            "case_count": len(strict_stratifiable),
            "report_indexed_normal": len(strict_normal),
            "report_indexed_abnormal": len(strict_abnormal),
            "case_ids_sha256": case_id_fingerprint(strict_stratifiable),
            "case_ids": sorted(strict_stratifiable),
        },
        "excluded_primary_graded_frame": {
            "reason": "report_index_indeterminate",
            "case_count": len(indeterminate),
            "case_ids_sha256": case_id_fingerprint(indeterminate),
        },
        "qa_complete_reference_source_count": len(complete_source),
        "split_ids_instantiated": True,
        "v9_outcomes_inspected": False,
    }
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for name, case_ids in partitions.items():
        write_id_file(args.manifest_output.with_name(f"v9_{name}_case_ids.txt"), case_ids)
    write_id_file(
        args.manifest_output.with_name("v9_strict_untouched_test_case_ids.txt"),
        strict_stratifiable,
    )

    freeze = {
        "freeze": "V9 full-source split instantiation",
        "status": "frozen_before_v9_outcome_execution",
        "protocol_commit": args.protocol_commit,
        "builder_path": portable_path(Path(__file__)),
        "builder_sha256": file_sha256(Path(__file__)),
        "config_path": portable_path(args.config),
        "config_sha256": file_sha256(args.config),
        "source_path": portable_path(args.cases),
        "source_sha256": file_sha256(args.cases),
        "stratifiable_case_count": len(stratifiable),
        "stratifiable_case_ids_sha256": case_id_fingerprint(stratifiable),
        "report_index_indeterminate_case_count": len(indeterminate),
        "report_index_indeterminate_case_ids_sha256": case_id_fingerprint(indeterminate),
        "partition_fingerprints": {
            name: {
                "case_count": summary["case_count"],
                "case_ids_sha256": summary["case_ids_sha256"],
                "complete_reference_case_count": summary[
                    "complete_findings_and_impression_reference"
                ],
                "complete_reference_case_ids_sha256": summary[
                    "complete_reference_case_ids_sha256"
                ],
            }
            for name, summary in summaries.items()
        },
        "strict_untouched_test_subset": {
            "case_count": len(strict_stratifiable),
            "case_ids_sha256": case_id_fingerprint(strict_stratifiable),
            "is_subset_of_test": strict_stratifiable <= test_ids,
        },
        "overlap_counts": {
            "train_validation": len(train_ids & validation_ids),
            "train_test": len(train_ids & test_ids),
            "validation_test": len(validation_ids & test_ids),
        },
        "manifest_path": portable_path(args.manifest_output),
        "manifest_sha256": file_sha256(args.manifest_output),
        "v9_outcomes_inspected": False,
    }
    args.freeze_output.write_text(
        json.dumps(freeze, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(freeze, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
