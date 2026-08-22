from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import (  # noqa: E402
    build_audit as build_v6_audit,
    canonical_case_id,
    canonical_ids,
    case_id_fingerprint,
    case_ids_from_payload,
    file_sha256,
    is_v6_eligible_case,
    read_json,
    read_jsonl,
    report_index_class,
)


DEFAULT_CONFIG = ROOT / "config" / "v7_adaptive_fusion_development.json"
DEFAULT_V5_CONFIG = ROOT / "config" / "multimodal_v5.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_V5_COHORT = ROOT / "data" / "processed" / "openi_multimodal_v5_cohort.json"
DEFAULT_V6_DEVELOPMENT = ROOT / "data" / "splits" / "v6" / "v6_development_case_ids.txt"
DEFAULT_V6_CONFIRMATION = ROOT / "data" / "splits" / "v6" / "v6_confirmation_cohort.json"
DEFAULT_AUDIT_OUTPUT = ROOT / "data" / "splits" / "v7" / "v7_prior_use_audit.json"
DEFAULT_MANIFEST_OUTPUT = ROOT / "data" / "splits" / "v7" / "v7_development_manifest.json"


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def read_case_id_file(path: Path) -> set[str]:
    return {
        canonical_case_id(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def hash_order_key(domain: str, seed: int, case_id: str) -> str:
    payload = f"{domain}|{seed}|{canonical_case_id(case_id)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordered_ids(values: Iterable[str], *, domain: str, seed: int) -> list[str]:
    return sorted(
        {canonical_case_id(value) for value in values},
        key=lambda case_id: (hash_order_key(domain, seed, case_id), case_id),
    )


def build_role_split(
    normal_ids: list[str],
    abnormal_ids: list[str],
    *,
    block_name: str,
    seed: int,
    assignment_domain: str,
) -> dict[str, Any]:
    if len(normal_ids) != 172 or len(abnormal_ids) != 68:
        raise ValueError(f"{block_name} does not have the frozen 172/68 composition.")
    normal_ordered = ordered_ids(normal_ids, domain=assignment_domain, seed=seed)
    abnormal_ordered = ordered_ids(abnormal_ids, domain=assignment_domain, seed=seed)
    targets = normal_ordered[:86] + abnormal_ordered[:34]
    distractors = normal_ordered[86:] + abnormal_ordered[34:]
    selected = sorted(normal_ids + abnormal_ids)
    if set(targets) | set(distractors) != set(selected):
        raise ValueError(f"{block_name} target/distractor roles do not cover the block.")
    if set(targets) & set(distractors):
        raise ValueError(f"{block_name} target/distractor roles overlap.")
    return {
        "case_count": 240,
        "report_indexed_normal": 172,
        "report_indexed_abnormal": 68,
        "case_ids": selected,
        "target_case_ids": sorted(targets),
        "distractor_case_ids": sorted(distractors),
        "target_report_indexed_normal": 86,
        "target_report_indexed_abnormal": 34,
        "distractor_report_indexed_normal": 86,
        "distractor_report_indexed_abnormal": 34,
    }


def build_development_blocks(
    *,
    normal_ids: set[str],
    abnormal_ids: set[str],
    seed: int,
    selection_domain: str,
    assignment_domain: str,
) -> tuple[dict[str, Any], set[str]]:
    normal_ordered = ordered_ids(normal_ids, domain=selection_domain, seed=seed)
    abnormal_ordered = ordered_ids(abnormal_ids, domain=selection_domain, seed=seed)
    block_names = ["train_a", "train_b", "validation"]
    required_normal = 172 * len(block_names)
    required_abnormal = 68 * len(block_names)
    if len(normal_ordered) < required_normal or len(abnormal_ordered) < required_abnormal:
        raise RuntimeError("The V7 source frame cannot produce three development blocks.")

    blocks: dict[str, Any] = {}
    development_ids: set[str] = set()
    for index, block_name in enumerate(block_names):
        start_normal = index * 172
        start_abnormal = index * 68
        block_normal = normal_ordered[start_normal : start_normal + 172]
        block_abnormal = abnormal_ordered[start_abnormal : start_abnormal + 68]
        block = build_role_split(
            block_normal,
            block_abnormal,
            block_name=block_name,
            seed=seed,
            assignment_domain=assignment_domain,
        )
        blocks[block_name] = block
        development_ids.update(block["case_ids"])

    if len(development_ids) != 720:
        raise RuntimeError("V7 development blocks are not case-ID disjoint.")
    return blocks, development_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit prior case use and instantiate V7 Train A, Train B, and Validation "
            "manifests without generating confirmation IDs."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v5-config", type=Path, default=DEFAULT_V5_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--v5-cohort", type=Path, default=DEFAULT_V5_COHORT)
    parser.add_argument("--v6-development", type=Path, default=DEFAULT_V6_DEVELOPMENT)
    parser.add_argument("--v6-confirmation", type=Path, default=DEFAULT_V6_CONFIRMATION)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()

    config = read_json(args.config)
    if config["cohort_generation"]["case_ids_instantiated"] is not False:
        raise RuntimeError("V7 config must not contain instantiated confirmation IDs.")
    seed = int(config["cohort_generation"]["seed"])
    selection_domain = str(config["cohort_generation"]["selection_domain"])
    assignment_domain = str(config["cohort_generation"]["assignment_domain"])

    all_cases = read_jsonl(args.cases)
    cases = {canonical_case_id(row["case_id"]): row for row in all_cases}
    if len(cases) != len(all_cases):
        raise RuntimeError("Source case IDs are not unique after canonicalization.")

    v5_config = read_json(args.v5_config)
    v5_cohort = read_json(args.v5_cohort)
    legacy_paths = [ROOT / relative for relative in v5_config["cohort"]["excluded_source_manifests"]]
    legacy_ids: set[str] = set()
    legacy_hashes: dict[str, str] = {}
    for path in legacy_paths:
        ids = case_ids_from_payload(read_json(path))
        legacy_ids.update(ids)
        legacy_hashes[portable_path(path)] = file_sha256(path)

    v5_ids = {canonical_case_id(value) for value in v5_cohort["case_ids"]}
    v6_development_ids = read_case_id_file(args.v6_development)
    v6_confirmation_payload = read_json(args.v6_confirmation)
    v6_confirmation_ids = {
        canonical_case_id(value) for value in v6_confirmation_payload["case_ids"]
    }
    if not v6_confirmation_ids.issubset(cases):
        raise RuntimeError("V6 confirmation manifest contains IDs absent from source.")

    v6_audit, v6_collections = build_v6_audit(
        all_cases=all_cases,
        prior_excluded_ids=legacy_ids,
        v5_case_ids=v5_ids,
        development_ids={
            canonical_case_id(value)
            for value in v5_cohort["split"]["development"]["case_ids"]
        },
    )
    expected_v6_counts = {
        "eligible": 1479,
        "stratifiable": 1462,
        "report_indexed_normal": 1045,
        "report_indexed_abnormal": 417,
        "report_index_indeterminate": 17,
    }
    observed_v6_counts = {
        "eligible": len(v6_collections["eligible"]),
        "stratifiable": len(v6_collections["stratifiable"]),
        "report_indexed_normal": len(v6_collections["report_indexed_normal"]),
        "report_indexed_abnormal": len(v6_collections["report_indexed_abnormal"]),
        "report_index_indeterminate": len(v6_collections["report_index_indeterminate"]),
    }
    if observed_v6_counts != expected_v6_counts:
        raise RuntimeError(
            f"V6 selection frame changed: expected {expected_v6_counts}, "
            f"observed {observed_v6_counts}."
        )
    if not v6_confirmation_ids.issubset(set(v6_collections["eligible"])):
        raise RuntimeError("V6 confirmation IDs are not contained in the V6 eligible frame.")
    if set(v6_development_ids) & set(v6_collections["eligible"]):
        raise RuntimeError("V6 development overlaps the V6 eligible frame.")

    v7_eligible_ids = set(v6_collections["eligible"]) - v6_confirmation_ids - v6_development_ids
    v7_strata = {
        "report_indexed_normal": {
            case_id for case_id in v7_eligible_ids if report_index_class(cases[case_id]) == "report_indexed_normal"
        },
        "report_indexed_abnormal": {
            case_id for case_id in v7_eligible_ids if report_index_class(cases[case_id]) == "report_indexed_abnormal"
        },
        "report_index_indeterminate": {
            case_id for case_id in v7_eligible_ids if report_index_class(cases[case_id]) == "report_index_indeterminate"
        },
    }
    observed_v7_counts = {name: len(ids) for name, ids in v7_strata.items()}
    expected_v7_counts = {
        "report_indexed_normal": 873,
        "report_indexed_abnormal": 349,
        "report_index_indeterminate": 17,
    }
    if observed_v7_counts != expected_v7_counts:
        raise RuntimeError(
            f"V7 source frame changed: expected {expected_v7_counts}, "
            f"observed {observed_v7_counts}."
        )

    blocks, development_ids = build_development_blocks(
        normal_ids=v7_strata["report_indexed_normal"],
        abnormal_ids=v7_strata["report_indexed_abnormal"],
        seed=seed,
        selection_domain=selection_domain,
        assignment_domain=assignment_domain,
    )
    formal_prior_ids = legacy_ids | v5_ids | v6_development_ids | v6_confirmation_ids
    if formal_prior_ids & development_ids:
        raise RuntimeError("V7 development manifest overlaps a prior-use case ID.")
    if any(set(block["case_ids"]) & set(v7_strata["report_index_indeterminate"]) for block in blocks.values()):
        raise RuntimeError("V7 development blocks contain indeterminate report-index cases.")

    v7_stratifiable_ids = v7_strata["report_indexed_normal"] | v7_strata["report_indexed_abnormal"]
    post_development_confirmation_frame_ids = v7_eligible_ids - development_ids
    post_development_confirmation_stratifiable_ids = v7_stratifiable_ids - development_ids
    manifest = {
        "benchmark": "OpenI V7 adaptive multimodal fusion development blocks",
        "version": "7.0-development",
        "status": "development_blocks_instantiated_confirmation_ids_not_generated",
        "protocol_path": portable_path(ROOT / "docs" / "V7_DEVELOPMENT_PROTOCOL.md"),
        "config_path": portable_path(args.config),
        "config_sha256": file_sha256(args.config),
        "source_cases": portable_path(args.cases),
        "source_cases_sha256": file_sha256(args.cases),
        "seed": seed,
        "selection_domain": selection_domain,
        "assignment_domain": assignment_domain,
        "case_id_canonicalization": "str(value).strip()",
        "case_id_fingerprint_serialization": "sorted unique canonical IDs joined by LF UTF-8 with no trailing LF",
        "blocks": blocks,
        "development_case_count": len(development_ids),
        "development_case_ids_sha256": case_id_fingerprint(development_ids),
        "confirmation_case_ids_instantiated": False,
        "post_development_confirmation_frame_count": len(post_development_confirmation_frame_ids),
        "post_development_confirmation_stratifiable_count": len(post_development_confirmation_stratifiable_ids),
    }

    audit = {
        "audit": "V7 prior-use and development-frame audit",
        "version": "1.0",
        "status": "development_manifest_instantiated_confirmation_ids_not_generated",
        "protocol_commit": "2ec6dce",
        "patient_level_independence_verified": False,
        "patient_level_boundary": "Reliable patient or subject identifiers are unavailable; only case-ID disjointness is asserted.",
        "source": {
            "path": portable_path(args.cases),
            "case_count": len(all_cases),
            "sha256": file_sha256(args.cases),
            "case_ids_sha256": case_id_fingerprint(cases),
        },
        "formal_prior_use": {
            "legacy_manifest_paths": [portable_path(path) for path in legacy_paths],
            "legacy_manifest_sha256": legacy_hashes,
            "legacy_case_count": len(legacy_ids),
            "legacy_case_ids_sha256": case_id_fingerprint(legacy_ids),
            "v5_cohort_path": portable_path(args.v5_cohort),
            "v5_cohort_sha256": file_sha256(args.v5_cohort),
            "v5_case_count": len(v5_ids),
            "v5_case_ids_sha256": case_id_fingerprint(v5_ids),
            "v6_development_path": portable_path(args.v6_development),
            "v6_development_sha256": file_sha256(args.v6_development),
            "v6_development_case_count": len(v6_development_ids),
            "v6_development_case_ids_sha256": case_id_fingerprint(v6_development_ids),
            "v6_confirmation_path": portable_path(args.v6_confirmation),
            "v6_confirmation_sha256": file_sha256(args.v6_confirmation),
            "v6_confirmation_case_count": len(v6_confirmation_ids),
            "v6_confirmation_case_ids_sha256": case_id_fingerprint(v6_confirmation_ids),
            "union_case_count": len(formal_prior_ids),
            "union_case_ids_sha256": case_id_fingerprint(formal_prior_ids),
        },
        "v6_frame_recomputed": {
            "eligible_case_count": len(v6_collections["eligible"]),
            "eligible_case_ids_sha256": case_id_fingerprint(v6_collections["eligible"]),
            "stratifiable_case_count": len(v6_collections["stratifiable"]),
            "stratifiable_case_ids_sha256": case_id_fingerprint(v6_collections["stratifiable"]),
            "expected_counts_match": observed_v6_counts == expected_v6_counts,
        },
        "v7_frame": {
            "eligible_case_count": len(v7_eligible_ids),
            "eligible_case_ids_sha256": case_id_fingerprint(v7_eligible_ids),
            "stratifiable_case_count": len(v7_stratifiable_ids),
            "stratifiable_case_ids_sha256": case_id_fingerprint(v7_stratifiable_ids),
            "report_indexed_normal_count": len(v7_strata["report_indexed_normal"]),
            "report_indexed_normal_ids_sha256": case_id_fingerprint(v7_strata["report_indexed_normal"]),
            "report_indexed_abnormal_count": len(v7_strata["report_indexed_abnormal"]),
            "report_indexed_abnormal_ids_sha256": case_id_fingerprint(v7_strata["report_indexed_abnormal"]),
            "report_index_indeterminate_count": len(v7_strata["report_index_indeterminate"]),
            "report_index_indeterminate_ids_sha256": case_id_fingerprint(v7_strata["report_index_indeterminate"]),
            "expected_counts_match": observed_v7_counts == expected_v7_counts,
        },
        "overlap_checks": {
            "development_with_formal_prior_use_count": len(development_ids & formal_prior_ids),
            "development_membership_in_pre_development_frame_count": len(development_ids & v7_eligible_ids),
            "development_with_post_development_confirmation_frame_count": len(
                development_ids & post_development_confirmation_frame_ids
            ),
            "development_with_post_development_confirmation_stratifiable_count": len(
                development_ids & post_development_confirmation_stratifiable_ids
            ),
            "development_blocks_pairwise_overlap_count": 0,
            "all_required_disjointness_checks_zero": True,
        },
        "post_development_confirmation_frame": {
            "case_count": len(post_development_confirmation_frame_ids),
            "case_ids_sha256": case_id_fingerprint(post_development_confirmation_frame_ids),
            "stratifiable_case_count": len(post_development_confirmation_stratifiable_ids),
            "stratifiable_case_ids_sha256": case_id_fingerprint(post_development_confirmation_stratifiable_ids),
            "case_ids_instantiated": False,
        },
        "development_manifest": portable_path(args.manifest_output),
        "development_manifest_sha256": file_sha256(args.manifest_output) if args.manifest_output.exists() else None,
        "excluded_from_primary_stratification": {
            "report_indexed_indeterminate_count": len(v7_strata["report_index_indeterminate"]),
            "reason": "problems field equals no indexing; not treated as normal or abnormal",
        },
    }
    required_zero_checks = (
        "development_with_formal_prior_use_count",
        "development_with_post_development_confirmation_frame_count",
        "development_with_post_development_confirmation_stratifiable_count",
        "development_blocks_pairwise_overlap_count",
    )
    if any(audit["overlap_checks"][key] != 0 for key in required_zero_checks):
        raise RuntimeError("A V7 development overlap check is non-zero.")

    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    audit["development_manifest_sha256"] = file_sha256(args.manifest_output)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"audit": audit, "manifest": manifest}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
