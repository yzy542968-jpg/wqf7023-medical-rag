from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import (  # noqa: E402
    canonical_case_id,
    case_ids_from_payload,
    case_id_fingerprint,
    file_sha256,
    is_v6_eligible_case,
    read_json,
    read_jsonl,
    report_index_class,
)


DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_V5_CONFIG = ROOT / "config" / "multimodal_v5.json"
DEFAULT_V5_COHORT = ROOT / "data" / "processed" / "openi_multimodal_v5_cohort.json"
DEFAULT_V6_DEVELOPMENT = ROOT / "data" / "splits" / "v6" / "v6_development_case_ids.txt"
DEFAULT_V6_CONFIRMATION = ROOT / "data" / "splits" / "v6" / "v6_confirmation_cohort.json"
DEFAULT_V7_DEVELOPMENT = ROOT / "data" / "splits" / "v7" / "v7_development_manifest.json"
DEFAULT_V7_CONFIRMATION = ROOT / "data" / "splits" / "v7" / "v7_confirmation_cohort.json"
DEFAULT_OUTPUT = ROOT / "data" / "splits" / "v8" / "v8_reuse_audit.json"


def read_case_id_file(path: Path) -> set[str]:
    return {
        canonical_case_id(value)
        for value in path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V8 development reuse and confirmation frame.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--v5-config", type=Path, default=DEFAULT_V5_CONFIG)
    parser.add_argument("--v5-cohort", type=Path, default=DEFAULT_V5_COHORT)
    parser.add_argument("--v6-development", type=Path, default=DEFAULT_V6_DEVELOPMENT)
    parser.add_argument("--v6-confirmation", type=Path, default=DEFAULT_V6_CONFIRMATION)
    parser.add_argument("--v7-development", type=Path, default=DEFAULT_V7_DEVELOPMENT)
    parser.add_argument("--v7-confirmation", type=Path, default=DEFAULT_V7_CONFIRMATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases = read_jsonl(args.cases)
    by_id = {canonical_case_id(case["case_id"]): case for case in cases}
    v5_config = read_json(args.v5_config)
    legacy_ids: set[str] = set()
    for relative in v5_config["cohort"]["excluded_source_manifests"]:
        legacy_ids.update(
            canonical_case_id(value)
            for value in case_ids_from_payload(read_json(ROOT / relative))
        )
    v5_ids = {
        canonical_case_id(value)
        for value in read_json(args.v5_cohort)["case_ids"]
    }
    v6_development_ids = read_case_id_file(args.v6_development)
    v6_confirmation_ids = {
        canonical_case_id(value)
        for value in read_json(args.v6_confirmation)["case_ids"]
    }
    v7_development = read_json(args.v7_development)
    v7_development_ids = {
        canonical_case_id(value)
        for block in v7_development["blocks"].values()
        for value in block["case_ids"]
    }
    v7_confirmation_ids = {
        canonical_case_id(value)
        for value in read_json(args.v7_confirmation)["case_ids"]
    }

    v6_frame = {
        cid
        for cid, case in by_id.items()
        if cid not in legacy_ids | v5_ids and is_v6_eligible_case(case)
    }
    v8_frame = v6_frame - v6_development_ids - v6_confirmation_ids
    v8_confirmation_frame = v8_frame - v7_development_ids - v7_confirmation_ids
    classes = Counter(report_index_class(by_id[cid]) for cid in v8_confirmation_frame)
    stratifiable = {
        cid
        for cid in v8_confirmation_frame
        if report_index_class(by_id[cid]) in {"report_indexed_normal", "report_indexed_abnormal"}
    }
    if len(v8_confirmation_frame) != 279 or len(stratifiable) != 262:
        raise RuntimeError("V8 confirmation frame counts changed from the audited state.")
    if classes["report_indexed_normal"] != 185 or classes["report_indexed_abnormal"] != 77 or classes["report_index_indeterminate"] != 17:
        raise RuntimeError("V8 confirmation frame strata changed from the audited state.")
    if len(v7_development_ids & v8_confirmation_frame) or len(v7_confirmation_ids & v8_confirmation_frame):
        raise RuntimeError("V8 confirmation frame overlaps V7 use.")
    if len(v8_confirmation_frame & (legacy_ids | v5_ids | v6_development_ids | v6_confirmation_ids)):
        raise RuntimeError("V8 confirmation frame overlaps formal prior use.")

    output = {
        "audit": "V8 technology reuse and confirmation frame audit",
        "version": "1.0",
        "status": "development_protocol_frame_audited_confirmation_ids_not_instantiated",
        "source": {
            "path": str(args.cases.relative_to(ROOT)).replace("\\", "/"),
            "case_count": len(by_id),
            "sha256": file_sha256(args.cases),
        },
        "development": {
            "path": str(args.v7_development.relative_to(ROOT)).replace("\\", "/"),
            "case_count": len(v7_development_ids),
            "case_ids_sha256": case_id_fingerprint(v7_development_ids),
            "blocks": sorted(v7_development["blocks"]),
            "confirmation_overlap_count": len(v7_development_ids & v8_confirmation_frame),
        },
        "v7_confirmation": {
            "path": str(args.v7_confirmation.relative_to(ROOT)).replace("\\", "/"),
            "case_count": len(v7_confirmation_ids),
            "case_ids_sha256": case_id_fingerprint(v7_confirmation_ids),
            "confirmation_overlap_count": len(v7_confirmation_ids & v8_confirmation_frame),
        },
        "v8_confirmation_frame": {
            "eligible_case_count": len(v8_confirmation_frame),
            "eligible_case_ids_sha256": case_id_fingerprint(v8_confirmation_frame),
            "stratifiable_case_count": len(stratifiable),
            "stratifiable_case_ids_sha256": case_id_fingerprint(stratifiable),
            "report_indexed_normal_count": classes["report_indexed_normal"],
            "report_indexed_abnormal_count": classes["report_indexed_abnormal"],
            "report_index_indeterminate_count": classes["report_index_indeterminate"],
            "case_ids_instantiated": False,
        },
        "predefined_confirmation_composition": {
            "candidate_pool_case_count": 240,
            "report_indexed_normal": 170,
            "report_indexed_abnormal": 70,
            "target_normal": 85,
            "target_abnormal": 35,
            "distractor_normal": 85,
            "distractor_abnormal": 35,
        },
        "overlap_checks": {
            "v7_development_with_v8_confirmation_frame": len(v7_development_ids & v8_confirmation_frame),
            "v7_confirmation_with_v8_confirmation_frame": len(v7_confirmation_ids & v8_confirmation_frame),
            "formal_prior_with_v8_confirmation_frame": len(v8_confirmation_frame & (legacy_ids | v5_ids | v6_development_ids | v6_confirmation_ids)),
        },
        "patient_level_independence_verified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
