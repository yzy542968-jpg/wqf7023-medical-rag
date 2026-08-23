from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import (  # noqa: E402
    canonical_case_id,
    case_id_fingerprint,
    file_sha256,
    read_json,
)
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v9_learned_reranker_development.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v9" / "v9_full_source_split.json"
DEFAULT_OUTPUT = ROOT / "data" / "splits" / "v9" / "v9_reranker_role_manifest.json"


def hash_key(domain: str, seed: int, case_id: str) -> tuple[str, str]:
    canonical = canonical_case_id(case_id)
    digest = hashlib.sha256(f"{domain}|{seed}|{canonical}".encode("utf-8")).hexdigest()
    return digest, canonical


def main() -> None:
    parser = argparse.ArgumentParser(description="Instantiate frozen V9 reranker roles.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config = read_json(args.config)
    split = read_json(args.split)
    cases = read_openi_paired_cases(
        args.cases, source_unique_patient=True, radgraph_path=args.radgraph
    )
    by_id = {case.study_id: case for case in cases}
    train_ids = {
        canonical_case_id(value) for value in split["partitions"]["train"]["case_ids"]
    }
    eligible = {
        case_id
        for case_id in train_ids
        if by_id[case_id].metadata["radgraph_annotation_available"] is True
    }
    normal = {
        case_id
        for case_id in eligible
        if by_id[case_id].metadata["report_index_class"] == "normal"
    }
    abnormal = eligible - normal
    if len(normal) != config["train_bank"]["normal"] or len(abnormal) != config["train_bank"]["abnormal"]:
        raise RuntimeError("V9 train-bank report-index spectrum changed.")

    ordered = {
        "normal": sorted(
            normal,
            key=lambda case_id: hash_key(config["role_domain"], int(config["seed"]), case_id),
        ),
        "abnormal": sorted(
            abnormal,
            key=lambda case_id: hash_key(config["role_domain"], int(config["seed"]), case_id),
        ),
    }
    roles: dict[str, set[str]] = {name: set() for name in config["roles"]}
    offsets = {"normal": 0, "abnormal": 0}
    for role in ("fit", "internal_early_stop", "bank_only"):
        for stratum in ("normal", "abnormal"):
            count = int(config["roles"][role][stratum])
            start = offsets[stratum]
            roles[role].update(ordered[stratum][start : start + count])
            offsets[stratum] += count
    if set.union(*roles.values()) != eligible:
        raise RuntimeError("Reranker roles do not cover the eligible train bank.")
    if any(
        left != right and roles[left] & roles[right]
        for left in roles
        for right in roles
    ):
        raise RuntimeError("Reranker roles overlap.")

    output: dict[str, Any] = {
        "manifest": "V9 learned-reranker train roles",
        "status": "instantiated_after_protocol_commit_before_training",
        "protocol_commit": "3098c8c",
        "config_path": "config/v9_learned_reranker_development.json",
        "config_sha256": file_sha256(args.config),
        "source_sha256": file_sha256(args.cases),
        "radgraph_sha256": file_sha256(args.radgraph),
        "split_sha256": file_sha256(args.split),
        "seed": config["seed"],
        "role_domain": config["role_domain"],
        "eligible_train_bank": {
            "case_count": len(eligible),
            "case_ids_sha256": case_id_fingerprint(eligible),
        },
        "roles": {},
        "training_started": False,
        "validation_outcomes_inspected_for_reranker": False,
        "test_queries_executed": 0,
    }
    for role, ids in roles.items():
        role_normal = ids & normal
        role_abnormal = ids & abnormal
        expected = config["roles"][role]
        if (
            len(ids) != expected["total"]
            or len(role_normal) != expected["normal"]
            or len(role_abnormal) != expected["abnormal"]
        ):
            raise RuntimeError(f"Reranker role {role} composition changed.")
        output["roles"][role] = {
            "case_count": len(ids),
            "report_indexed_normal": len(role_normal),
            "report_indexed_abnormal": len(role_abnormal),
            "case_ids_sha256": case_id_fingerprint(ids),
            "case_ids": sorted(ids),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({key: value for key, value in output.items() if key != "roles"}, indent=2))
    print(
        json.dumps(
            {
                role: {
                    key: value for key, value in block.items() if key != "case_ids"
                }
                for role, block in output["roles"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
