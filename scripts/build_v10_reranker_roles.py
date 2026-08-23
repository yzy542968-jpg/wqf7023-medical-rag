from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v10" / "v10_cluster_disjoint_split.json"
DEFAULT_CONFIG = ROOT / "config" / "v10_reranker_development.json"
DEFAULT_OUTPUT = ROOT / "data" / "splits" / "v10" / "v10_reranker_roles.json"


def role_value(domain: str, seed: int, cluster_id: str) -> float:
    digest = hashlib.sha256(f"{domain}|{seed}|{cluster_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def select_role(value: float, intervals: dict[str, list[float]]) -> str:
    for role, (lower, upper) in intervals.items():
        if float(lower) <= value < float(upper):
            return role
    raise ValueError(f"role value {value} is outside configured intervals")


def main() -> None:
    parser = argparse.ArgumentParser(description="Instantiate V10 Train-cluster reranker roles.")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["validation_outcomes_inspected"] or config["test_outcomes_inspected"]:
        raise RuntimeError("role config records inspected outcomes")
    train_ids = set(split["partitions"]["train"]["case_ids"])
    train_clusters = [
        row for row in split["clusters"] if set(row["case_ids"]) <= train_ids
    ]
    covered = {case_id for row in train_clusters for case_id in row["case_ids"]}
    if covered != train_ids:
        raise RuntimeError("Train clusters do not exactly cover Train cases")

    roles: dict[str, dict[str, Any]] = {
        role: {"cluster_ids": [], "case_ids": []}
        for role in config["role_intervals"]
    }
    for cluster in train_clusters:
        value = role_value(config["role_domain"], int(config["role_seed"]), cluster["cluster_id"])
        role = select_role(value, config["role_intervals"])
        roles[role]["cluster_ids"].append(cluster["cluster_id"])
        roles[role]["case_ids"].extend(cluster["case_ids"])
    for payload in roles.values():
        payload["cluster_ids"].sort()
        payload["case_ids"].sort()
        payload["cluster_count"] = len(payload["cluster_ids"])
        payload["case_count"] = len(payload["case_ids"])
        payload["case_ids_sha256"] = hashlib.sha256("\n".join(payload["case_ids"]).encode("utf-8")).hexdigest()

    manifest = {
        "study": "V10 reranker Train roles",
        "status": "instantiated_before_training_or_validation_outcomes",
        "split_manifest_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "roles": roles,
        "role_case_overlap_count": sum(
            len(set(roles[left]["case_ids"]) & set(roles[right]["case_ids"]))
            for index, left in enumerate(roles)
            for right in list(roles)[index + 1 :]
        ),
        "role_cluster_overlap_count": sum(
            len(set(roles[left]["cluster_ids"]) & set(roles[right]["cluster_ids"]))
            for index, left in enumerate(roles)
            for right in list(roles)[index + 1 :]
        ),
        "validation_outcomes_inspected": False,
        "test_outcomes_inspected": False,
    }
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({role: {"cases": row["case_count"], "clusters": row["cluster_count"]} for role, row in roles.items()}, indent=2))


if __name__ == "__main__":
    main()
