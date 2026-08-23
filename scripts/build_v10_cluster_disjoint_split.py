from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.multimodal.openi_images import official_filename_candidates  # noqa: E402
from medical_rag.similar_case.v10_split import (  # noqa: E402
    PARTITION_ORDER,
    assign_clusters,
    build_duplicate_clusters,
    canonical_fingerprint,
    file_sha256,
    normalized_report_text,
    report_index_spectrum,
)


DEFAULT_PROTOCOL_COMMIT = "79e0b45"
DEFAULT_CONFIG = ROOT / "config" / "v10_development_protocol.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_IMAGES = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_MANIFEST = ROOT / "data" / "splits" / "v10" / "v10_cluster_disjoint_split.json"
DEFAULT_FREEZE = ROOT / "data" / "splits" / "v10" / "v10_cluster_disjoint_split_freeze.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def committed_bytes(commit: str, path: Path) -> bytes:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return result.stdout


def resolve_image(case_id: str, filename: str, image_root: Path) -> Path:
    for candidate in official_filename_candidates(case_id, filename):
        path = image_root / candidate
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not resolve {case_id} image {filename}")


def image_hashes(cases: list[dict[str, Any]], image_root: Path) -> dict[str, list[str]]:
    cache: dict[Path, str] = {}
    result: dict[str, list[str]] = {}
    for case in cases:
        case_id = str(case["case_id"]).strip()
        values = []
        for image in case.get("images") or []:
            path = resolve_image(case_id, str(image.get("filename", "")), image_root)
            if path not in cache:
                cache[path] = file_sha256(path)
            values.append(cache[path])
        result[case_id] = sorted(set(values))
    return result


def portable(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen V10 cluster-disjoint split.")
    parser.add_argument("--protocol-commit", default=DEFAULT_PROTOCOL_COMMIT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--freeze-output", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()

    current_config = args.config.read_bytes()
    if current_config != committed_bytes(args.protocol_commit, args.config):
        raise RuntimeError("Current V10 config differs from the committed protocol config")
    config = json.loads(current_config.decode("utf-8"))
    if config["test_outcomes_inspected"] is not False:
        raise RuntimeError("V10 protocol unexpectedly records inspected Test outcomes")

    cases = read_jsonl(args.cases)
    case_ids = [str(case["case_id"]).strip() for case in cases]
    if len(case_ids) != 3851 or len(case_ids) != len(set(case_ids)):
        raise RuntimeError("Expected 3,851 unique OpenI case IDs")
    texts = [
        normalized_report_text(case.get("findings"), case.get("impression"))
        for case in cases
    ]
    spectrum = {
        case_id: report_index_spectrum(case.get("problems"))
        for case_id, case in zip(case_ids, cases, strict=True)
    }
    images = image_hashes(cases, args.image_root)

    duplicate_config = config["duplicate_clustering"]
    clustered = build_duplicate_clusters(
        case_ids,
        texts,
        cosine_threshold=float(duplicate_config["tfidf_cosine_threshold"]),
        image_sha256_by_case=images,
    )
    assignments = assign_clusters(
        clustered.clusters,
        spectrum,
        config["partition_fractions"],
        domain=config["partition_domain"],
        seed=int(config["seed"]),
    )

    cluster_by_case = {
        case_id: cluster[0]
        for cluster in clustered.clusters
        for case_id in cluster
    }
    if len(cluster_by_case) != len(case_ids):
        raise RuntimeError("Cluster manifest does not cover the source universe")
    partition_clusters = {
        partition: {cluster_by_case[case_id] for case_id in values}
        for partition, values in assignments.items()
    }
    for left in PARTITION_ORDER:
        for right in PARTITION_ORDER:
            if left != right and partition_clusters[left] & partition_clusters[right]:
                raise RuntimeError(f"Cluster leakage between {left} and {right}")

    partitions = {}
    for partition in PARTITION_ORDER:
        values = assignments[partition]
        counts = Counter(spectrum[case_id] for case_id in values)
        partitions[partition] = {
            "case_count": len(values),
            "cluster_count": len(partition_clusters[partition]),
            "normal": counts["normal"],
            "abnormal": counts["abnormal"],
            "indeterminate": counts["indeterminate"],
            "case_ids_sha256": canonical_fingerprint(values),
            "cluster_ids_sha256": canonical_fingerprint(partition_clusters[partition]),
            "case_ids": values,
        }

    cluster_rows = [
        {
            "cluster_id": cluster[0],
            "size": len(cluster),
            "case_ids": list(cluster),
            "case_ids_sha256": canonical_fingerprint(cluster),
        }
        for cluster in clustered.clusters
    ]
    manifest = {
        "study": "V10 cluster-disjoint OpenI publication extension",
        "status": "instantiated_after_protocol_commit_before_v10_outcomes",
        "protocol_commit": args.protocol_commit,
        "config_path": portable(args.config),
        "config_sha256": file_sha256(args.config),
        "source_path": portable(args.cases),
        "source_sha256": file_sha256(args.cases),
        "image_root": portable(args.image_root),
        "source_case_count": len(case_ids),
        "source_case_ids_sha256": canonical_fingerprint(case_ids),
        "cluster_count": len(cluster_rows),
        "multi_case_cluster_count": sum(row["size"] > 1 for row in cluster_rows),
        "largest_cluster_size": max(row["size"] for row in cluster_rows),
        "clustering_edges": {
            "exact_text": clustered.exact_text_edges,
            "near_text": clustered.near_text_edges,
            "exact_image": clustered.exact_image_edges,
        },
        "clusters": cluster_rows,
        "partitions": partitions,
        "case_overlap_counts": {
            f"{left}_{right}": len(set(assignments[left]) & set(assignments[right]))
            for index, left in enumerate(PARTITION_ORDER)
            for right in PARTITION_ORDER[index + 1 :]
        },
        "cluster_overlap_counts": {
            f"{left}_{right}": len(partition_clusters[left] & partition_clusters[right])
            for index, left in enumerate(PARTITION_ORDER)
            for right in PARTITION_ORDER[index + 1 :]
        },
        "v10_outcomes_inspected": False,
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    freeze = {
        "study": "V10 cluster-disjoint split freeze",
        "protocol_commit": args.protocol_commit,
        "builder_path": portable(Path(__file__)),
        "builder_sha256": file_sha256(Path(__file__)),
        "manifest_path": portable(args.manifest_output),
        "manifest_sha256": file_sha256(args.manifest_output),
        "partition_fingerprints": {
            partition: {
                "case_count": partitions[partition]["case_count"],
                "cluster_count": partitions[partition]["cluster_count"],
                "case_ids_sha256": partitions[partition]["case_ids_sha256"],
                "cluster_ids_sha256": partitions[partition]["cluster_ids_sha256"],
            }
            for partition in PARTITION_ORDER
        },
        "all_case_overlap_counts_zero": all(
            value == 0 for value in manifest["case_overlap_counts"].values()
        ),
        "all_cluster_overlap_counts_zero": all(
            value == 0 for value in manifest["cluster_overlap_counts"].values()
        ),
        "v10_outcomes_inspected": False,
    }
    args.freeze_output.write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "cluster_count": len(cluster_rows),
                "partitions": {
                    key: value["case_count"] for key, value in partitions.items()
                },
                "manifest_sha256": freeze["manifest_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

