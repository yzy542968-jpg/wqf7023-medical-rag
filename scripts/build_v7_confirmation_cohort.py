from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import (  # noqa: E402
    build_audit as build_v6_audit,
    canonical_case_id,
    case_id_fingerprint,
    case_ids_from_payload,
    file_sha256,
    read_json,
    read_jsonl,
    report_index_class,
)
from medical_rag.evaluation.case_scoped_benchmark import (  # noqa: E402
    build_case_chunks,
    build_case_questions,
)
from medical_rag.multimodal.openi_images import resolve_official_image  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v7_confirmation.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_AUDIT = ROOT / "data" / "splits" / "v7" / "v7_prior_use_audit.json"
DEFAULT_DEVELOPMENT_MANIFEST = ROOT / "data" / "splits" / "v7" / "v7_development_manifest.json"
DEFAULT_V5_CONFIG = ROOT / "config" / "multimodal_v5.json"
DEFAULT_V5_COHORT = ROOT / "data" / "processed" / "openi_multimodal_v5_cohort.json"
DEFAULT_V6_DEVELOPMENT = ROOT / "data" / "splits" / "v6" / "v6_development_case_ids.txt"
DEFAULT_V6_CONFIRMATION = ROOT / "data" / "splits" / "v6" / "v6_confirmation_cohort.json"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_COHORT = ROOT / "data" / "splits" / "v7" / "v7_confirmation_cohort.json"
DEFAULT_TARGETS = ROOT / "data" / "splits" / "v7" / "v7_confirmation_target_case_ids.txt"
DEFAULT_DISTRACTORS = ROOT / "data" / "splits" / "v7" / "v7_confirmation_distractor_case_ids.txt"
DEFAULT_FREEZE = ROOT / "data" / "splits" / "v7" / "v7_confirmation_cohort_freeze.json"


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def canonical_ids(values: Iterable[Any]) -> list[str]:
    ids = [canonical_case_id(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError("Case-ID collection contains duplicates.")
    return sorted(ids)


def case_id_payload(values: Iterable[Any]) -> str:
    return "\n".join(canonical_ids(values))


def hash_order_key(domain: str, seed: int, case_id: str) -> str:
    return hashlib.sha256(
        f"{domain}|{seed}|{canonical_case_id(case_id)}".encode("utf-8")
    ).hexdigest()


def image_lookup(image_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in image_root.rglob("*.png"):
        if path.name in result:
            duplicates.add(path.name)
        result[path.name] = path
    if duplicates:
        raise RuntimeError(f"Duplicate official image names: {sorted(duplicates)[:5]}")
    return result


def verify_images(
    case_ids: Iterable[str],
    cases: Mapping[str, Mapping[str, Any]],
    lookup: Mapping[str, Path],
) -> dict[str, list[str]]:
    output: dict[str, list[str]] = {}
    for case_id in canonical_ids(case_ids):
        paths: list[str] = []
        for image in cases[case_id].get("images", []):
            path = resolve_official_image(case_id, str(image["filename"]), lookup)
            if path is None:
                continue
            with Image.open(path) as image_handle:
                image_handle.verify()
            paths.append(portable_path(path))
        if not paths:
            raise RuntimeError(f"Confirmation case {case_id} has no readable official image.")
        output[case_id] = paths
    return output


def committed_json(commit: str, path: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout.decode("utf-8"))


def read_case_ids(path: Path) -> set[str]:
    return {
        canonical_case_id(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Instantiate the frozen V7 confirmation cohort once.")
    parser.add_argument("--protocol-commit", default="4821f38")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--development-manifest", type=Path, default=DEFAULT_DEVELOPMENT_MANIFEST)
    parser.add_argument("--v5-config", type=Path, default=DEFAULT_V5_CONFIG)
    parser.add_argument("--v5-cohort", type=Path, default=DEFAULT_V5_COHORT)
    parser.add_argument("--v6-development", type=Path, default=DEFAULT_V6_DEVELOPMENT)
    parser.add_argument("--v6-confirmation", type=Path, default=DEFAULT_V6_CONFIRMATION)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--cohort-output", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--target-output", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--distractor-output", type=Path, default=DEFAULT_DISTRACTORS)
    parser.add_argument("--freeze-output", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()

    config = read_json(args.config)
    committed_config = committed_json(args.protocol_commit, args.config)
    if config != committed_config:
        raise RuntimeError("Current V7 confirmation config differs from committed protocol config.")
    if config["cohort_generation"]["case_ids_instantiated"] is not False:
        raise RuntimeError("V7 confirmation config already contains instantiated IDs.")
    if file_sha256(args.cases) != config["source"]["cases_sha256"]:
        raise RuntimeError("Source case JSONL differs from frozen V7 confirmation config.")

    audit = read_json(args.audit)
    development_manifest = read_json(args.development_manifest)
    if audit["status"] != "development_manifest_instantiated_confirmation_ids_not_generated":
        raise RuntimeError("V7 prior-use audit status is not the expected pre-confirmation state.")
    if development_manifest["confirmation_case_ids_instantiated"] is not False:
        raise RuntimeError("Development manifest unexpectedly contains confirmation IDs.")

    all_cases = read_jsonl(args.cases)
    cases = {canonical_case_id(case["case_id"]): case for case in all_cases}
    v5_config = read_json(args.v5_config)
    v5_cohort = read_json(args.v5_cohort)
    legacy_ids: set[str] = set()
    for relative in v5_config["cohort"]["excluded_source_manifests"]:
        legacy_ids.update(case_ids_from_payload(read_json(ROOT / relative)))
    v5_ids = {canonical_case_id(value) for value in v5_cohort["case_ids"]}
    v6_development_ids = read_case_ids(args.v6_development)
    v6_confirmation_ids = {
        canonical_case_id(value) for value in read_json(args.v6_confirmation)["case_ids"]
    }
    v5_development_ids = {
        canonical_case_id(value)
        for value in v5_cohort["split"]["development"]["case_ids"]
    }
    _, v6_collections = build_v6_audit(
        all_cases=all_cases,
        prior_excluded_ids=legacy_ids,
        v5_case_ids=v5_ids,
        development_ids=v5_development_ids,
    )
    pre_confirmation_ids = set(v6_collections["eligible"]) - v6_confirmation_ids - v6_development_ids
    development_ids = {
        canonical_case_id(value)
        for block in development_manifest["blocks"].values()
        for value in block["case_ids"]
    }
    frame_ids = pre_confirmation_ids - development_ids
    if len(frame_ids) != 519:
        raise RuntimeError(f"Post-development frame changed: expected 519, found {len(frame_ids)}.")

    normal_ids = {
        case_id for case_id in frame_ids if report_index_class(cases[case_id]) == "report_indexed_normal"
    }
    abnormal_ids = {
        case_id for case_id in frame_ids if report_index_class(cases[case_id]) == "report_indexed_abnormal"
    }
    indeterminate_ids = {
        case_id for case_id in frame_ids if report_index_class(cases[case_id]) == "report_index_indeterminate"
    }
    expected_frame = config["source"]
    if case_id_fingerprint(frame_ids) != expected_frame["pre_confirmation_frame_ids_sha256"]:
        raise RuntimeError("Post-development confirmation frame fingerprint changed.")
    if len(normal_ids) != 357 or len(abnormal_ids) != 145 or len(indeterminate_ids) != 17:
        raise RuntimeError("Post-development report-index composition changed.")

    rule = config["cohort_generation"]
    selected_normal = sorted(
        normal_ids,
        key=lambda case_id: (hash_order_key(rule["selection_domain"], int(rule["seed"]), case_id), case_id),
    )[:172]
    selected_abnormal = sorted(
        abnormal_ids,
        key=lambda case_id: (hash_order_key(rule["selection_domain"], int(rule["seed"]), case_id), case_id),
    )[:68]
    selected = set(selected_normal) | set(selected_abnormal)
    if len(selected) != 240:
        raise RuntimeError("V7 selection did not produce 240 unique cases.")
    normal_assignment = sorted(
        selected_normal,
        key=lambda case_id: (hash_order_key(rule["assignment_domain"], int(rule["seed"]), case_id), case_id),
    )
    abnormal_assignment = sorted(
        selected_abnormal,
        key=lambda case_id: (hash_order_key(rule["assignment_domain"], int(rule["seed"]), case_id), case_id),
    )
    targets = set(normal_assignment[:86] + abnormal_assignment[:34])
    distractors = set(normal_assignment[86:] + abnormal_assignment[34:])
    if len(targets) != 120 or len(distractors) != 120 or targets & distractors:
        raise RuntimeError("V7 target/distractor assignment is invalid.")
    if selected & development_ids or selected & (legacy_ids | v5_ids | v6_confirmation_ids | v6_development_ids):
        raise RuntimeError("V7 confirmation selection overlaps a prior or development case.")

    selected_images = verify_images(selected, cases, image_lookup(args.image_root))
    questions: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for case_id in sorted(targets):
        case_chunks = build_case_chunks(cases[case_id])
        chunks.extend(case_chunks)
        questions.extend(build_case_questions(cases[case_id], case_chunks))
    if len(questions) != 360 or len({str(row["case_id"]) for row in questions}) != 120:
        raise RuntimeError("V7 confirmation target cases did not produce 360 questions.")

    role_by_id = {
        **{case_id: "target" for case_id in targets},
        **{case_id: "distractor" for case_id in distractors},
    }
    manifest_rows = [
        {
            "case_id": case_id,
            "role": role_by_id[case_id],
            "report_index_class": report_index_class(cases[case_id]),
            "image_paths": selected_images[case_id],
        }
        for case_id in sorted(selected)
    ]
    cohort = {
        "benchmark": "OpenI V7 adaptive multimodal fusion confirmation cohort",
        "version": "7.0-confirmation",
        "status": "instantiated_after_confirmation_protocol_commit_before_outcomes",
        "protocol_commit": args.protocol_commit,
        "config_path": portable_path(args.config),
        "config_sha256": file_sha256(args.config),
        "source_cases": portable_path(args.cases),
        "source_cases_sha256": file_sha256(args.cases),
        "case_count": 240,
        "case_ids": sorted(selected),
        "target_case_ids": sorted(targets),
        "distractor_case_ids": sorted(distractors),
        "cases": manifest_rows,
        "question_count": len(questions),
        "questions": questions,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }
    args.cohort_output.parent.mkdir(parents=True, exist_ok=True)
    args.cohort_output.write_text(
        json.dumps(cohort, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.target_output.write_text(case_id_payload(targets), encoding="utf-8", newline="")
    args.distractor_output.write_text(case_id_payload(distractors), encoding="utf-8", newline="")

    fingerprints = {
        "selected_240_case_ids_sha256": case_id_fingerprint(selected),
        "target_120_case_ids_sha256": case_id_fingerprint(targets),
        "distractor_120_case_ids_sha256": case_id_fingerprint(distractors),
        "selected_normal_172_case_ids_sha256": case_id_fingerprint(selected_normal),
        "selected_abnormal_68_case_ids_sha256": case_id_fingerprint(selected_abnormal),
        "target_normal_86_case_ids_sha256": case_id_fingerprint(set(normal_assignment[:86])),
        "target_abnormal_34_case_ids_sha256": case_id_fingerprint(set(abnormal_assignment[:34])),
        "distractor_normal_86_case_ids_sha256": case_id_fingerprint(set(normal_assignment[86:])),
        "distractor_abnormal_34_case_ids_sha256": case_id_fingerprint(set(abnormal_assignment[34:])),
    }
    freeze = {
        "freeze": "V7 instantiated confirmation cohort",
        "status": "frozen_before_confirmation_outcome_execution",
        "protocol_commit": args.protocol_commit,
        "builder": portable_path(Path(__file__)),
        "builder_sha256": file_sha256(Path(__file__)),
        "config": portable_path(args.config),
        "config_sha256": file_sha256(args.config),
        "source_cases_sha256": file_sha256(args.cases),
        "pre_confirmation_frame": config["source"],
        "cohort_generation": config["cohort_generation"],
        "observed_composition": {
            "selected": {"normal": 172, "abnormal": 68, "total": 240},
            "target": {"normal": 86, "abnormal": 34, "total": 120},
            "distractor": {"normal": 86, "abnormal": 34, "total": 120},
        },
        "readability_check": {
            "case_count": len(selected_images),
            "image_count": sum(len(paths) for paths in selected_images.values()),
            "all_selected_cases_have_readable_official_image": True,
        },
        "question_count": len(questions),
        "chunk_count": len(chunks),
        "fingerprints": fingerprints,
        "artifacts": {
            "cohort": portable_path(args.cohort_output),
            "cohort_sha256": file_sha256(args.cohort_output),
            "targets": portable_path(args.target_output),
            "targets_sha256": file_sha256(args.target_output),
            "distractors": portable_path(args.distractor_output),
            "distractors_sha256": file_sha256(args.distractor_output),
        },
        "claim_boundary": "Case-ID disjoint same-source confirmation; patient-level independence and external validation are not established.",
    }
    args.freeze_output.write_text(
        json.dumps(freeze, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(freeze, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
