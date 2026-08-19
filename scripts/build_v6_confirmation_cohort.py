from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import (  # noqa: E402
    build_audit,
    canonical_case_id,
    case_id_fingerprint,
    case_id_payload,
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


DEFAULT_CONFIG = ROOT / "config" / "v6_confirmation.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_V5_CONFIG = ROOT / "config" / "multimodal_v5.json"
DEFAULT_V5_COHORT = ROOT / "data" / "processed" / "openi_multimodal_v5_cohort.json"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_COHORT = ROOT / "data" / "splits" / "v6" / "v6_confirmation_cohort.json"
DEFAULT_TARGETS = ROOT / "data" / "splits" / "v6" / "v6_confirmation_target_case_ids.txt"
DEFAULT_DISTRACTORS = ROOT / "data" / "splits" / "v6" / "v6_confirmation_distractor_case_ids.txt"
DEFAULT_FREEZE = ROOT / "data" / "splits" / "v6" / "v6_confirmation_cohort_freeze.json"


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def hash_order_key(domain: str, seed: int, case_id: str) -> str:
    payload = f"{domain}|{seed}|{canonical_case_id(case_id)}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_and_assign(
    normal_ids: Iterable[str],
    abnormal_ids: Iterable[str],
    *,
    seed: int,
    selection_domain: str,
    assignment_domain: str,
) -> dict[str, list[str]]:
    normal = sorted(
        [canonical_case_id(value) for value in normal_ids],
        key=lambda case_id: (hash_order_key(selection_domain, seed, case_id), case_id),
    )[:172]
    abnormal = sorted(
        [canonical_case_id(value) for value in abnormal_ids],
        key=lambda case_id: (hash_order_key(selection_domain, seed, case_id), case_id),
    )[:68]
    if len(normal) != 172 or len(abnormal) != 68:
        raise RuntimeError("The frozen strata do not contain enough cases.")

    normal_assignment = sorted(
        normal,
        key=lambda case_id: (hash_order_key(assignment_domain, seed, case_id), case_id),
    )
    abnormal_assignment = sorted(
        abnormal,
        key=lambda case_id: (hash_order_key(assignment_domain, seed, case_id), case_id),
    )
    targets = normal_assignment[:86] + abnormal_assignment[:34]
    distractors = normal_assignment[86:] + abnormal_assignment[34:]
    selected = normal + abnormal
    if set(targets) & set(distractors):
        raise RuntimeError("Target and distractor assignments overlap.")
    if set(targets) | set(distractors) != set(selected):
        raise RuntimeError("Target and distractor assignments do not cover the selection.")
    return {
        "selected": sorted(selected),
        "selected_normal": sorted(normal),
        "selected_abnormal": sorted(abnormal),
        "targets": sorted(targets),
        "target_normal": sorted(normal_assignment[:86]),
        "target_abnormal": sorted(abnormal_assignment[:34]),
        "distractors": sorted(distractors),
        "distractor_normal": sorted(normal_assignment[86:]),
        "distractor_abnormal": sorted(abnormal_assignment[34:]),
    }


def committed_json(commit: str, path: Path) -> dict[str, Any]:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return json.loads(result.stdout.decode("utf-8"))


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


def verify_selected_images(
    selected_ids: Iterable[str],
    cases: dict[str, dict[str, Any]],
    images: dict[str, Path],
) -> dict[str, list[str]]:
    selected_images: dict[str, list[str]] = {}
    for case_id in selected_ids:
        paths = []
        for row in cases[case_id].get("images", []):
            path = resolve_official_image(case_id, str(row["filename"]), images)
            if path is None:
                continue
            with Image.open(path) as image:
                image.verify()
            paths.append(portable_path(path))
        if not paths:
            raise RuntimeError(f"Selected case {case_id} has no readable official image.")
        selected_images[case_id] = paths
    return selected_images


def assert_frame_matches_config(config: dict[str, Any], audit: dict[str, Any]) -> None:
    frame = audit["confirmation_selection_frame"]
    expected = config["selection_frame"]
    pairs = {
        "v6_eligible_case_count": "eligible_case_count",
        "v6_eligible_case_ids_sha256": "eligible_case_ids_sha256",
        "v6_stratifiable_case_count": "stratifiable_case_count",
        "v6_stratifiable_case_ids_sha256": "stratifiable_case_ids_sha256",
        "report_indexed_normal_case_count": "report_indexed_normal_count",
        "report_indexed_normal_case_ids_sha256": "report_indexed_normal_ids_sha256",
        "report_indexed_abnormal_case_count": "report_indexed_abnormal_count",
        "report_indexed_abnormal_case_ids_sha256": "report_indexed_abnormal_ids_sha256",
        "report_index_indeterminate_case_count": "report_index_indeterminate_count",
        "report_index_indeterminate_case_ids_sha256": "report_index_indeterminate_ids_sha256",
    }
    for audit_key, config_key in pairs.items():
        if frame[audit_key] != expected[config_key]:
            raise RuntimeError(
                f"Selection frame differs from frozen config: {audit_key}={frame[audit_key]!r}, "
                f"expected {expected[config_key]!r}."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Instantiate the frozen V6 confirmation cohort once.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol-commit", default="eee7405")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--v5-config", type=Path, default=DEFAULT_V5_CONFIG)
    parser.add_argument("--v5-cohort", type=Path, default=DEFAULT_V5_COHORT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--cohort-output", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--target-output", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--distractor-output", type=Path, default=DEFAULT_DISTRACTORS)
    parser.add_argument("--freeze-output", type=Path, default=DEFAULT_FREEZE)
    args = parser.parse_args()

    config = read_json(args.config)
    if config != committed_json(args.protocol_commit, args.config):
        raise RuntimeError("Current confirmation config differs from the committed protocol config.")
    if config["cohort_generation"]["case_ids_instantiated"] is not False:
        raise RuntimeError("The frozen protocol config must predate cohort instantiation.")
    if file_sha256(args.cases) != config["source"]["cases_sha256"]:
        raise RuntimeError("Source case JSONL differs from the frozen confirmation config.")

    all_cases = read_jsonl(args.cases)
    cases = {canonical_case_id(case["case_id"]): case for case in all_cases}
    v5_config = read_json(args.v5_config)
    v5_cohort = read_json(args.v5_cohort)
    prior_excluded: set[str] = set()
    for relative in v5_config["cohort"]["excluded_source_manifests"]:
        prior_excluded.update(case_ids_from_payload(read_json(ROOT / relative)))
    v5_ids = {canonical_case_id(value) for value in v5_cohort["case_ids"]}
    development_ids = {
        canonical_case_id(value)
        for value in v5_cohort["split"]["development"]["case_ids"]
    }
    audit, collections = build_audit(
        all_cases=all_cases,
        prior_excluded_ids=prior_excluded,
        v5_case_ids=v5_ids,
        development_ids=development_ids,
    )
    assert_frame_matches_config(config, audit)

    rule = config["cohort_generation"]
    selection = select_and_assign(
        collections["report_indexed_normal"],
        collections["report_indexed_abnormal"],
        seed=int(rule["seed"]),
        selection_domain=str(rule["selection_domain"]),
        assignment_domain=str(rule["assignment_domain"]),
    )
    selected_images = verify_selected_images(
        selection["selected"], cases, image_lookup(args.image_root)
    )

    target_set = set(selection["targets"])
    questions: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    for case_id in selection["targets"]:
        case_chunks = build_case_chunks(cases[case_id])
        chunks.extend(case_chunks)
        questions.extend(build_case_questions(cases[case_id], case_chunks))
    if len(questions) != 360 or len({str(row["case_id"]) for row in questions}) != 120:
        raise RuntimeError("The frozen target cases did not produce 360 questions.")
    if {str(row["case_id"]) for row in questions} != target_set:
        raise RuntimeError("Question cases differ from frozen target assignment.")

    role_by_id = {
        **{case_id: "target" for case_id in selection["targets"]},
        **{case_id: "distractor" for case_id in selection["distractors"]},
    }
    manifest_rows = [
        {
            "case_id": case_id,
            "role": role_by_id[case_id],
            "report_index_class": report_index_class(cases[case_id]),
            "image_paths": selected_images[case_id],
        }
        for case_id in selection["selected"]
    ]
    cohort = {
        "benchmark": "OpenI V6 model-modernized confirmation cohort",
        "version": "6.0",
        "status": "instantiated_after_confirmation_protocol_commit_before_outcomes",
        "protocol_commit": args.protocol_commit,
        "config_path": portable_path(args.config),
        "config_sha256": file_sha256(args.config),
        "source_cases": portable_path(args.cases),
        "source_cases_sha256": file_sha256(args.cases),
        "case_count": len(selection["selected"]),
        "case_ids": selection["selected"],
        "target_case_ids": selection["targets"],
        "distractor_case_ids": selection["distractors"],
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
    args.target_output.write_text(
        case_id_payload(selection["targets"]), encoding="utf-8", newline=""
    )
    args.distractor_output.write_text(
        case_id_payload(selection["distractors"]), encoding="utf-8", newline=""
    )

    fingerprints = {
        "selected_240_case_ids_sha256": case_id_fingerprint(selection["selected"]),
        "target_120_case_ids_sha256": case_id_fingerprint(selection["targets"]),
        "distractor_120_case_ids_sha256": case_id_fingerprint(selection["distractors"]),
        "selected_normal_172_case_ids_sha256": case_id_fingerprint(selection["selected_normal"]),
        "selected_abnormal_68_case_ids_sha256": case_id_fingerprint(selection["selected_abnormal"]),
        "target_normal_86_case_ids_sha256": case_id_fingerprint(selection["target_normal"]),
        "target_abnormal_34_case_ids_sha256": case_id_fingerprint(selection["target_abnormal"]),
        "distractor_normal_86_case_ids_sha256": case_id_fingerprint(selection["distractor_normal"]),
        "distractor_abnormal_34_case_ids_sha256": case_id_fingerprint(selection["distractor_abnormal"]),
    }
    freeze = {
        "freeze": "V6 instantiated confirmation cohort",
        "status": "frozen_before_confirmation_outcome_execution",
        "protocol_commit": args.protocol_commit,
        "builder": portable_path(Path(__file__)),
        "builder_sha256": file_sha256(Path(__file__)),
        "config": portable_path(args.config),
        "config_sha256": file_sha256(args.config),
        "source_cases_sha256": file_sha256(args.cases),
        "selection_frame": config["selection_frame"],
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
        "claim_boundary": (
            "Case-ID disjoint same-source confirmation; patient-level independence "
            "and external validation are not established."
        ),
    }
    args.freeze_output.write_text(
        json.dumps(freeze, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(freeze, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
