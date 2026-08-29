from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.qa.radrestruct import RadReStructCase, iter_radrestruct_cases


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_ids(values: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _placeholder_ratio(text: object) -> float:
    tokens = str(text or "").split()
    if not tokens:
        return 1.0
    return sum("XXXX" in token.upper() for token in tokens) / len(tokens)


def _image_suffix(value: object) -> str:
    name = Path(str(value)).name
    for suffix in (".dcm.png", ".png", ".dcm"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    marker = name.upper().find("IM-")
    return name[marker:].upper() if marker >= 0 else name.upper()


def _git_commit(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _display_path(path: Path) -> str:
    for base in (ROOT, ROOT.parent):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.name


def _partition_maps(split_manifest: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    partition_by_case: dict[str, str] = {}
    for partition, payload in split_manifest["partitions"].items():
        for case_id in payload["case_ids"]:
            if case_id in partition_by_case:
                raise ValueError(f"Case appears in multiple V10 partitions: {case_id}")
            partition_by_case[case_id] = partition
    cluster_by_case: dict[str, str] = {}
    for cluster in split_manifest["clusters"]:
        cluster_id = str(cluster["cluster_id"])
        for case_id in cluster["case_ids"]:
            if case_id in cluster_by_case:
                raise ValueError(f"Case appears in multiple V10 clusters: {case_id}")
            cluster_by_case[case_id] = cluster_id
    return partition_by_case, cluster_by_case


def _question_summary(cases: list[RadReStructCase]) -> dict[str, Any]:
    answer_types: Counter[str] = Counter()
    answers: Counter[str] = Counter()
    paths: Counter[str] = Counter()
    path_roots: Counter[str] = Counter()
    question_texts: Counter[str] = Counter()
    option_counts: Counter[str] = Counter()
    question_counts: list[int] = []
    empty_answers = 0
    multi_answer_rows = 0
    history_rows = 0
    yes_no_rows = 0
    yes_no_answers: Counter[str] = Counter()
    for case in cases:
        question_counts.append(len(case.questions))
        for question in case.questions:
            answer_types[question.answer_type or "<missing>"] += 1
            question_texts[question.question] += 1
            paths[question.path or "<missing>"] += 1
            path_roots[(question.path.split("_", 1)[0] or "<missing>")] += 1
            option_counts[str(len(question.options))] += 1
            if not question.answers:
                empty_answers += 1
            if len(question.answers) > 1:
                multi_answer_rows += 1
            if question.history:
                history_rows += 1
            normalized_answers = [answer.strip().lower() for answer in question.answers]
            answers.update(normalized_answers)
            normalized_options = {option.strip().lower() for option in question.options}
            if {"yes", "no"}.issubset(normalized_options):
                yes_no_rows += 1
                yes_no_answers.update(normalized_answers)
    total_questions = sum(question_counts)
    return {
        "total_questions": total_questions,
        "unique_question_texts": len(question_texts),
        "unique_paths": len(paths),
        "questions_per_case": {
            "minimum": min(question_counts, default=0),
            "median": statistics.median(question_counts) if question_counts else 0,
            "mean": statistics.fmean(question_counts) if question_counts else 0,
            "maximum": max(question_counts, default=0),
        },
        "answer_type_counts": dict(answer_types.most_common()),
        "option_count_distribution": dict(sorted(option_counts.items())),
        "empty_answer_rows": empty_answers,
        "multi_answer_rows": multi_answer_rows,
        "rows_with_history": history_rows,
        "yes_no_option_rows": yes_no_rows,
        "yes_no_answer_counts": dict(yes_no_answers.most_common()),
        "top_answers": dict(answers.most_common(30)),
        "top_path_roots": dict(path_roots.most_common(30)),
        "top_question_texts": dict(question_texts.most_common(30)),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    openi_path = args.openi_cases.resolve()
    rad_root = args.radrestruct_root.resolve()
    split_path = args.v10_split.resolve()
    openi_rows = _load_jsonl(openi_path)
    openi_by_id = {str(row["case_id"]): row for row in openi_rows}
    if len(openi_by_id) != len(openi_rows):
        raise ValueError("OpenI source contains duplicate case IDs")

    split_manifest = _load_json(split_path)
    partition_by_case, cluster_by_case = _partition_maps(split_manifest)
    rad_cases = list(iter_radrestruct_cases(rad_root))

    official_split_counts = Counter(case.official_split for case in rad_cases)
    matched = [case for case in rad_cases if case.case_id in openi_by_id]
    unmatched = [case.case_id for case in rad_cases if case.case_id not in openi_by_id]
    v10_partition_counts: Counter[str] = Counter()
    official_v10_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    clusters_to_official_splits: dict[str, set[str]] = defaultdict(set)
    mapping_rows: list[dict[str, Any]] = []
    rad_image_count = 0
    matched_image_count = 0
    frontal_mismatch_count = 0

    for case in matched:
        source = openi_by_id[case.case_id]
        partition = partition_by_case.get(case.case_id, "<missing>")
        cluster = cluster_by_case.get(case.case_id, "<missing>")
        v10_partition_counts[partition] += 1
        official_v10_matrix[case.official_split][partition] += 1
        clusters_to_official_splits[cluster].add(case.official_split)

        local_images = source.get("images", [])
        local_suffixes = {_image_suffix(image.get("filename", "")) for image in local_images}
        local_frontal = {
            _image_suffix(image.get("filename", ""))
            for image in local_images
            if str(image.get("projection", "")).strip().lower() == "frontal"
        }
        rad_suffixes = {_image_suffix(image_id) for image_id in case.image_ids}
        matched_images = rad_suffixes & local_suffixes
        rad_image_count += len(rad_suffixes)
        matched_image_count += len(matched_images)
        frontal_mismatch_count += len(matched_images - local_frontal)
        indication = str(source.get("indication", "")).strip()
        mapping_rows.append(
            {
                "source_report_id": case.source_report_id,
                "case_id": case.case_id,
                "official_split": case.official_split,
                "v10_partition": partition,
                "v10_cluster_id": cluster,
                "question_count": len(case.questions),
                "radrestruct_image_count": len(rad_suffixes),
                "matched_local_image_count": len(matched_images),
                "indication_available": bool(indication),
                "indication_placeholder_ratio": round(_placeholder_ratio(indication), 6),
                "findings_available": bool(str(source.get("findings", "")).strip()),
                "impression_available": bool(str(source.get("impression", "")).strip()),
            }
        )

    cross_official_clusters = {
        cluster: sorted(splits)
        for cluster, splits in clusters_to_official_splits.items()
        if cluster != "<missing>" and len(splits) > 1
    }
    mapped_case_ids = [case.case_id for case in matched]
    question_distribution_by_v10_partition: dict[str, Any] = {}
    for partition in ("train", "calibration", "validation", "test"):
        partition_cases = [
            case for case in matched if partition_by_case.get(case.case_id) == partition
        ]
        partition_questions = _question_summary(partition_cases)
        question_distribution_by_v10_partition[partition] = {
            "case_count": len(partition_cases),
            "total_questions": partition_questions["total_questions"],
            "answer_type_counts": partition_questions["answer_type_counts"],
            "yes_no_option_rows": partition_questions["yes_no_option_rows"],
            "yes_no_answer_counts": partition_questions["yes_no_answer_counts"],
        }
    summary = {
        "study": "Final QA feasibility audit",
        "status": "data_mapping_audit_only",
        "sources": {
            "openi_cases_path": _display_path(openi_path),
            "openi_cases_sha256": _sha256_file(openi_path),
            "openi_case_count": len(openi_rows),
            "v10_split_path": _display_path(split_path),
            "v10_split_sha256": _sha256_file(split_path),
            "radrestruct_root": _display_path(rad_root),
            "radrestruct_commit": _git_commit(rad_root.parent.parent),
        },
        "mapping": {
            "radrestruct_case_count": len(rad_cases),
            "official_split_counts": dict(official_split_counts),
            "matched_openi_case_count": len(matched),
            "unmatched_openi_case_count": len(unmatched),
            "unmatched_case_ids": sorted(unmatched),
            "mapping_rate": len(matched) / len(rad_cases) if rad_cases else 0.0,
            "mapped_case_ids_sha256": _sha256_ids(mapped_case_ids),
            "v10_partition_counts": dict(v10_partition_counts),
            "official_split_by_v10_partition": {
                split: dict(counts) for split, counts in official_v10_matrix.items()
            },
            "v10_cluster_count": len(
                {cluster_by_case[case_id] for case_id in mapped_case_ids}
            ),
            "official_split_crossing_v10_cluster_count": len(cross_official_clusters),
            "official_split_crossing_v10_clusters": cross_official_clusters,
        },
        "images": {
            "radrestruct_frontal_image_references": rad_image_count,
            "matched_local_image_references": matched_image_count,
            "image_reference_match_rate": (
                matched_image_count / rad_image_count if rad_image_count else 0.0
            ),
            "matched_images_not_marked_frontal_locally": frontal_mismatch_count,
        },
        "questions": _question_summary(matched),
        "question_distribution_by_v10_partition": question_distribution_by_v10_partition,
        "openi_fields_on_mapped_cases": {
            "indication_nonempty": sum(row["indication_available"] for row in mapping_rows),
            "indication_placeholder_ratio_le_0_5": sum(
                row["indication_placeholder_ratio"] <= 0.5 for row in mapping_rows
            ),
            "findings_nonempty": sum(row["findings_available"] for row in mapping_rows),
            "impression_nonempty": sum(row["impression_available"] for row in mapping_rows),
        },
        "audit_decisions": {
            "radrestruct_replaces_openi": False,
            "use_v10_cluster_disjoint_partitioning": True,
            "use_radrestruct_official_split_for_final_confirmation": False,
            "reason": (
                "Rad-ReStruct supplies structured QA labels on IU-Xray/OpenI. "
                "The existing V10 partition is retained to prevent exact/near-duplicate "
                "clusters from crossing development and confirmation roles."
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.mapping_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.mapping_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping_rows[0]) if mapping_rows else [])
        if mapping_rows:
            writer.writeheader()
            writer.writerows(mapping_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--openi-cases",
        type=Path,
        default=ROOT / "data/processed/openi_cases.jsonl",
    )
    parser.add_argument(
        "--radrestruct-root",
        type=Path,
        required=True,
        help="Path to the official repository's data/radrestruct directory.",
    )
    parser.add_argument(
        "--v10-split",
        type=Path,
        default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_feasibility_audit.json",
    )
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_case_mapping.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = audit(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
