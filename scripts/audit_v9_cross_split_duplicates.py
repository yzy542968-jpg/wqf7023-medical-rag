from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import file_sha256, read_json  # noqa: E402
from medical_rag.multimodal.openi_images import resolve_official_image  # noqa: E402
from run_v9_development_medsiglip import image_lookup  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v9_supplemental_validity.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v9" / "v9_full_source_split.json"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_RETRIEVAL_ROWS = (
    ROOT / "experiments" / "post_submission_v9" / "v9_retrieval_confirmation_rows.jsonl"
)
DEFAULT_CASE_OUTPUT = (
    ROOT / "data" / "splits" / "v9" / "v9_cross_split_duplicate_audit.csv"
)
DEFAULT_SUMMARY = (
    ROOT / "data" / "splits" / "v9" / "v9_cross_split_duplicate_summary.json"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalized_report(case: Mapping[str, Any]) -> str:
    text = "\n".join(
        [str(case.get("findings") or ""), str(case.get("impression") or "")]
    )
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join(text.split())


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dhash64(path: Path) -> int:
    with Image.open(path) as image:
        values = np.asarray(
            image.convert("L").resize((9, 8), Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    comparisons = values[:, 1:] > values[:, :-1]
    output = 0
    for value in comparisons.ravel():
        output = (output << 1) | int(value)
    return output


def nearest_train_reports(
    train_ids: Sequence[str],
    query_ids: Sequence[str],
    reports: Mapping[str, str],
    *,
    batch_size: int = 64,
) -> dict[str, tuple[str, float]]:
    nonempty_train = [case_id for case_id in train_ids if reports.get(case_id)]
    nonempty_query = [case_id for case_id in query_ids if reports.get(case_id)]
    corpus = [reports[case_id] for case_id in nonempty_train + nonempty_query]
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(corpus)
    train_matrix = matrix[: len(nonempty_train)]
    query_matrix = matrix[len(nonempty_train) :]
    nearest: dict[str, tuple[str, float]] = {}
    for start in range(0, len(nonempty_query), batch_size):
        stop = min(start + batch_size, len(nonempty_query))
        similarities = (query_matrix[start:stop] @ train_matrix.T).toarray()
        indices = np.argmax(similarities, axis=1)
        values = similarities[np.arange(stop - start), indices]
        for case_id, index, value in zip(
            nonempty_query[start:stop], indices, values, strict=True
        ):
            nearest[case_id] = (nonempty_train[int(index)], float(value))
    return nearest


def case_image_hashes(
    case_ids: Sequence[str],
    cases: Mapping[str, Mapping[str, Any]],
    image_root: Path,
) -> dict[str, list[int]]:
    lookup = image_lookup(image_root)
    output: dict[str, list[int]] = {}
    for index, case_id in enumerate(case_ids, start=1):
        hashes: list[int] = []
        for image in cases[case_id].get("images", []):
            path = resolve_official_image(case_id, str(image.get("filename", "")), lookup)
            if path is not None:
                hashes.append(dhash64(path))
        output[case_id] = hashes
        if index % 500 == 0 or index == len(case_ids):
            print(f"image_hash_cases={index}/{len(case_ids)}", flush=True)
    return output


def nearest_train_images(
    train_ids: Sequence[str],
    query_ids: Sequence[str],
    hashes: Mapping[str, Sequence[int]],
) -> dict[str, tuple[str, int]]:
    train_views = [
        (case_id, value)
        for case_id in train_ids
        for value in hashes.get(case_id, [])
    ]
    nearest: dict[str, tuple[str, int]] = {}
    for case_id in query_ids:
        query_hashes = hashes.get(case_id, [])
        if not query_hashes:
            continue
        best_case = ""
        best_distance = 65
        for train_case_id, train_hash in train_views:
            distance = min((query_hash ^ train_hash).bit_count() for query_hash in query_hashes)
            if distance < best_distance or (
                distance == best_distance and train_case_id < best_case
            ):
                best_case = train_case_id
                best_distance = distance
        nearest[case_id] = (best_case, best_distance)
    return nearest


def aggregate_retrieval(
    rows: Sequence[Mapping[str, Any]], case_ids: set[str]
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["case_id"]) in case_ids:
            grouped[str(row["system"])].append(row)
    metrics = ("ndcg@1", "ndcg@5", "ndcg@10", "recall@1", "recall@5", "recall@10", "mrr")
    return {
        system: {
            "case_count": len({str(row["case_id"]) for row in system_rows}),
            "question_count": len(system_rows),
            **{
                metric: statistics.fmean(float(row[metric]) for row in system_rows)
                for metric in metrics
            },
        }
        for system, system_rows in sorted(grouped.items())
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit V9 cross-split near duplicates.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--retrieval-rows", type=Path, default=DEFAULT_RETRIEVAL_ROWS)
    parser.add_argument("--case-output", type=Path, default=DEFAULT_CASE_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    split = read_json(args.split)
    cases_list = read_jsonl(args.cases)
    cases = {str(case["case_id"]): case for case in cases_list}
    reports = {case_id: normalized_report(case) for case_id, case in cases.items()}
    train_ids = sorted(map(str, split["partitions"]["train"]["case_ids"]))
    validation_ids = sorted(map(str, split["partitions"]["validation"]["case_ids"]))
    test_ids = sorted(map(str, split["partitions"]["test"]["case_ids"]))
    query_ids = validation_ids + test_ids

    nearest_reports = nearest_train_reports(train_ids, query_ids, reports)
    exact_train: dict[str, list[str]] = defaultdict(list)
    for case_id in train_ids:
        if reports[case_id]:
            exact_train[text_sha256(reports[case_id])].append(case_id)

    required_for_images = train_ids + query_ids
    hashes = case_image_hashes(required_for_images, cases, args.image_root)
    nearest_images = nearest_train_images(train_ids, query_ids, hashes)

    rows: list[dict[str, Any]] = []
    validation_set = set(validation_ids)
    for case_id in query_ids:
        report_match, similarity = nearest_reports.get(case_id, ("", float("nan")))
        image_match, image_distance = nearest_images.get(case_id, ("", -1))
        exact_matches = exact_train.get(text_sha256(reports[case_id]), []) if reports[case_id] else []
        rows.append(
            {
                "case_id": case_id,
                "split": "validation" if case_id in validation_set else "test",
                "report_available": bool(reports[case_id]),
                "exact_train_report_duplicate": bool(exact_matches),
                "exact_train_report_case_ids": ";".join(sorted(exact_matches)),
                "nearest_train_report_case_id": report_match,
                "nearest_train_report_cosine": similarity,
                "nearest_train_image_case_id": image_match,
                "nearest_train_image_dhash_distance": image_distance,
                "image_view_count": len(hashes.get(case_id, [])),
            }
        )

    args.case_output.parent.mkdir(parents=True, exist_ok=True)
    with args.case_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    report_thresholds = config["duplicate_audit"]["report_similarity_thresholds"]
    image_thresholds = config["duplicate_audit"]["image_hash"]["hamming_thresholds"]
    by_split: dict[str, dict[str, Any]] = {}
    for split_name in ("validation", "test"):
        split_rows = [row for row in rows if row["split"] == split_name]
        by_split[split_name] = {
            "case_count": len(split_rows),
            "report_available_count": sum(bool(row["report_available"]) for row in split_rows),
            "exact_train_report_duplicate_count": sum(
                bool(row["exact_train_report_duplicate"]) for row in split_rows
            ),
            "report_near_duplicate_counts": {
                str(threshold): sum(
                    bool(row["report_available"])
                    and float(row["nearest_train_report_cosine"]) >= float(threshold)
                    for row in split_rows
                )
                for threshold in report_thresholds
            },
            "image_dhash_counts": {
                str(threshold): sum(
                    int(row["nearest_train_image_dhash_distance"]) >= 0
                    and int(row["nearest_train_image_dhash_distance"]) <= int(threshold)
                    for row in split_rows
                )
                for threshold in image_thresholds
            },
        }

    exclusion_threshold = float(
        config["duplicate_audit"]["primary_sensitivity_exclusion_threshold"]
    )
    retained_test_ids = {
        str(row["case_id"])
        for row in rows
        if row["split"] == "test"
        and (
            not bool(row["report_available"])
            or float(row["nearest_train_report_cosine"]) < exclusion_threshold
        )
    }
    retrieval_rows = read_jsonl(args.retrieval_rows)
    summary = {
        "study": "V9 cross-split duplicate and near-duplicate audit",
        "status": "post_hoc_exploratory_complete",
        "config_sha256": file_sha256(args.config),
        "source_sha256": file_sha256(args.cases),
        "split_sha256": file_sha256(args.split),
        "case_output_sha256": file_sha256(args.case_output),
        "train_case_count": len(train_ids),
        "validation_case_count": len(validation_ids),
        "test_case_count": len(test_ids),
        "split_counts": by_split,
        "sensitivity_exclusion": {
            "report_cosine_threshold": exclusion_threshold,
            "retained_test_case_count": len(retained_test_ids),
            "excluded_test_case_count": len(test_ids) - len(retained_test_ids),
            "frozen_retrieval_metrics": aggregate_retrieval(
                retrieval_rows, retained_test_ids
            ),
        },
        "claim_boundary": (
            "Text and image hashes diagnose cross-split similarity; they do not "
            "establish patient identity or clinical leakage."
        ),
    }
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
