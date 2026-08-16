from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.multimodal.biomedclip import BiomedClipEncoder
from medical_rag.multimodal.evaluation import (
    aggregate_case_images,
    build_report_embedding_text,
    build_text_query,
    cosine_ranking,
    evaluate_rankings_and_answers,
)
from medical_rag.multimodal.fusion import reciprocal_rank_fusion, select_text_weight
from medical_rag.multimodal.openi_images import resolve_official_image
from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _git_blob(commit: str, path: str) -> bytes:
    result = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout


def verify_preregistered_config(commit: str, config_path: Path) -> None:
    relative = str(config_path.relative_to(ROOT)).replace("\\", "/")
    if _git_blob(commit, relative) != config_path.read_bytes():
        raise RuntimeError("Current multimodal configuration differs from the preregistered Git blob.")


def verify_committed_selection(commit: str, selection_path: Path) -> None:
    relative = str(selection_path.relative_to(ROOT)).replace("\\", "/")
    if _git_blob(commit, relative) != selection_path.read_bytes():
        raise RuntimeError("Current development selection differs from the committed Git blob.")


def candidate_case_ids(config: dict[str, Any]) -> list[str]:
    result = set()
    for split in ("development", "confirmation"):
        benchmark = load_json(ROOT / config["cohorts"][split]["benchmark_path"])
        result.update(str(row["case_id"]) for row in benchmark["questions"])
    return sorted(result)


def image_lookup(image_root: Path) -> dict[str, Path]:
    paths = list(image_root.rglob("*.png"))
    lookup: dict[str, Path] = {}
    duplicates = set()
    for path in paths:
        if path.name in lookup:
            duplicates.add(path.name)
        lookup[path.name] = path
    if duplicates:
        raise RuntimeError(f"Duplicate image names: {sorted(duplicates)[:5]}")
    return lookup


def eligible_cases(
    requested_ids: list[str],
    case_lookup: dict[str, dict[str, Any]],
    images: dict[str, Path],
) -> tuple[list[str], dict[str, list[Path]], list[dict[str, Any]]]:
    eligible = []
    case_images = {}
    exclusions = []
    for case_id in requested_ids:
        declared = [str(row["filename"]) for row in case_lookup[case_id].get("images", [])]
        matched = []
        for name in declared:
            path = resolve_official_image(case_id, name, images)
            if path is not None:
                matched.append(path)
        if not matched:
            exclusions.append({"case_id": case_id, "declared_images": declared})
            continue
        eligible.append(case_id)
        case_images[case_id] = matched
    return eligible, case_images, exclusions


def build_or_load_embeddings(
    cache_path: Path,
    case_ids: list[str],
    cases: dict[str, dict[str, Any]],
    case_images: dict[str, list[Path]],
    device: str,
    image_batch_size: int,
    text_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    case_fingerprint = hashlib.sha256("\n".join(case_ids).encode("utf-8")).hexdigest()
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=False)
        cached_ids = cache["case_ids"].tolist()
        cached_fingerprint = str(cache["case_fingerprint"].item())
        if cached_ids == case_ids and cached_fingerprint == case_fingerprint:
            return cache["image_embeddings"], cache["report_embeddings"], {"cache_hit": True}

    started = time.perf_counter()
    encoder = BiomedClipEncoder(device=device)
    report_texts = [build_report_embedding_text(cases[case_id]) for case_id in case_ids]
    report_embeddings = encoder.encode_texts(report_texts, batch_size=text_batch_size)

    view_paths = []
    view_case_ids = []
    for case_id in case_ids:
        for path in case_images[case_id]:
            view_paths.append(path)
            view_case_ids.append(case_id)
    view_embeddings = encoder.encode_images(view_paths, batch_size=image_batch_size)
    image_embeddings = aggregate_case_images(view_embeddings, view_case_ids, case_ids)
    runtime = {
        "cache_hit": False,
        "encoding_seconds": time.perf_counter() - started,
        "case_count": len(case_ids),
        "view_count": len(view_paths),
        "embedding_dimension": int(report_embeddings.shape[1]),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        case_ids=np.asarray(case_ids),
        case_fingerprint=np.asarray(case_fingerprint),
        image_embeddings=image_embeddings,
        report_embeddings=report_embeddings,
    )
    return image_embeddings, report_embeddings, runtime


def build_rankings(
    questions: list[dict[str, Any]],
    candidate_ids: list[str],
    cases: dict[str, dict[str, Any]],
    image_embeddings: np.ndarray,
    report_embeddings: np.ndarray,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    candidate_cases = [cases[case_id] for case_id in candidate_ids]
    bm25 = BM25Retriever().fit(candidate_cases)
    image_index = {case_id: index for index, case_id in enumerate(candidate_ids)}
    image_rank_by_case = {
        case_id: cosine_ranking(image_embeddings[index], report_embeddings, candidate_ids)
        for case_id, index in image_index.items()
    }
    text_rankings = {}
    image_rankings = {}
    for question in questions:
        qid = str(question["qid"])
        case_id = str(question["case_id"])
        query = build_text_query(cases[case_id], question)
        text_rankings[qid] = [row["case_id"] for row in bm25.search(query, top_k=len(candidate_ids))]
        image_rankings[qid] = image_rank_by_case[case_id]
    return text_rankings, image_rankings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the preregistered V4 paired image-report retrieval experiment.")
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "multimodal_v4.json")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "raw" / "openi_official_images")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "processed" / "multimodal_v4_embeddings.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "post_submission_v4")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--prereg-commit", default="795b1c9")
    parser.add_argument("--selection-commit")
    args = parser.parse_args()

    verify_preregistered_config(args.prereg_commit, args.config)
    config = load_json(args.config)
    all_cases = {str(row["case_id"]): row for row in load_cases_jsonl(ROOT / config["source"]["cases_path"])}
    requested = candidate_case_ids(config)
    images = image_lookup(args.image_root)
    eligible, case_images, exclusions = eligible_cases(requested, all_cases, images)

    benchmark = load_json(ROOT / config["cohorts"][args.split]["benchmark_path"])
    questions = [row for row in benchmark["questions"] if str(row["case_id"]) in set(eligible)]
    split_case_count = len({str(row["case_id"]) for row in questions})
    registered_count = int(config["cohorts"][args.split]["case_count"])
    if split_case_count != registered_count:
        raise RuntimeError(
            f"Eligible {args.split} cases ({split_case_count}) differ from the registered count ({registered_count})."
        )

    image_embeddings, report_embeddings, runtime = build_or_load_embeddings(
        args.cache,
        eligible,
        all_cases,
        case_images,
        args.device,
        args.image_batch_size,
        args.text_batch_size,
    )
    text_rankings, image_rankings = build_rankings(
        questions, eligible, all_cases, image_embeddings, report_embeddings
    )
    relevant = {str(row["qid"]): str(row["case_id"]) for row in questions}
    selection_path = args.output_dir / "development_retrieval_summary.json"

    if args.split == "development":
        selection = select_text_weight(
            text_rankings,
            image_rankings,
            relevant,
            config["retrieval"]["text_weight_grid"],
            constant=int(config["retrieval"]["rrf_constant"]),
        )
        text_weight = float(selection["selected_text_weight"])
    else:
        if not args.selection_commit:
            raise ValueError("--selection-commit is required for confirmation evaluation.")
        verify_committed_selection(args.selection_commit, selection_path)
        selection = load_json(selection_path)["development_selection"]
        text_weight = float(selection["selected_text_weight"])

    fused_rankings = {
        qid: reciprocal_rank_fusion(
            text_rankings[qid],
            image_rankings[qid],
            text_weight,
            constant=int(config["retrieval"]["rrf_constant"]),
        )
        for qid in relevant
    }
    systems = {
        "report_only_bm25": text_rankings,
        "image_only_biomedclip": image_rankings,
        "paired_rrf_fusion": fused_rankings,
    }
    metrics = {}
    output_rows = []
    for name, rankings in systems.items():
        system_metrics, rows = evaluate_rankings_and_answers(questions, rankings, all_cases)
        metrics[name] = system_metrics
        output_rows.extend({"system": name, **row} for row in rows)

    summary = {
        "experiment": config["experiment"],
        "split": args.split,
        "preregistration_commit": args.prereg_commit,
        "selection_commit": args.selection_commit,
        "config_sha256": sha256(args.config),
        "candidate_case_count": len(eligible),
        "split_case_count": split_case_count,
        "question_count": len(questions),
        "excluded_cases": exclusions,
        "development_selection": selection,
        "selected_text_weight": text_weight,
        "metrics": metrics,
        "runtime": runtime,
    }
    summary_path = args.output_dir / f"{args.split}_retrieval_summary.json"
    rows_path = args.output_dir / f"{args.split}_retrieval_rows.jsonl"
    write_json(summary_path, summary)
    write_jsonl(rows_path, output_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
