from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_multimodal_v4_retrieval import (
    build_rankings,
    candidate_case_ids,
    eligible_cases,
    image_lookup,
    load_json,
    sha256,
    verify_committed_selection,
    verify_preregistered_config,
    write_json,
    write_jsonl,
)

from medical_rag.multimodal.biovilt import BioVilTEncoder
from medical_rag.multimodal.evaluation import (
    aggregate_case_images,
    build_report_embedding_text,
    evaluate_confirmation_gate,
    evaluate_rankings_and_answers,
)
from medical_rag.multimodal.fusion import reciprocal_rank_fusion, select_text_weight
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def build_or_load_embeddings(
    cache_path: Path,
    case_ids: list[str],
    cases: dict[str, dict[str, Any]],
    case_images: dict[str, list[Path]],
    config: dict[str, Any],
    device: str,
    image_batch_size: int,
    text_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    case_fingerprint = hashlib.sha256("\n".join(case_ids).encode("utf-8")).hexdigest()
    encoder_fingerprint = hashlib.sha256(
        json.dumps(config["retrieval"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    if cache_path.exists():
        cache = np.load(cache_path, allow_pickle=False)
        if (
            cache["case_ids"].tolist() == case_ids
            and str(cache["case_fingerprint"].item()) == case_fingerprint
            and str(cache["encoder_fingerprint"].item()) == encoder_fingerprint
        ):
            return cache["image_embeddings"], cache["report_embeddings"], {"cache_hit": True}

    import torch

    started = time.perf_counter()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    encoder = BioVilTEncoder(
        model_name=config["retrieval"]["joint_encoder"],
        text_revision=config["retrieval"]["text_model_revision"],
        device=device,
        text_max_length=int(config["retrieval"]["text_max_length"]),
    )
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
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if device.startswith("cuda") else None
        ),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        case_ids=np.asarray(case_ids),
        case_fingerprint=np.asarray(case_fingerprint),
        encoder_fingerprint=np.asarray(encoder_fingerprint),
        image_embeddings=image_embeddings,
        report_embeddings=report_embeddings,
    )
    return image_embeddings, report_embeddings, runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preregistered BioViL-T V4.1 paired retrieval.")
    parser.add_argument("--split", choices=("development", "confirmation"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "multimodal_v41.json")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data" / "raw" / "openi_official_images")
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data" / "processed" / "multimodal_v41_biovil_t_embeddings.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "post_submission_v41")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--prereg-commit", default="a8cd6d1")
    parser.add_argument("--selection-commit")
    args = parser.parse_args()

    verify_preregistered_config(args.prereg_commit, args.config)
    config = load_json(args.config)
    all_cases = {
        str(row["case_id"]): row
        for row in load_cases_jsonl(ROOT / config["source"]["cases_path"])
    }
    requested = candidate_case_ids(config)
    images = image_lookup(args.image_root)
    eligible, case_images, exclusions = eligible_cases(requested, all_cases, images)

    benchmark = load_json(ROOT / config["cohorts"][args.split]["benchmark_path"])
    eligible_set = set(eligible)
    questions = [row for row in benchmark["questions"] if str(row["case_id"]) in eligible_set]
    split_case_count = len({str(row["case_id"]) for row in questions})
    if split_case_count != int(config["cohorts"][args.split]["case_count"]):
        raise RuntimeError("Eligible split case count differs from the preregistered count.")

    image_embeddings, report_embeddings, runtime = build_or_load_embeddings(
        args.cache,
        eligible,
        all_cases,
        case_images,
        config,
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
        development = load_json(selection_path)
        if not development["confirmation_gate"]["passed"]:
            raise RuntimeError("The preregistered development confirmation gate did not pass.")
        selection = development["development_selection"]
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
        "image_only_biovil_t": image_rankings,
        "paired_biovil_t_rrf": fused_rankings,
    }
    metrics = {}
    output_rows = []
    for name, rankings in systems.items():
        system_metrics, rows = evaluate_rankings_and_answers(questions, rankings, all_cases)
        metrics[name] = system_metrics
        output_rows.extend({"system": name, **row} for row in rows)

    gate = evaluate_confirmation_gate(config, metrics, text_weight)
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
        "confirmation_gate": gate,
        "runtime": runtime,
    }
    write_json(args.output_dir / f"{args.split}_retrieval_summary.json", summary)
    write_jsonl(args.output_dir / f"{args.split}_retrieval_rows.jsonl", output_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
