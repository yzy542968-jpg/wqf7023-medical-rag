from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import file_sha256, read_json  # noqa: E402
from medical_rag.evaluation.graded_retrieval import (  # noqa: E402
    binary_recall_at_k,
    ndcg_at_k,
    reciprocal_rank_at_threshold,
)
from medical_rag.multimodal.evaluation import aggregate_case_images  # noqa: E402
from medical_rag.multimodal.medsiglip import DEFAULT_REVISION, MedSiglipEncoder  # noqa: E402
from medical_rag.multimodal.openi_images import resolve_official_image  # noqa: E402
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from run_v9_development_medsiglip import image_lookup  # noqa: E402
from train_v9_learned_reranker import (  # noqa: E402
    MLPScorer,
    exact_leave_one_out_bm25_scores,
    feature_matrix,
    relevance_array,
)
from medical_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from medical_rag.similar_case.relevance import active_label_weights  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v9_retrieval_confirmation.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v9" / "v9_full_source_split.json"
DEFAULT_DEV_EMBEDDINGS = ROOT / "data" / "processed" / "v9_medsiglip_development_embeddings.npz"
DEFAULT_TEST_EMBEDDINGS = ROOT / "data" / "processed" / "v9_medsiglip_test_embeddings.npz"
DEFAULT_CHECKPOINT = ROOT / "experiments" / "post_submission_v9" / "reranker_checkpoints" / "v9_mlp_best.pt"
DEFAULT_PROTOCOL = ROOT / "config" / "v9_similar_case_rag_development.json"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_ROWS = ROOT / "experiments" / "post_submission_v9" / "v9_retrieval_confirmation_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v9" / "v9_retrieval_confirmation_summary.json"


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def shuffled_orders(case_ids: Sequence[str], *, seed: int, domain: str, count: int) -> list[dict[str, str]]:
    ordered = sorted(
        case_ids,
        key=lambda case_id: (
            hashlib.sha256(f"{domain}|{seed}|{case_id}".encode("utf-8")).hexdigest(),
            case_id,
        ),
    )
    assignments = []
    for shift in range(1, count + 1):
        mapping = {
            case_id: ordered[(index + shift) % len(ordered)]
            for index, case_id in enumerate(ordered)
        }
        if any(source == target for source, target in mapping.items()):
            raise RuntimeError("Shuffled assignment contains a fixed point.")
        assignments.append(mapping)
    return assignments


def metric_row(qrels: Mapping[str, float], ranking: Sequence[str]) -> dict[str, float]:
    return {
        "ndcg@1": ndcg_at_k(qrels, ranking, 1),
        "ndcg@5": ndcg_at_k(qrels, ranking, 5),
        "ndcg@10": ndcg_at_k(qrels, ranking, 10),
        "recall@1": binary_recall_at_k(qrels, ranking, 1, threshold=0.5),
        "recall@5": binary_recall_at_k(qrels, ranking, 5, threshold=0.5),
        "recall@10": binary_recall_at_k(qrels, ranking, 10, threshold=0.5),
        "mrr": reciprocal_rank_at_threshold(qrels, ranking, threshold=0.5),
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    names = ("ndcg@1", "ndcg@5", "ndcg@10", "recall@1", "recall@5", "recall@10", "mrr")
    return {name: statistics.fmean(float(row[name]) for row in rows) for name in names}


def bootstrap_difference(
    by_case: Mapping[str, float], *, iterations: int, seed: int
) -> tuple[float, float, float]:
    values = np.asarray([by_case[key] for key in sorted(by_case)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draws[index] = values[rng.integers(0, len(values), size=len(values))].mean()
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen V9 retrieval confirmation once.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--development-embeddings", type=Path, default=DEFAULT_DEV_EMBEDDINGS)
    parser.add_argument("--test-embeddings", type=Path, default=DEFAULT_TEST_EMBEDDINGS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--image-batch-size", type=int, default=4)
    args = parser.parse_args()

    started = time.perf_counter()
    config = read_json(args.config)
    split = read_json(args.split)
    protocol = read_json(args.protocol)
    if file_sha256(args.checkpoint) != config["systems"]["r4"]["checkpoint_sha256"]:
        raise RuntimeError("R4 checkpoint differs from confirmation protocol.")

    cases_list = read_openi_paired_cases(
        args.cases, source_unique_patient=True, radgraph_path=args.radgraph
    )
    cases = {case.study_id: case for case in cases_list}
    raw_cases = {
        str(row["case_id"]): row
        for row in (
            json.loads(line)
            for line in args.cases.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    with np.load(args.development_embeddings, allow_pickle=False) as cache:
        candidate_ids = [str(value) for value in cache["candidate_ids"].tolist()]
        bank_images = np.asarray(cache["candidate_image_embeddings"], dtype=np.float32)
        report_means = np.asarray(cache["report_mean_embeddings"], dtype=np.float32)
    test_ids = sorted(str(value) for value in split["partitions"]["test"]["case_ids"])
    if len(candidate_ids) != config["candidate_bank_count"] or len(test_ids) != config["test_case_count"]:
        raise RuntimeError("Frozen confirmation frame counts changed.")
    if set(candidate_ids) & set(test_ids):
        raise RuntimeError("Test study entered the historical candidate bank.")

    test_signature = hashlib.sha256(
        json.dumps(
            {
                "protocol_sha256": file_sha256(args.config),
                "source_sha256": file_sha256(args.cases),
                "test_ids": test_ids,
                "revision": DEFAULT_REVISION,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    test_embeddings: np.ndarray | None = None
    embedding_seconds = 0.0
    peak_mib = 0.0
    image_count = 0
    if args.test_embeddings.exists():
        with np.load(args.test_embeddings, allow_pickle=False) as cache:
            if str(cache["signature"].item()) == test_signature:
                test_embeddings = np.asarray(cache["test_image_embeddings"], dtype=np.float32)
                embedding_seconds = float(cache["embedding_seconds"].item())
                peak_mib = float(cache["peak_gpu_memory_mib"].item())
                image_count = int(cache["image_count"].item())
    if test_embeddings is None:
        lookup = image_lookup(args.image_root)
        view_case_ids: list[str] = []
        view_paths: list[Path] = []
        for case_id in test_ids:
            paths = [
                resolve_official_image(case_id, str(row["filename"]), lookup)
                for row in raw_cases[case_id].get("images", [])
            ]
            paths = [path for path in paths if path is not None]
            if not paths:
                raise RuntimeError(f"Test case {case_id} has no readable image.")
            view_case_ids.extend([case_id] * len(paths))
            view_paths.extend(paths)
        os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
        encoder = MedSiglipEncoder(
            revision=DEFAULT_REVISION, cache_dir=ROOT / ".hf_cache", local_files_only=True
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        embedding_started = time.perf_counter()
        views = encoder.encode_images(view_paths, batch_size=args.image_batch_size)
        test_embeddings = aggregate_case_images(views, view_case_ids, test_ids)
        embedding_seconds = time.perf_counter() - embedding_started
        peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
        image_count = len(view_paths)
        args.test_embeddings.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.test_embeddings,
            signature=np.asarray(test_signature),
            test_ids=np.asarray(test_ids),
            test_image_embeddings=test_embeddings.astype(np.float32),
            embedding_seconds=np.asarray(embedding_seconds),
            peak_gpu_memory_mib=np.asarray(peak_mib),
            image_count=np.asarray(image_count),
        )

    bank = [cases[case_id] for case_id in candidate_ids]
    bm25 = BM25Retriever().fit(
        [{"case_id": case.study_id, "report_text": case.report_text} for case in bank]
    )
    term_cache: dict[str, tuple[np.ndarray, int]] = {}
    prepared_labels = [active_label_weights(case.labels) for case in bank]
    prepared_facts = [case.radgraph_facts for case in bank]
    model = MLPScorer()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    fixed = config["systems"]["r3"]["weights"]
    rows: list[dict[str, Any]] = []
    state: dict[str, dict[str, Any]] = {}
    for query_index, case_id in enumerate(test_ids):
        query = cases[case_id]
        gains = relevance_array(
            query, bank, None, prepared_labels=prepared_labels, prepared_facts=prepared_facts
        )
        qrels = dict(zip(candidate_ids, map(float, gains), strict=True))
        ii = bank_images @ test_embeddings[query_index]
        ir = report_means @ test_embeddings[query_index]
        state[case_id] = {"gains": gains, "qrels": qrels, "ii": ii, "ir": ir, "bm25": {}}
        for question_type, question in protocol["question_suite"].items():
            text = exact_leave_one_out_bm25_scores(
                bm25, query.query_text(question), excluded_index=None, term_cache=term_cache
            )
            state[case_id]["bm25"][question_type] = text
            features = feature_matrix(
                text, ii, ir, question_type=question_type, excluded_index=None
            )
            with torch.inference_mode():
                learned = model(torch.from_numpy(features)).numpy()
            channels = {
                "r0_bm25": text,
                "r1_image_image": ii,
                "r2_image_report": ir,
                "r3_fixed_multimodal": (
                    fixed["bm25"] * features[:, 0]
                    + fixed["image_image"] * features[:, 1]
                    + fixed["image_report"] * features[:, 2]
                ),
                "r4_learned_mlp": learned,
            }
            for system, scores in channels.items():
                order = np.lexsort((np.arange(len(scores)), -scores))
                ranking = [candidate_ids[index] for index in order]
                rows.append(
                    {
                        "case_id": case_id,
                        "qid": f"{case_id}:{question_type}",
                        "question_type": question_type,
                        "system": system,
                        **metric_row(qrels, ranking),
                    }
                )
        if (query_index + 1) % 50 == 0 or query_index + 1 == len(test_ids):
            print(f"aligned_test_cases={query_index + 1}/{len(test_ids)}", flush=True)

    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)
    metrics = {name: aggregate(values) for name, values in by_system.items()}
    by_case_difference = {
        case_id: statistics.fmean(
            row["ndcg@10"]
            for row in rows
            if row["case_id"] == case_id and row["system"] == "r4_learned_mlp"
        )
        - statistics.fmean(
            row["ndcg@10"]
            for row in rows
            if row["case_id"] == case_id and row["system"] == "r1_image_image"
        )
        for case_id in test_ids
    }
    observed, ci_low, ci_high = bootstrap_difference(
        by_case_difference,
        iterations=int(config["bootstrap"]["iterations"]),
        seed=int(config["bootstrap"]["seed"]),
    )

    strict_ids = set(split["strict_project_history_untouched_test_subset"]["case_ids"])
    strict_metrics = {
        system: aggregate([row for row in values if row["case_id"] in strict_ids])
        for system, values in by_system.items()
    }

    assignments = shuffled_orders(
        test_ids,
        seed=int(config["shuffled_control"]["seed"]),
        domain=config["shuffled_control"]["order_domain"],
        count=int(config["shuffled_control"]["assignment_count"]),
    )
    test_index = {case_id: index for index, case_id in enumerate(test_ids)}
    shuffled_values: list[float] = []
    for assignment_index, assignment in enumerate(assignments, start=1):
        ndcgs: list[float] = []
        for case_id in test_ids:
            wrong_embedding = test_embeddings[test_index[assignment[case_id]]]
            ii = bank_images @ wrong_embedding
            ir = report_means @ wrong_embedding
            qrels = state[case_id]["qrels"]
            for question_type in protocol["question_suite"]:
                features = feature_matrix(
                    state[case_id]["bm25"][question_type],
                    ii,
                    ir,
                    question_type=question_type,
                    excluded_index=None,
                )
                with torch.inference_mode():
                    scores = model(torch.from_numpy(features)).numpy()
                order = np.lexsort((np.arange(len(scores)), -scores))
                ranking = [candidate_ids[index] for index in order[:10]]
                ndcgs.append(ndcg_at_k(qrels, ranking, 10))
        shuffled_values.append(statistics.fmean(ndcgs))
        print(f"shuffled_assignment={assignment_index}/{len(assignments)} ndcg10={shuffled_values[-1]:.6f}", flush=True)

    aligned = metrics["r4_learned_mlp"]["ndcg@10"]
    plus_one_p = (1 + sum(value >= aligned for value in shuffled_values)) / (
        len(shuffled_values) + 1
    )
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    with args.rows_output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    summary = {
        "study": "V9 retrieval confirmation",
        "status": "confirmation_complete_no_retuning",
        "candidate_bank_count": len(candidate_ids),
        "test_case_count": len(test_ids),
        "test_question_count": len(test_ids) * len(protocol["question_suite"]),
        "metrics": metrics,
        "primary_comparison_r4_minus_r1": {
            "difference": observed,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "bootstrap_iterations": config["bootstrap"]["iterations"],
            "confirmed_superiority": ci_low > 0,
        },
        "strict_untouched_subset": {"case_count": len(strict_ids), "metrics": strict_metrics},
        "shuffled_control": {
            "assignments": len(shuffled_values),
            "aligned_ndcg@10": aligned,
            "shuffled_mean_ndcg@10": statistics.fmean(shuffled_values),
            "shuffled_std_ndcg@10": statistics.stdev(shuffled_values),
            "shuffled_2_5_percentile": float(np.quantile(shuffled_values, 0.025)),
            "shuffled_97_5_percentile": float(np.quantile(shuffled_values, 0.975)),
            "plus_one_p_value": plus_one_p,
            "alignment_dependence_confirmed": plus_one_p <= 0.05,
            "values": shuffled_values,
        },
        "test_embedding_cache": {
            "path": portable_path(args.test_embeddings),
            "sha256": file_sha256(args.test_embeddings),
            "image_count": image_count,
            "build_seconds": embedding_seconds,
            "peak_gpu_memory_mib": peak_mib,
            "committed_to_public_repository": False,
        },
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "rows_output": {
            "path": portable_path(args.rows_output),
            "sha256": file_sha256(args.rows_output),
            "committed_to_public_repository": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "retuning_after_confirmation": False,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
