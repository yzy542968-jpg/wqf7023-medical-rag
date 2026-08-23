from __future__ import annotations

import argparse
import gc
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
from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.multimodal.v9_generation import (  # noqa: E402
    MEDGEMMA_MODEL,
    MEDGEMMA_REVISION,
    MedGemmaImageGenerator,
    build_v9_qa_prompt,
    parse_v9_output,
    select_primary_image,
)
from medical_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from train_v9_learned_reranker import (  # noqa: E402
    MLPScorer,
    exact_leave_one_out_bm25_scores,
    feature_matrix,
)


DEFAULT_CONFIG = ROOT / "config" / "v9_qa_agent_confirmation.json"
DEFAULT_RETRIEVAL_CONFIG = ROOT / "config" / "v9_retrieval_confirmation.json"
DEFAULT_PROTOCOL = ROOT / "config" / "v9_similar_case_rag_development.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v9" / "v9_full_source_split.json"
DEFAULT_RETRIEVAL_SUMMARY = ROOT / "data" / "splits" / "v9" / "v9_retrieval_confirmation_summary.json"
DEFAULT_DEV_EMBEDDINGS = ROOT / "data" / "processed" / "v9_medsiglip_development_embeddings.npz"
DEFAULT_TEST_EMBEDDINGS = ROOT / "data" / "processed" / "v9_medsiglip_test_embeddings.npz"
DEFAULT_CHECKPOINT = ROOT / "experiments" / "post_submission_v9" / "reranker_checkpoints" / "v9_mlp_best.pt"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "post_submission_v9"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def exact_match(prediction: str, reference: str) -> float:
    normalize = lambda value: " ".join(str(value).lower().split())
    return float(normalize(prediction) == normalize(reference))


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    rows = read_jsonl(path)
    keys = [(str(row["system"]), str(row["qid"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Partial V9 QA output contains duplicate system/qid keys.")
    return set(keys)


def bootstrap_case_difference(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, *, iterations: int, seed: int
) -> dict[str, float]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["case_id"])][str(row["system"])].append(float(row["token_f1"]))
    differences = np.asarray(
        [statistics.fmean(values[left]) - statistics.fmean(values[right]) for values in grouped.values()],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        samples[index] = float(rng.choice(differences, size=len(differences), replace=True).mean())
    return {
        "difference": float(differences.mean()),
        "ci_95_low": float(np.quantile(samples, 0.025)),
        "ci_95_high": float(np.quantile(samples, 0.975)),
    }


def build_rankings(
    *,
    cases: Mapping[str, Any],
    raw_cases: Mapping[str, Mapping[str, Any]],
    qa_ids: Sequence[str],
    candidate_ids: Sequence[str],
    bank_images: np.ndarray,
    report_means: np.ndarray,
    test_embeddings: np.ndarray,
    test_ids: Sequence[str],
    checkpoint: Path,
    retrieval_config: Mapping[str, Any],
    questions: Mapping[str, str],
    top_k: int,
) -> list[dict[str, Any]]:
    bank = [cases[case_id] for case_id in candidate_ids]
    bm25 = BM25Retriever().fit(
        [{"case_id": case.study_id, "report_text": case.report_text} for case in bank]
    )
    term_cache: dict[str, tuple[np.ndarray, int]] = {}
    model = MLPScorer()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    fixed = retrieval_config["systems"]["r3"]["weights"]
    test_index = {case_id: index for index, case_id in enumerate(test_ids)}
    rows = []
    for case_index, case_id in enumerate(qa_ids, start=1):
        query = cases[case_id]
        embedding = test_embeddings[test_index[case_id]]
        image_image = bank_images @ embedding
        image_report = report_means @ embedding
        for question_type, question in questions.items():
            bm25_scores = exact_leave_one_out_bm25_scores(
                bm25, query.query_text(question), excluded_index=None, term_cache=term_cache
            )
            features = feature_matrix(
                bm25_scores,
                image_image,
                image_report,
                question_type=question_type,
                excluded_index=None,
            )
            with torch.inference_mode():
                learned = model(torch.from_numpy(features)).numpy()
            channels = {
                "r0_bm25": bm25_scores,
                "r1_image_image": image_image,
                "r3_fixed_multimodal": (
                    fixed["bm25"] * features[:, 0]
                    + fixed["image_image"] * features[:, 1]
                    + fixed["image_report"] * features[:, 2]
                ),
                "r4_learned_mlp": learned,
            }
            for system, scores in channels.items():
                order = np.lexsort((np.arange(len(scores)), -scores))[:top_k]
                rows.append(
                    {
                        "case_id": case_id,
                        "qid": f"{case_id}:{question_type}",
                        "question_type": question_type,
                        "system": system,
                        "top_case_ids": [candidate_ids[index] for index in order],
                        "top_scores": [float(scores[index]) for index in order],
                    }
                )
        if case_index % 100 == 0 or case_index == len(qa_ids):
            print(f"qa_rankings={case_index}/{len(qa_ids)}", flush=True)
    return rows


def build_tasks(
    *,
    qa_ids: Sequence[str],
    raw_cases: Mapping[str, Mapping[str, Any]],
    ranking_rows: Sequence[Mapping[str, Any]],
    questions: Mapping[str, str],
    image_root: Path,
) -> list[dict[str, Any]]:
    ranking = {(str(row["qid"]), str(row["system"])): row for row in ranking_rows}
    conditions = {
        "g0_no_retrieval": None,
        "g1_bm25_rag": "r0_bm25",
        "g2_fixed_multimodal_rag": "r3_fixed_multimodal",
        "g3_learned_multimodal_rag": "r4_learned_mlp",
    }
    tasks = []
    for system, retrieval_system in conditions.items():
        for case_id in qa_ids:
            source = raw_cases[case_id]
            image_path = select_primary_image(source, image_root)
            for question_type, question in questions.items():
                qid = f"{case_id}:{question_type}"
                top_ids = [] if retrieval_system is None else list(ranking[(qid, retrieval_system)]["top_case_ids"])
                reference = str(source[question_type])
                evidence = [raw_cases[value] for value in top_ids]
                tasks.append(
                    {
                        "system": system,
                        "retrieval_system": retrieval_system or "none",
                        "case_id": case_id,
                        "qid": qid,
                        "question_type": question_type,
                        "question": question,
                        "reference_answer": reference,
                        "target_image_path": str(image_path),
                        "retrieved_case_ids": top_ids,
                        "prompt": build_v9_qa_prompt(question, str(source.get("indication", "")), evidence),
                    }
                )
    return tasks


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["system"])].append(row)
    result = {}
    for system, values in sorted(grouped.items()):
        result[system] = {
            "row_count": len(values),
            "case_count": len({str(row["case_id"]) for row in values}),
            "token_f1": statistics.fmean(float(row["token_f1"]) for row in values),
            "exact_match": statistics.fmean(float(row["exact_match"]) for row in values),
            "structured_output_valid_rate": statistics.fmean(float(row["structured_output_valid"]) for row in values),
            "answer_abstention_rate": statistics.fmean(float(row["abstain"]) for row in values),
            "mean_input_tokens": statistics.fmean(int(row["input_tokens"]) for row in values),
            "mean_output_tokens": statistics.fmean(int(row["output_tokens"]) for row in values),
            "by_question_type": {
                question_type: statistics.fmean(
                    float(row["token_f1"]) for row in values if row["question_type"] == question_type
                )
                for question_type in ("findings", "impression")
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen V9 multimodal QA confirmation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--retrieval-config", type=Path, default=DEFAULT_RETRIEVAL_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--retrieval-summary", type=Path, default=DEFAULT_RETRIEVAL_SUMMARY)
    parser.add_argument("--development-embeddings", type=Path, default=DEFAULT_DEV_EMBEDDINGS)
    parser.add_argument("--test-embeddings", type=Path, default=DEFAULT_TEST_EMBEDDINGS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    config = read_json(args.config)
    retrieval_config = read_json(args.retrieval_config)
    protocol = read_json(args.protocol)
    split = read_json(args.split)
    retrieval_summary = read_json(args.retrieval_summary)
    if retrieval_summary["status"] != "confirmation_complete_no_retuning":
        raise RuntimeError("V9 retrieval confirmation is not frozen.")
    if file_sha256(args.checkpoint) != retrieval_config["systems"]["r4"]["checkpoint_sha256"]:
        raise RuntimeError("The frozen R4 checkpoint changed.")
    if file_sha256(args.test_embeddings) != retrieval_summary["test_embedding_cache"]["sha256"]:
        raise RuntimeError("The frozen Test image embedding cache changed.")

    raw_list = read_jsonl(args.cases)
    raw_cases = {str(row["case_id"]): row for row in raw_list}
    paired = read_openi_paired_cases(args.cases, source_unique_patient=True, radgraph_path=args.radgraph)
    cases = {case.study_id: case for case in paired}
    test_ids = sorted(str(value) for value in split["partitions"]["test"]["case_ids"])
    qa_ids = [case_id for case_id in test_ids if str(raw_cases[case_id].get("findings", "")).strip() and str(raw_cases[case_id].get("impression", "")).strip()]
    if len(qa_ids) != int(config["qa_frame"]["case_count"]):
        raise RuntimeError(f"V9 QA frame changed: observed {len(qa_ids)} cases.")
    questions = {key: protocol["question_suite"][key] for key in config["qa_frame"]["question_types"]}

    with np.load(args.development_embeddings, allow_pickle=False) as cache:
        candidate_ids = [str(value) for value in cache["candidate_ids"].tolist()]
        bank_images = np.asarray(cache["candidate_image_embeddings"], dtype=np.float32)
        report_means = np.asarray(cache["report_mean_embeddings"], dtype=np.float32)
    with np.load(args.test_embeddings, allow_pickle=False) as cache:
        embedded_test_ids = [str(value) for value in cache["test_ids"].tolist()]
        test_embeddings = np.asarray(cache["test_image_embeddings"], dtype=np.float32)
    if embedded_test_ids != test_ids:
        raise RuntimeError("Frozen Test embeddings do not match the Test split order.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rankings_path = args.output_dir / "v9_qa_top3_rankings.jsonl"
    rows_path = args.output_dir / "v9_qa_raw_rows.jsonl"
    summary_path = args.output_dir / "v9_qa_raw_summary.json"
    if summary_path.exists():
        raise RuntimeError("V9 QA summary already exists; refusing a formal rerun.")
    if rankings_path.exists():
        ranking_rows = read_jsonl(rankings_path)
    else:
        ranking_rows = build_rankings(
            cases=cases,
            raw_cases=raw_cases,
            qa_ids=qa_ids,
            candidate_ids=candidate_ids,
            bank_images=bank_images,
            report_means=report_means,
            test_embeddings=test_embeddings,
            test_ids=test_ids,
            checkpoint=args.checkpoint,
            retrieval_config=retrieval_config,
            questions=questions,
            top_k=int(config["historical_evidence"]["top_k"]),
        )
        with rankings_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in ranking_rows:
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    tasks = build_tasks(
        qa_ids=qa_ids,
        raw_cases=raw_cases,
        ranking_rows=ranking_rows,
        questions=questions,
        image_root=args.image_root,
    )
    expected_count = len(qa_ids) * len(questions) * 4
    if len(tasks) != expected_count:
        raise RuntimeError("V9 QA task matrix is incomplete.")
    completed = completed_keys(rows_path)
    expected_keys = {(str(task["system"]), str(task["qid"])) for task in tasks}
    if not completed.issubset(expected_keys):
        raise RuntimeError("Partial V9 QA output contains unexpected tasks.")
    pending = [task for task in tasks if (str(task["system"]), str(task["qid"])) not in completed]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    generator = MedGemmaImageGenerator(
        revision=MEDGEMMA_REVISION, cache_dir=ROOT / ".hf_cache", local_files_only=True
    )
    load_seconds = time.perf_counter() - load_started
    batch_size = int(config["generator"]["batch_size"])
    generation_started = time.perf_counter()
    with rows_path.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            outputs = generator.generate_batch(
                [str(task["prompt"]) for task in batch],
                [Path(str(task["target_image_path"])) for task in batch],
                max_new_tokens=int(config["generator"]["max_new_tokens"]),
            )
            for task, output in zip(batch, outputs, strict=True):
                parsed = parse_v9_output(str(output["answer"]), task["retrieved_case_ids"])
                row = {
                    **task,
                    **parsed,
                    "token_f1": token_f1(parsed["answer"], str(task["reference_answer"])),
                    "exact_match": exact_match(parsed["answer"], str(task["reference_answer"])),
                    "input_tokens": int(output["input_tokens"]),
                    "output_tokens": int(output["output_tokens"]),
                }
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            done = min(start + len(batch), len(pending))
            if done % 80 == 0 or done == len(pending):
                print(json.dumps({"generated_this_run": done, "pending_at_start": len(pending)}), flush=True)
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_started
    peak_mib = float(torch.cuda.max_memory_allocated() / (1024**2))
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    rows = read_jsonl(rows_path)
    observed_keys = {(str(row["system"]), str(row["qid"])) for row in rows}
    if len(rows) != expected_count or observed_keys != expected_keys:
        raise RuntimeError("Completed V9 QA output does not match the frozen matrix.")
    bootstrap = config["bootstrap"]
    summary = {
        "study": "V9 QA confirmation",
        "status": "formal_test_qa_outcomes_frozen_no_retuning",
        "upstream_retrieval_result_commit": config["upstream_retrieval_result_commit"],
        "config_sha256": file_sha256(args.config),
        "split_sha256": file_sha256(args.split),
        "retrieval_summary_sha256": file_sha256(args.retrieval_summary),
        "retrieval_config_sha256": file_sha256(args.retrieval_config),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "implementation_sha256": file_sha256(Path(__file__)),
        "prompt_implementation_sha256": file_sha256(ROOT / "src" / "medical_rag" / "multimodal" / "v9_generation.py"),
        "qa_case_count": len(qa_ids),
        "question_count": len(qa_ids) * len(questions),
        "row_count": len(rows),
        "metrics": summarize(rows),
        "primary_comparison_g3_minus_g0": bootstrap_case_difference(
            rows,
            "g3_learned_multimodal_rag",
            "g0_no_retrieval",
            iterations=int(bootstrap["iterations"]),
            seed=int(bootstrap["seed"]),
        ),
        "runtime": {
            "model": MEDGEMMA_MODEL,
            "revision": MEDGEMMA_REVISION,
            "load_seconds": load_seconds,
            "generation_seconds": generation_seconds,
            "records_per_second": len(pending) / generation_seconds if generation_seconds else None,
            "peak_gpu_memory_allocated_mib": peak_mib,
            "batch_size": batch_size,
        },
        "outputs": {
            "rankings_path": str(rankings_path.relative_to(ROOT)).replace("\\", "/"),
            "rankings_sha256": file_sha256(rankings_path),
            "rows_path": str(rows_path.relative_to(ROOT)).replace("\\", "/"),
            "rows_sha256": file_sha256(rows_path),
            "large_rows_committed": False,
        },
        "retuning_after_test_generation": False,
        "claim_boundary": "Automated same-source report-reference consistency, not physician-adjudicated clinical correctness.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
