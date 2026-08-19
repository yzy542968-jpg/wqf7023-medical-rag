from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.multimodal.v6_generation import (  # noqa: E402
    MEDGEMMA_MODEL,
    MEDGEMMA_REVISION,
    QWEN_MODEL,
    QWEN_REVISION,
    MedGemmaTextGenerator,
    QwenTextGenerator,
    build_v6_qa_prompt,
    select_preflight_qids,
)
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v6_development.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_COHORT = ROOT / "data" / "processed" / "openi_multimodal_v5_cohort.json"
DEFAULT_RETRIEVAL_ROWS = ROOT / "experiments" / "post_submission_v6" / "development_multimodal_retrieval_rows.jsonl"
DEFAULT_RETRIEVAL_SUMMARY = ROOT / "experiments" / "post_submission_v6" / "development_multimodal_retrieval_summary.json"
DEFAULT_TEXT_SUMMARY = ROOT / "experiments" / "post_submission_v6" / "development_text_retrieval_summary.json"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "post_submission_v6"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def exact_match(prediction: str, reference: str) -> float:
    normalize = lambda value: " ".join(str(value).lower().split())
    return float(normalize(prediction) == normalize(reference))


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    return {(str(row["system"]), str(row["qid"])) for row in read_jsonl(path)}


def validate_frozen_development_state(
    text_summary: dict[str, Any],
    retrieval_summary: dict[str, Any],
) -> None:
    if text_summary["selection"]["selected_text_retriever"] != "bm25":
        raise RuntimeError("The frozen V6 text retriever selection is no longer BM25.")
    selected = retrieval_summary["chunk_policy_selection"]["selected_chunk_policy"]
    if selected != "maximum_image_chunk_cosine":
        raise RuntimeError("The frozen V6 MedSigLIP chunk policy is no longer maximum cosine.")
    if retrieval_summary["selected_standardized_encoder_systems"]["medsiglip"] != "medsiglip_max_chunk_reranker":
        raise RuntimeError("The frozen V6 MedSigLIP system identifier changed.")


def build_tasks(
    questions: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_system_qid = {
        (str(row["system"]), str(row["qid"])): row for row in retrieval_rows
    }
    retrieval_systems = {
        "bm25": "indication_question_bm25",
        "medsiglip": "medsiglip_max_chunk_reranker",
    }
    tasks = []
    for retrieval_label, retrieval_system in retrieval_systems.items():
        for question in questions:
            qid = str(question["qid"])
            row = by_system_qid[(retrieval_system, qid)]
            selected_case_id = str(row["selected_case_id"])
            source_case_id = str(question["case_id"])
            tasks.append(
                {
                    "retrieval": retrieval_label,
                    "retrieval_system": retrieval_system,
                    "qid": qid,
                    "case_id": source_case_id,
                    "question_type": question.get("question_type"),
                    "question": str(question["question"]),
                    "reference_answer": str(question["reference_answer"]),
                    "selected_case_id": selected_case_id,
                    "retrieval_rank_correct": selected_case_id == source_case_id,
                    "prompt": build_v6_qa_prompt(
                        question,
                        cases[source_case_id],
                        cases.get(selected_case_id),
                    ),
                }
            )
    return tasks


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["system"])].append(row)
    return {
        system: {
            "row_count": len(system_rows),
            "case_count": len({str(row["case_id"]) for row in system_rows}),
            "raw_token_f1": mean(float(row["raw_token_f1"]) for row in system_rows),
            "exact_match": mean(float(row["exact_match"]) for row in system_rows),
            "retrieval_top1_accuracy": mean(float(row["retrieval_rank_correct"]) for row in system_rows),
            "mean_input_tokens": mean(int(row["input_tokens"]) for row in system_rows),
            "mean_output_tokens": mean(int(row["output_tokens"]) for row in system_rows),
        }
        for system, system_rows in sorted(grouped.items())
    }


def run_model(
    *,
    model_label: str,
    generator: Any,
    tasks: list[dict[str, Any]],
    output_path: Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    completed = completed_keys(output_path)
    pending = [
        task
        for task in tasks
        if (f"{task['retrieval']}_{model_label}", str(task["qid"])) not in completed
    ]
    started = time.perf_counter()
    generated_count = 0
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for task in pending:
            result = generator.generate_one(task["prompt"], max_new_tokens=max_new_tokens)
            answer = str(result["answer"])
            row = {
                **task,
                "system": f"{task['retrieval']}_{model_label}",
                "generator": model_label,
                "answer": answer,
                "raw_token_f1": token_f1(answer, task["reference_answer"]),
                "exact_match": exact_match(answer, task["reference_answer"]),
                "input_tokens": int(result["input_tokens"]),
                "output_tokens": int(result["output_tokens"]),
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            handle.flush()
            generated_count += 1
            if generated_count % 25 == 0:
                print(
                    json.dumps(
                        {
                            "model": model_label,
                            "generated_this_run": generated_count,
                            "pending_at_start": len(pending),
                        }
                    ),
                    flush=True,
                )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    return {
        "pending_at_start": len(pending),
        "generated_this_run": generated_count,
        "generation_seconds": seconds,
        "records_per_second": generated_count / seconds if seconds else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen V6 development 2x2 QA factorial.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--retrieval-rows", type=Path, default=DEFAULT_RETRIEVAL_ROWS)
    parser.add_argument("--retrieval-summary", type=Path, default=DEFAULT_RETRIEVAL_SUMMARY)
    parser.add_argument("--text-summary", type=Path, default=DEFAULT_TEXT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--preflight-count", type=int)
    parser.add_argument("--models", choices=["both", "qwen", "medgemma"], default="both")
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    config = read_json(args.config)
    cohort = read_json(args.cohort)
    cases = {str(case["case_id"]): case for case in load_cases_jsonl(args.cases)}
    retrieval_summary = read_json(args.retrieval_summary)
    text_summary = read_json(args.text_summary)
    validate_frozen_development_state(text_summary, retrieval_summary)

    development_qids = {str(value) for value in cohort["split"]["development"]["qids"]}
    if args.preflight_count is not None:
        development_qids = set(select_preflight_qids(list(development_qids), args.preflight_count))
    questions = [row for row in cohort["questions"] if str(row["qid"]) in development_qids]
    tasks = build_tasks(questions, cases, read_jsonl(args.retrieval_rows))
    expected_tasks = 2 * len(questions)
    if len(tasks) != expected_tasks:
        raise RuntimeError("The V6 QA task matrix is incomplete.")

    mode = f"preflight_{args.preflight_count}" if args.preflight_count is not None else "full"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"development_qa_factorial_{mode}_rows.jsonl"
    if output_path.exists() and not completed_keys(output_path):
        output_path.unlink()
    runtime: dict[str, Any] = {}
    max_new_tokens = int(config["generation"]["max_new_tokens"])

    if args.models in {"both", "qwen"}:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        qwen = QwenTextGenerator(
            revision=QWEN_REVISION,
            cache_dir=ROOT / ".hf_cache" / "hub",
            local_files_only=True,
        )
        load_seconds = time.perf_counter() - load_started
        qwen_runtime = run_model(
            model_label="qwen2_5",
            generator=qwen,
            tasks=tasks,
            output_path=output_path,
            max_new_tokens=max_new_tokens,
        )
        runtime["qwen2_5"] = {
            "model": QWEN_MODEL,
            "revision": QWEN_REVISION,
            "precision": "float16",
            "load_seconds": load_seconds,
            "peak_gpu_memory_allocated_mib": float(torch.cuda.max_memory_allocated() / (1024**2)),
            **qwen_runtime,
        }
        del qwen
        gc.collect()
        torch.cuda.empty_cache()

    if args.models in {"both", "medgemma"}:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        medgemma = MedGemmaTextGenerator(
            revision=MEDGEMMA_REVISION,
            cache_dir=ROOT / ".hf_cache",
            local_files_only=True,
        )
        load_seconds = time.perf_counter() - load_started
        medgemma_runtime = run_model(
            model_label="medgemma_1_5",
            generator=medgemma,
            tasks=tasks,
            output_path=output_path,
            max_new_tokens=max_new_tokens,
        )
        runtime["medgemma_1_5"] = {
            "model": MEDGEMMA_MODEL,
            "revision": MEDGEMMA_REVISION,
            "precision": "4-bit NF4, double quantization, bfloat16 compute",
            "load_seconds": load_seconds,
            "peak_gpu_memory_allocated_mib": float(torch.cuda.max_memory_allocated() / (1024**2)),
            **medgemma_runtime,
        }
        del medgemma
        gc.collect()
        torch.cuda.empty_cache()

    all_rows = read_jsonl(output_path)
    expected_systems = (
        {"bm25_qwen2_5", "medsiglip_qwen2_5", "bm25_medgemma_1_5", "medsiglip_medgemma_1_5"}
        if args.models == "both"
        else None
    )
    observed_systems = {str(row["system"]) for row in all_rows}
    complete = expected_systems is None or (
        observed_systems == expected_systems
        and len(all_rows) == len(questions) * len(expected_systems)
    )
    summary_path = args.output_dir / f"development_qa_factorial_{mode}_summary.json"
    summary = {
        "experiment": f"V6 development QA factorial {mode}",
        "protocol": "docs/V6_DEVELOPMENT_PROTOCOL.md",
        "complete": complete,
        "config_sha256": file_sha256(args.config),
        "implementation_sha256": file_sha256(Path(__file__)),
        "prompt_implementation_sha256": file_sha256(
            ROOT / "src" / "medical_rag" / "multimodal" / "v6_generation.py"
        ),
        "input_artifacts": {
            "cohort_sha256": file_sha256(args.cohort),
            "retrieval_rows_sha256": file_sha256(args.retrieval_rows),
            "retrieval_summary_sha256": file_sha256(args.retrieval_summary),
            "text_summary_sha256": file_sha256(args.text_summary),
        },
        "development": {
            "question_count": len(questions),
            "retrieval_conditions": ["bm25", "medsiglip"],
            "generator_conditions": sorted(runtime),
            "expected_factorial_row_count": len(questions) * 4,
            "observed_row_count": len(all_rows),
        },
        "generation_policy": {
            "semantic_prompt_identical_across_models": True,
            "model_specific_chat_templates_only": True,
            "image_pixels_in_primary_factorial": False,
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
        "metrics": summarize(all_rows),
        "runtime": runtime,
        "outputs": {
            "rows": portable_path(output_path),
            "rows_sha256": file_sha256(output_path),
            "summary": portable_path(summary_path),
        },
        "claim_boundary": "Development data only; no V6 confirmation case IDs were generated or inspected.",
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
