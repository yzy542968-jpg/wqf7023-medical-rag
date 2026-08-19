from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
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
)
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl  # noqa: E402


PROTOCOL_COMMIT = "eee7405"
COHORT_COMMIT = "43fe1a0"
RETRIEVAL_OUTCOME_COMMIT = "c6442c9"

DEFAULT_CONFIG = ROOT / "config" / "v6_confirmation.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_COHORT = ROOT / "data" / "splits" / "v6" / "v6_confirmation_cohort.json"
DEFAULT_RETRIEVAL_ROWS = (
    ROOT / "experiments" / "post_submission_v6" / "confirmation_retrieval_rows.jsonl"
)
DEFAULT_RETRIEVAL_SUMMARY = (
    ROOT / "experiments" / "post_submission_v6" / "confirmation_retrieval_summary.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "post_submission_v6"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def commit_exists(commit: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def exact_match(prediction: str, reference: str) -> float:
    normalize = lambda value: " ".join(str(value).lower().split())
    return float(normalize(prediction) == normalize(reference))


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    rows = read_jsonl(path)
    keys = [(str(row["system"]), str(row["qid"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("The partial confirmation QA output contains duplicate keys.")
    return set(keys)


def validate_frozen_inputs(
    *,
    config_path: Path,
    cohort_path: Path,
    retrieval_rows_path: Path,
    retrieval_summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for commit in (PROTOCOL_COMMIT, COHORT_COMMIT, RETRIEVAL_OUTCOME_COMMIT):
        if not commit_exists(commit):
            raise RuntimeError(f"Required frozen commit is unavailable: {commit}")

    config = read_json(config_path)
    cohort = read_json(cohort_path)
    retrieval_summary = read_json(retrieval_summary_path)

    if cohort["protocol_commit"] != PROTOCOL_COMMIT:
        raise RuntimeError("The confirmation cohort protocol commit changed.")
    if cohort["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("The confirmation config no longer matches the cohort freeze.")
    if cohort["case_count"] != 240 or cohort["question_count"] != 360:
        raise RuntimeError("The frozen confirmation cohort dimensions changed.")
    if retrieval_summary["status"] != "formal_confirmation_outcomes_frozen":
        raise RuntimeError("The confirmation retrieval outcome is not frozen.")
    if retrieval_summary["protocol_commit"] != PROTOCOL_COMMIT:
        raise RuntimeError("The retrieval protocol commit changed.")
    if retrieval_summary["cohort_commit"] != COHORT_COMMIT:
        raise RuntimeError("The retrieval cohort commit changed.")
    if retrieval_summary["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("The retrieval result uses a different confirmation config.")
    if retrieval_summary["cohort_sha256"] != file_sha256(cohort_path):
        raise RuntimeError("The retrieval result uses a different confirmation cohort.")
    if retrieval_summary["outputs"]["rows_sha256"] != file_sha256(retrieval_rows_path):
        raise RuntimeError("The local retrieval rows do not match the frozen summary.")
    if retrieval_summary["outputs"]["row_count"] != 1440:
        raise RuntimeError("The frozen retrieval matrix is incomplete.")

    generation = config["generation"]
    if generation["qwen"]["model"] != QWEN_MODEL:
        raise RuntimeError("The frozen Qwen model changed.")
    if generation["qwen"]["revision"] != QWEN_REVISION:
        raise RuntimeError("The frozen Qwen revision changed.")
    if generation["medgemma"]["model"] != MEDGEMMA_MODEL:
        raise RuntimeError("The frozen MedGemma model changed.")
    if generation["medgemma"]["revision"] != MEDGEMMA_REVISION:
        raise RuntimeError("The frozen MedGemma revision changed.")
    if generation["input"] != (
        "clinical_indication_question_and_top1_report_findings_impression_without_image_pixels"
    ):
        raise RuntimeError("The frozen generator input policy changed.")
    return config, cohort, retrieval_summary


def build_tasks(
    questions: list[dict[str, Any]],
    cases: dict[str, dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    required_systems = {"bm25", "medsiglip_max_chunk_reranker"}
    selected_rows = [
        row for row in retrieval_rows if str(row["system"]) in required_systems
    ]
    by_system_qid = {
        (str(row["system"]), str(row["qid"])): row for row in selected_rows
    }
    if len(by_system_qid) != 2 * len(questions):
        raise RuntimeError("The BM25/MedSigLIP confirmation retrieval matrix is incomplete.")

    retrieval_systems = {
        "bm25": "bm25",
        "medsiglip": "medsiglip_max_chunk_reranker",
    }
    tasks = []
    for retrieval_label, retrieval_system in retrieval_systems.items():
        for question in questions:
            qid = str(question["qid"])
            row = by_system_qid[(retrieval_system, qid)]
            selected_case_id = str(row["selected_case_id"])
            source_case_id = str(question["case_id"])
            if source_case_id not in cases or selected_case_id not in cases:
                raise RuntimeError(f"Missing source or retrieved case for {qid}.")
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
                        cases[selected_case_id],
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
            "retrieval_top1_accuracy": mean(
                float(row["retrieval_rank_correct"]) for row in system_rows
            ),
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
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, task in enumerate(pending, start=1):
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
            if index % 25 == 0 or index == len(pending):
                print(
                    json.dumps(
                        {
                            "model": model_label,
                            "generated_this_run": index,
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
        "generated_this_run": len(pending),
        "generation_seconds": seconds,
        "records_per_second": len(pending) / seconds if seconds else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen V6 confirmation 2x2 QA factorial."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--retrieval-rows", type=Path, default=DEFAULT_RETRIEVAL_ROWS)
    parser.add_argument("--retrieval-summary", type=Path, default=DEFAULT_RETRIEVAL_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "confirmation_qa_factorial_rows.jsonl"
    summary_path = args.output_dir / "confirmation_qa_factorial_summary.json"
    if summary_path.exists():
        raise RuntimeError("Formal confirmation QA summary already exists; refusing rerun.")

    config, cohort, retrieval_summary = validate_frozen_inputs(
        config_path=args.config,
        cohort_path=args.cohort,
        retrieval_rows_path=args.retrieval_rows,
        retrieval_summary_path=args.retrieval_summary,
    )
    cases = {str(case["case_id"]): case for case in load_cases_jsonl(args.cases)}
    questions = list(cohort["questions"])
    tasks = build_tasks(questions, cases, read_jsonl(args.retrieval_rows))
    if len(tasks) != 720:
        raise RuntimeError("The frozen confirmation QA task matrix must contain 720 tasks.")

    runtime: dict[str, Any] = {}
    max_new_tokens = int(config["generation"]["max_new_tokens"])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    qwen = QwenTextGenerator(
        revision=QWEN_REVISION,
        cache_dir=ROOT / ".hf_cache" / "hub",
        local_files_only=True,
    )
    runtime["qwen2_5"] = {
        "model": QWEN_MODEL,
        "revision": QWEN_REVISION,
        "precision": "float16",
        "load_seconds": time.perf_counter() - load_started,
        **run_model(
            model_label="qwen2_5",
            generator=qwen,
            tasks=tasks,
            output_path=output_path,
            max_new_tokens=max_new_tokens,
        ),
        "peak_gpu_memory_allocated_mib": float(torch.cuda.max_memory_allocated() / (1024**2)),
    }
    del qwen
    gc.collect()
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    medgemma = MedGemmaTextGenerator(
        revision=MEDGEMMA_REVISION,
        cache_dir=ROOT / ".hf_cache",
        local_files_only=True,
    )
    runtime["medgemma_1_5"] = {
        "model": MEDGEMMA_MODEL,
        "revision": MEDGEMMA_REVISION,
        "precision": "4-bit NF4, double quantization, bfloat16 compute",
        "load_seconds": time.perf_counter() - load_started,
        **run_model(
            model_label="medgemma_1_5",
            generator=medgemma,
            tasks=tasks,
            output_path=output_path,
            max_new_tokens=max_new_tokens,
        ),
        "peak_gpu_memory_allocated_mib": float(torch.cuda.max_memory_allocated() / (1024**2)),
    }
    del medgemma
    gc.collect()
    torch.cuda.empty_cache()

    rows = read_jsonl(output_path)
    expected_systems = {
        "bm25_qwen2_5",
        "medsiglip_qwen2_5",
        "bm25_medgemma_1_5",
        "medsiglip_medgemma_1_5",
    }
    observed_systems = {str(row["system"]) for row in rows}
    if observed_systems != expected_systems or len(rows) != 1440:
        raise RuntimeError("The formal V6 confirmation QA factorial is incomplete.")
    for system in expected_systems:
        system_row_count = sum(row["system"] == system for row in rows)
        if system_row_count != 360:
            raise RuntimeError(
                f"Formal V6 confirmation QA cell {system} has {system_row_count} rows."
            )

    summary = {
        "experiment": "V6 model-modernized confirmation QA factorial",
        "status": "formal_confirmation_raw_qa_outcomes_frozen",
        "protocol_commit": PROTOCOL_COMMIT,
        "cohort_commit": COHORT_COMMIT,
        "retrieval_outcome_commit": RETRIEVAL_OUTCOME_COMMIT,
        "config_sha256": file_sha256(args.config),
        "cohort_sha256": file_sha256(args.cohort),
        "retrieval_summary_sha256": file_sha256(args.retrieval_summary),
        "retrieval_rows_sha256": file_sha256(args.retrieval_rows),
        "implementation_sha256": file_sha256(Path(__file__)),
        "prompt_implementation_sha256": file_sha256(
            ROOT / "src" / "medical_rag" / "multimodal" / "v6_generation.py"
        ),
        "factorial": {
            "question_count": len(questions),
            "retrieval_conditions": ["bm25", "medsiglip"],
            "generator_conditions": ["qwen2_5", "medgemma_1_5"],
            "row_count": len(rows),
        },
        "generation_policy": {
            "semantic_prompt_identical_across_models": True,
            "model_specific_chat_templates_only": True,
            "image_pixels_in_primary_factorial": False,
            "do_sample": False,
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
        "metrics": summarize(rows),
        "runtime": runtime,
        "outputs": {
            "rows": portable_path(output_path),
            "rows_sha256": file_sha256(output_path),
            "row_count": len(rows),
            "summary": portable_path(summary_path),
        },
        "retrieval_summary_status": retrieval_summary["status"],
        "claim_boundary": (
            "Same-source report-grounded reference consistency; raw automated metrics are "
            "not clinical correctness or physician adjudication."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
