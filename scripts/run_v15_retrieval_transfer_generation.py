"""Generate the stronger-retrieval arm of the paired V15 QA transfer study."""

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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.multimodal.v9_generation import (  # noqa: E402
    MEDGEMMA_REVISION,
    MedGemmaImageGenerator,
    select_primary_image,
)
from medical_rag.similar_case.v10_evidence import sentence_units  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from medical_rag.similar_case.v11_output_contract import answer_only_generation_prompt  # noqa: E402
from medical_rag.similar_case.v11_question_planner import (  # noqa: E402
    plan_question,
    render_planner_instruction,
)
from run_v12_generation_pilot import (  # noqa: E402
    QUESTIONS,
    clean_answer,
    output_contract_diagnostic,
    read_json,
    read_jsonl,
    reference_for,
    sha256_ids,
    whole_report_units,
)

PROTOCOL_COMMIT = "6eb575d"
DEEPER_MODEL_SHA256 = "8c83d6188daa66939ae6a7865c14eada827c4cf625cc0314beaa4988ec2f086c"


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def paired_bootstrap(
    baseline: Sequence[Mapping[str, Any]],
    deeper: Sequence[Mapping[str, Any]],
    *,
    primary_only: bool,
    iterations: int = 10_000,
    seed: int = 1515,
) -> dict[str, float | int | bool]:
    baseline_by_key = {
        (str(row["case_id"]), str(row["question_type"])): float(row["token_f1"])
        for row in baseline
        if not primary_only or not bool(row["reference_is_proxy"])
    }
    deeper_by_key = {
        (str(row["case_id"]), str(row["question_type"])): float(row["token_f1"])
        for row in deeper
        if not primary_only or not bool(row["reference_is_proxy"])
    }
    if baseline_by_key.keys() != deeper_by_key.keys():
        raise RuntimeError("V15 paired Token-F1 keys differ")
    grouped: dict[str, list[float]] = defaultdict(list)
    for key in sorted(baseline_by_key):
        grouped[key[0]].append(deeper_by_key[key] - baseline_by_key[key])
    values = np.asarray([mean(grouped[case_id]) for case_id in sorted(grouped)])
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    low = float(np.quantile(draws, 0.025))
    high = float(np.quantile(draws, 0.975))
    return {
        "case_count": len(values),
        "mean_difference": float(values.mean()),
        "ci_95_low": low,
        "ci_95_high": high,
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "iterations": iterations,
        "seed": seed,
    }


def summarize(rows: Sequence[Mapping[str, Any]], *, primary_only: bool) -> dict[str, Any]:
    selected = [row for row in rows if not primary_only or not bool(row["reference_is_proxy"])]
    return {
        "row_count": len(selected),
        "case_count": len({str(row["case_id"]) for row in selected}),
        "token_f1": mean([float(row["token_f1"]) for row in selected]),
        "answer_contract_valid_rate": mean(
            [float(row["answer_only_contract_valid"]) for row in selected]
        ),
        "provenance_valid_rate": mean(
            [float(row["evidence_provenance_valid"]) for row in selected]
        ),
        "token_ceiling_rate": mean([float(row["hit_token_ceiling"]) for row in selected]),
        "mean_input_tokens": mean([float(row["input_tokens"]) for row in selected]),
        "mean_output_tokens": mean([float(row["output_tokens"]) for row in selected]),
        "mean_latency_seconds": mean([float(row["latency_seconds"]) for row in selected]),
        "by_question_type": {
            question_type: {
                "rows": sum(str(row["question_type"]) == question_type for row in selected),
                "token_f1": mean(
                    [
                        float(row["token_f1"])
                        for row in selected
                        if str(row["question_type"]) == question_type
                    ]
                ),
            }
            for question_type in QUESTIONS
            if any(str(row["question_type"]) == question_type for row in selected)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--ranking-rows", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_qwen3_validation_rankings_rows.jsonl")
    parser.add_argument("--ranking-summary", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_qwen3_validation_rankings.json")
    parser.add_argument("--selection-rows", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_selection_rows.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_manifest.json")
    parser.add_argument("--baseline-rows", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_96_rows.jsonl")
    parser.add_argument("--baseline-summary", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_96_summary.json")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/openi_official_images")
    parser.add_argument("--rows-output", type=Path, default=ROOT / "experiments/v15_retrieval_transfer/v15_deeper_rows.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/splits/v15/v15_retrieval_transfer_generation_summary.json")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    selection = read_jsonl(args.selection_rows)
    selected_case_ids = sorted({str(row["case_id"]) for row in selection})
    manifest = read_json(args.manifest)
    if len(selected_case_ids) != 48 or manifest["selected_case_ids_sha256"] != sha256_ids(selected_case_ids):
        raise RuntimeError("V15 source manifest differs from the frozen V12 48-case cohort")

    baseline_summary = read_json(args.baseline_summary)
    expected_model = baseline_summary["inputs"]["model"]
    if expected_model["revision"] != MEDGEMMA_REVISION or baseline_summary["max_new_tokens"] != 96:
        raise RuntimeError("V15 baseline model revision or token budget differs")
    if baseline_summary["inputs"]["selection_rows_sha256"] != file_sha256(args.selection_rows):
        raise RuntimeError("V15 baseline selection hash differs")
    if baseline_summary["generation_rows_sha256"] != file_sha256(args.baseline_rows):
        raise RuntimeError("V15 baseline row hash differs")
    baseline_rows = [
        row
        for row in read_jsonl(args.baseline_rows)
        if row["policy"] == "whole_report" and int(row["max_new_tokens"]) == 96
    ]
    expected_keys = {
        (case_id, question_type)
        for case_id in selected_case_ids
        for question_type in QUESTIONS
    }
    if {(str(row["case_id"]), str(row["question_type"])) for row in baseline_rows} != expected_keys:
        raise RuntimeError("V15 baseline rows do not cover the fixed matrix")

    ranking_summary = read_json(args.ranking_summary)
    if ranking_summary["inputs"]["model_sha256"] != DEEPER_MODEL_SHA256:
        raise RuntimeError("V15 ranking summary is not the frozen deeper model")
    ranking_rows = {
        (str(row["case_id"]), str(row["question_type"])): row
        for row in read_jsonl(args.ranking_rows)
        if str(row["case_id"]) in set(selected_case_ids)
    }
    if set(ranking_rows) != expected_keys:
        raise RuntimeError("V15 deeper ranking rows do not cover the fixed matrix")

    tasks: list[dict[str, Any]] = []
    for case_id in selected_case_ids:
        source = cases[case_id]
        for question_type, question in QUESTIONS.items():
            top_ids = [
                str(value)
                for value in ranking_rows[(case_id, question_type)]["rankings"]["rrf_lambdamart"][:3]
            ]
            evidence = whole_report_units([cases[candidate_id] for candidate_id in top_ids])
            plan = plan_question(question, str(source.get("indication", "")))
            prompt = answer_only_generation_prompt(
                indication=str(source.get("indication", "")),
                question=question,
                planner_instruction=render_planner_instruction(plan),
                evidence=evidence,
                abstain=not evidence,
            )
            reference, reference_proxy = reference_for(source, question_type)
            tasks.append(
                {
                    "case_id": case_id,
                    "question_type": question_type,
                    "question": question,
                    "reference_answer": reference,
                    "reference_is_proxy": reference_proxy,
                    "retrieved_case_ids": top_ids,
                    "evidence": evidence,
                    "prompt": prompt,
                    "target_image_path": str(select_primary_image(source, args.image_root)),
                }
            )
    if len(tasks) != 144:
        raise RuntimeError("V15 task matrix is incomplete")

    def key(row: Mapping[str, Any]) -> tuple[str, str]:
        return str(row["case_id"]), str(row["question_type"])

    completed = {key(row) for row in read_jsonl(args.rows_output)} if args.rows_output.exists() else set()
    pending = [task for task in tasks if key(task) not in completed]
    for task in pending:
        if not Path(task["target_image_path"]).is_file():
            raise FileNotFoundError(task["target_image_path"])

    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    generator = None
    if pending:
        generator = MedGemmaImageGenerator(
            revision=MEDGEMMA_REVISION,
            cache_dir=ROOT / ".hf_cache",
            local_files_only=True,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    run_started = time.perf_counter()
    with args.rows_output.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(pending), max(1, args.batch_size)):
            batch = pending[start : start + max(1, args.batch_size)]
            batch_started = time.perf_counter()
            outputs = generator.generate_batch(
                [str(task["prompt"]) for task in batch],
                [Path(str(task["target_image_path"])) for task in batch],
                max_new_tokens=96,
                stop_token="<end_of_turn>",
            )
            latency = (time.perf_counter() - batch_started) / len(batch)
            peak_mib = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
            for task, output in zip(batch, outputs, strict=True):
                raw_answer = str(output["answer"])
                answer = clean_answer(raw_answer)
                diagnostics = output_contract_diagnostic(
                    raw_answer, answer, hit_ceiling=bool(output["hit_token_ceiling"])
                )
                row = {
                    "condition": "deeper_17",
                    "case_id": task["case_id"],
                    "question_type": task["question_type"],
                    "question": task["question"],
                    "reference_answer": task["reference_answer"],
                    "reference_is_proxy": task["reference_is_proxy"],
                    "retrieved_case_ids": task["retrieved_case_ids"],
                    "target_image_path": task["target_image_path"],
                    "answer": answer,
                    "raw_output": raw_answer,
                    "token_f1": token_f1(answer, task["reference_answer"]),
                    "answer_only_contract_valid": diagnostics["answer_only_contract_valid"],
                    "serialization_leak_detected": diagnostics["serialization_leak_detected"],
                    "hit_token_ceiling": diagnostics["hit_token_ceiling"],
                    "evidence_provenance_valid": 1.0,
                    "input_tokens": int(output["input_tokens"]),
                    "output_tokens": int(output["output_tokens"]),
                    "latency_seconds": latency,
                    "peak_gpu_memory_mib": peak_mib,
                }
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            done = min(start + len(batch), len(pending))
            if done % 24 == 0 or done == len(pending):
                print(json.dumps({"generated": done, "pending_at_start": len(pending)}), flush=True)

    deeper_rows = read_jsonl(args.rows_output)
    if len(deeper_rows) != 144 or {key(row) for row in deeper_rows} != expected_keys:
        raise RuntimeError("V15 deeper generation rows are incomplete")
    summary = {
        "study": "V15 stronger retrieval to QA transfer",
        "status": "generation_complete_metric_extension_pending",
        "protocol_commit": PROTOCOL_COMMIT,
        "no_test_evaluation": True,
        "counts": {"cases": 48, "questions": 144, "primary_non_proxy_rows": 96},
        "conditions": {
            "default_17": {
                "all": summarize(baseline_rows, primary_only=False),
                "primary": summarize(baseline_rows, primary_only=True),
            },
            "deeper_17": {
                "all": summarize(deeper_rows, primary_only=False),
                "primary": summarize(deeper_rows, primary_only=True),
            },
        },
        "paired_token_f1": {
            "all": paired_bootstrap(baseline_rows, deeper_rows, primary_only=False),
            "primary": paired_bootstrap(baseline_rows, deeper_rows, primary_only=True),
        },
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "selection_rows_sha256": file_sha256(args.selection_rows),
            "selected_case_ids_sha256": sha256_ids(selected_case_ids),
            "baseline_rows_sha256": file_sha256(args.baseline_rows),
            "deeper_ranking_rows_sha256": file_sha256(args.ranking_rows),
            "deeper_model_sha256": DEEPER_MODEL_SHA256,
            "model": expected_model,
            "max_new_tokens": 96,
            "evidence_policy": "whole_report_top3",
        },
        "artifacts": {
            "deeper_rows": str(args.rows_output.resolve().relative_to(ROOT).as_posix()),
            "deeper_rows_sha256": file_sha256(args.rows_output),
        },
        "runtime": {
            "generated_this_run": len(pending),
            "elapsed_seconds": time.perf_counter() - run_started,
            "batch_size": args.batch_size,
            "max_peak_gpu_memory_mib": max(
                [float(row["peak_gpu_memory_mib"]) for row in deeper_rows], default=0.0
            ),
        },
        "claim_boundary": (
            "Validation-only automated consistency with same-source report references; "
            "not diagnosis, clinical correctness, safety, or external validation."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
