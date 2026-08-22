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
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.multimodal.v6_generation import (  # noqa: E402
    MEDGEMMA_MODEL,
    MEDGEMMA_REVISION,
    MedGemmaTextGenerator,
    build_v6_qa_prompt,
)
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl  # noqa: E402


PROTOCOL_COMMIT = "4821f38"
COHORT_COMMIT = "25a39d8"
RETRIEVAL_RESULT_COMMIT = "ff629f4"

DEFAULT_CONFIG = ROOT / "config" / "v7_confirmation.json"
DEFAULT_V6_CONFIG = ROOT / "config" / "v6_confirmation.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_COHORT = ROOT / "data" / "splits" / "v7" / "v7_confirmation_cohort.json"
DEFAULT_RETRIEVAL_ROWS = (
    ROOT / "experiments" / "post_submission_v7" / "v7_confirmation_retrieval_rows.jsonl"
)
DEFAULT_RETRIEVAL_SUMMARY = (
    ROOT / "experiments" / "post_submission_v7" / "v7_confirmation_retrieval_summary.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "post_submission_v7"


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


def ranked_case_ids(row: Mapping[str, Any], alpha: float) -> list[str]:
    candidate_ids = [str(value) for value in row["candidate_case_ids"]]
    text_scores = [float(value) for value in row["text_scores_normalized"]]
    image_scores = [float(value) for value in row["image_scores_normalized"]]
    fused = [float(alpha) * text + (1.0 - float(alpha)) * image for text, image in zip(text_scores, image_scores, strict=True)]
    return [
        case_id
        for case_id, _ in sorted(
            zip(candidate_ids, fused, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    rows = read_jsonl(path)
    keys = [(str(row["system"]), str(row["qid"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("The partial V7 QA output contains duplicate keys.")
    return set(keys)


def validate_frozen_inputs(
    *,
    config_path: Path,
    v6_config_path: Path,
    cohort_path: Path,
    retrieval_rows_path: Path,
    retrieval_summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for commit in (PROTOCOL_COMMIT, COHORT_COMMIT, RETRIEVAL_RESULT_COMMIT):
        if not commit_exists(commit):
            raise RuntimeError(f"Required frozen commit is unavailable: {commit}")

    config = read_json(config_path)
    v6_config = read_json(v6_config_path)
    cohort = read_json(cohort_path)
    retrieval_summary = read_json(retrieval_summary_path)

    if config["secondary_qa"] != {
        "enabled_after_retrieval_freeze": True,
        "generator": "reuse_v6_medgemma_1_5",
        "prompt": "reuse_v6_prompt",
        "verifier": "reuse_v6_verifier",
        "image_pixels_to_generator": False,
        "model_selection_allowed": False,
    }:
        raise RuntimeError("The V7 secondary QA policy changed.")
    if cohort["protocol_commit"] != PROTOCOL_COMMIT:
        raise RuntimeError("The V7 confirmation protocol commit changed.")
    if cohort["case_count"] != 240 or cohort["question_count"] != 360:
        raise RuntimeError("The frozen V7 confirmation cohort dimensions changed.")
    if retrieval_summary["status"] != "formal_confirmation_retrieval_outcomes_frozen":
        raise RuntimeError("The V7 retrieval outcome is not frozen.")
    if retrieval_summary["protocol_commit"] != PROTOCOL_COMMIT:
        raise RuntimeError("The V7 retrieval protocol commit changed.")
    if retrieval_summary["cohort_sha256"] != file_sha256(cohort_path):
        raise RuntimeError("The retrieval result uses a different V7 cohort.")
    if retrieval_summary["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("The retrieval result uses a different V7 config.")
    if retrieval_summary["outputs"]["rows_sha256"] != file_sha256(retrieval_rows_path):
        raise RuntimeError("The local V7 retrieval rows do not match the frozen result.")
    if retrieval_summary["outputs"]["row_count"] != 360:
        raise RuntimeError("The V7 retrieval rows are incomplete.")

    v6_generation = v6_config["generation"]
    if v6_generation["medgemma"]["model"] != MEDGEMMA_MODEL:
        raise RuntimeError("The frozen V6 MedGemma model changed.")
    if v6_generation["medgemma"]["revision"] != MEDGEMMA_REVISION:
        raise RuntimeError("The frozen V6 MedGemma revision changed.")
    if v6_generation["input"] != (
        "clinical_indication_question_and_top1_report_findings_impression_without_image_pixels"
    ):
        raise RuntimeError("The frozen V6 generator input policy changed.")
    if v6_generation["do_sample"] is not False or float(v6_generation["temperature"]) != 0.0:
        raise RuntimeError("The frozen V6 deterministic decoding policy changed.")

    return config, v6_config, cohort, retrieval_summary


def validate_retrieval_rows(
    rows: Sequence[Mapping[str, Any]],
    questions: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    expected_qids = {str(question["qid"]) for question in questions}
    by_qid = {str(row["qid"]): row for row in rows}
    if len(rows) != 360 or set(by_qid) != expected_qids:
        raise RuntimeError("The V7 retrieval matrix does not match the frozen questions.")
    if any(not row.get("candidate_case_ids") for row in rows):
        raise RuntimeError("A V7 retrieval row has no BM25 shortlist.")
    return by_qid


def build_tasks(
    questions: Sequence[Mapping[str, Any]],
    cases: Mapping[str, Mapping[str, Any]],
    retrieval_rows: Mapping[str, Mapping[str, Any]],
    global_alpha: float,
) -> list[dict[str, Any]]:
    condition_alphas = {
        "bm25": lambda row: 1.0,
        "global_alpha_0_52": lambda row: float(global_alpha),
        "adaptive_alpha_q": lambda row: float(row["adaptive_alpha_q"]),
    }
    tasks: list[dict[str, Any]] = []
    for condition, alpha_fn in condition_alphas.items():
        for question in questions:
            qid = str(question["qid"])
            row = retrieval_rows[qid]
            source_case_id = str(question["case_id"])
            if source_case_id not in cases:
                raise RuntimeError(f"Missing source case for {qid}.")
            ranking = ranked_case_ids(row, alpha_fn(row))
            selected_case_id = ranking[0]
            if selected_case_id not in cases:
                raise RuntimeError(f"Missing selected case for {qid}.")
            tasks.append(
                {
                    "system": f"{condition}_medgemma_1_5",
                    "retrieval_condition": condition,
                    "qid": qid,
                    "case_id": source_case_id,
                    "question_type": str(question["question_type"]),
                    "question": str(question["question"]),
                    "reference_answer": str(question["reference_answer"]),
                    "selected_case_id": selected_case_id,
                    "retrieval_rank": ranking.index(source_case_id) + 1
                    if source_case_id in ranking
                    else None,
                    "retrieval_rank_correct": selected_case_id == source_case_id,
                    "target_in_shortlist": bool(row["target_in_shortlist"]),
                    "alpha": float(alpha_fn(row)),
                    "prompt": build_v6_qa_prompt(
                        question,
                        cases[source_case_id],
                        cases[selected_case_id],
                    ),
                }
            )
    if len(tasks) != 1080:
        raise RuntimeError("The V7 QA task matrix must contain 1080 tasks.")
    return tasks


def run_model(
    *,
    generator: MedGemmaTextGenerator,
    tasks: Sequence[Mapping[str, Any]],
    output_path: Path,
    max_new_tokens: int,
) -> dict[str, Any]:
    completed = completed_keys(output_path)
    expected_keys = {(str(task["system"]), str(task["qid"])) for task in tasks}
    if not completed.issubset(expected_keys):
        raise RuntimeError("The partial V7 QA output contains an unexpected task.")
    pending = [
        task
        for task in tasks
        if (str(task["system"]), str(task["qid"])) not in completed
    ]
    started = time.perf_counter()
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, task in enumerate(pending, start=1):
            result = generator.generate_one(str(task["prompt"]), max_new_tokens=max_new_tokens)
            answer = str(result["answer"])
            row = {
                **task,
                "answer": answer,
                "raw_token_f1": token_f1(answer, str(task["reference_answer"])),
                "exact_match": exact_match(answer, str(task["reference_answer"])),
                "input_tokens": int(result["input_tokens"]),
                "output_tokens": int(result["output_tokens"]),
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
            handle.flush()
            if index % 25 == 0 or index == len(pending):
                print(
                    json.dumps(
                        {
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


def validate_generation_rows(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]]
) -> None:
    expected = {(str(task["system"]), str(task["qid"])) for task in tasks}
    observed = {(str(row["system"]), str(row["qid"])) for row in rows}
    if len(rows) != 1080 or observed != expected:
        raise RuntimeError("The V7 raw QA matrix is incomplete or has unexpected tasks.")


def source_balanced(values: Sequence[Mapping[str, Any]], metric: str) -> float:
    by_case_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in values:
        by_case_type[str(row["case_id"])][str(row["question_type"])].append(float(row[metric]))
    case_scores = []
    for type_values in by_case_type.values():
        findings = mean(type_values["case_scoped_findings"])
        impression = mean(type_values["case_scoped_impression"])
        summary = mean(type_values["case_scoped_summary"])
        case_scores.append(0.50 * findings + 0.25 * impression + 0.25 * summary)
    return mean(case_scores) if case_scores else 0.0


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["system"])].append(row)
    result: dict[str, Any] = {}
    for system, system_rows in sorted(grouped.items()):
        result[system] = {
            "row_count": len(system_rows),
            "case_count": len({str(row["case_id"]) for row in system_rows}),
            "raw_token_f1": mean(float(row["raw_token_f1"]) for row in system_rows),
            "exact_match": mean(float(row["exact_match"]) for row in system_rows),
            "source_balanced_raw_token_f1": source_balanced(system_rows, "raw_token_f1"),
            "retrieval_top1_accuracy": mean(
                float(row["retrieval_rank_correct"]) for row in system_rows
            ),
            "target_outside_shortlist_rate": mean(
                float(not bool(row["target_in_shortlist"])) for row in system_rows
            ),
            "mean_input_tokens": mean(int(row["input_tokens"]) for row in system_rows),
            "mean_output_tokens": mean(int(row["output_tokens"]) for row in system_rows),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen V7 secondary MedGemma QA transfer.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v6-config", type=Path, default=DEFAULT_V6_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--retrieval-rows", type=Path, default=DEFAULT_RETRIEVAL_ROWS)
    parser.add_argument("--retrieval-summary", type=Path, default=DEFAULT_RETRIEVAL_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "v7_confirmation_qa_raw_rows.jsonl"
    summary_path = args.output_dir / "v7_confirmation_qa_raw_summary.json"
    if summary_path.exists():
        raise RuntimeError("The V7 raw QA summary already exists; refusing rerun.")

    config, v6_config, cohort, retrieval_summary = validate_frozen_inputs(
        config_path=args.config,
        v6_config_path=args.v6_config,
        cohort_path=args.cohort,
        retrieval_rows_path=args.retrieval_rows,
        retrieval_summary_path=args.retrieval_summary,
    )
    cases = {str(case["case_id"]): case for case in load_cases_jsonl(args.cases)}
    questions = list(cohort["questions"])
    retrieval_rows = validate_retrieval_rows(read_jsonl(args.retrieval_rows), questions)
    tasks = build_tasks(
        questions,
        cases,
        retrieval_rows,
        float(config["retrieval"]["global_alpha_star"]),
    )
    runtime: dict[str, Any] = {}
    max_new_tokens = int(v6_config["generation"]["max_new_tokens"])

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    generator = MedGemmaTextGenerator(
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
            generator=generator,
            tasks=tasks,
            output_path=output_path,
            max_new_tokens=max_new_tokens,
        ),
        "peak_gpu_memory_allocated_mib": float(
            torch.cuda.max_memory_allocated() / (1024**2)
        ),
    }
    del generator
    gc.collect()
    torch.cuda.empty_cache()

    rows = read_jsonl(output_path)
    validate_generation_rows(rows, tasks)
    summary = {
        "experiment": "V7 secondary MedGemma QA transfer",
        "status": "formal_confirmation_raw_qa_outcomes_frozen",
        "protocol_commit": PROTOCOL_COMMIT,
        "cohort_commit": COHORT_COMMIT,
        "retrieval_result_commit": RETRIEVAL_RESULT_COMMIT,
        "config_sha256": file_sha256(args.config),
        "v6_config_sha256": file_sha256(args.v6_config),
        "cohort_sha256": file_sha256(args.cohort),
        "retrieval_summary_sha256": file_sha256(args.retrieval_summary),
        "retrieval_rows_sha256": file_sha256(args.retrieval_rows),
        "implementation_sha256": file_sha256(Path(__file__)),
        "prompt_implementation_sha256": file_sha256(
            ROOT / "src" / "medical_rag" / "multimodal" / "v6_generation.py"
        ),
        "question_count": len(questions),
        "system_conditions": [
            "bm25_medgemma_1_5",
            "global_alpha_0_52_medgemma_1_5",
            "adaptive_alpha_q_medgemma_1_5",
        ],
        "row_count": len(rows),
        "generation_policy": {
            "generator": MEDGEMMA_MODEL,
            "revision": MEDGEMMA_REVISION,
            "prompt_reused_from_v6": True,
            "image_pixels_to_generator": False,
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
