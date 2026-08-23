from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.multimodal.v9_generation import (  # noqa: E402
    MEDGEMMA_REVISION,
    MedGemmaImageGenerator,
)
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v10_generation import (  # noqa: E402
    assemble_deterministic_output,
    build_plain_answer_prompt,
    deterministic_historical_evidence,
    parse_plain_answer,
)
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from run_v10_evidence_generation_development import (  # noqa: E402
    DEFAULT_CASES,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_RADGRAPH,
    DEFAULT_RANKINGS,
    build_tasks,
    completed_keys,
    read_json,
    read_jsonl,
    select_policy,
    serialize_unit,
    summarize,
)


DEFAULT_CONFIG = ROOT / "config" / "v10_evidence_generation_revision1.json"
DEFAULT_ROWS = ROOT / "experiments" / "v10_publication" / "v10_evidence_generation_revision1_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_evidence_generation_revision1_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V10 answer-first deterministic assembly.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    config = read_json(args.config)
    if any(
        config[name]
        for name in (
            "validation_outcomes_inspected",
            "calibration_outcomes_inspected",
            "test_outcomes_inspected",
        )
    ):
        raise RuntimeError("revision config records forbidden outcomes")
    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    rankings = read_jsonl(args.rankings)
    radgraph = read_radgraph_case_records(args.radgraph)
    tasks = build_tasks(
        cases,
        rankings,
        radgraph,
        config["evidence_policies"],
        config["question_types"],
        args.image_root,
    )
    expected = len({str(row["case_id"]) for row in rankings}) * len(config["question_types"]) * len(
        config["evidence_policies"]
    )
    if len(tasks) != expected:
        raise RuntimeError(f"incomplete generation matrix: {len(tasks)} != {expected}")
    completed = completed_keys(args.rows_output)
    pending = [
        task
        for task in tasks
        if (task["case_id"], task["question_type"], task["evidence_policy"]) not in completed
    ]
    for task in pending:
        task["answer_prompt"] = build_plain_answer_prompt(
            indication=str(cases[task["case_id"]].get("indication", "")),
            question=str(task["question"]),
            evidence=task["evidence"],
            no_reliable_history=False,
        )

    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    generator = MedGemmaImageGenerator(
        revision=MEDGEMMA_REVISION,
        cache_dir=ROOT / ".hf_cache",
        local_files_only=True,
    )
    batch_size = int(config["generator"]["batch_size"])
    stop_token = str(config["generator"]["stop_token"])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with args.rows_output.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            outputs = generator.generate_batch(
                [str(task["answer_prompt"]) for task in batch],
                [Path(str(task["target_image_path"])) for task in batch],
                max_new_tokens=int(config["generator"]["answer_max_new_tokens"]),
                stop_token=stop_token,
            )
            for task, output in zip(batch, outputs, strict=True):
                answer_stage = parse_plain_answer(str(output["answer"]), stop_token=stop_token)
                support = deterministic_historical_evidence(
                    task["evidence"],
                    query="\n".join((str(task["question"]), answer_stage["answer"])),
                    retrieved_case_ids=task["retrieved_case_ids"],
                    maximum_units=int(config["provenance_attachment"]["maximum_units"]),
                )
                assembled = assemble_deterministic_output(
                    answer_stage,
                    support,
                    no_reliable_history=False,
                )
                allowed = {unit.provenance_id for unit in task["evidence"]}
                citation_valid = all(
                    row["provenance_id"] in allowed for row in assembled["historical_support"]
                )
                row = {
                    "case_id": task["case_id"],
                    "question_type": task["question_type"],
                    "question": task["question"],
                    "reference_answer": task["reference_answer"],
                    "evidence_policy": task["evidence_policy"],
                    "retrieved_case_ids": task["retrieved_case_ids"],
                    "target_image_path": task["target_image_path"],
                    "evidence": [serialize_unit(unit) for unit in task["evidence"]],
                    "answer_prompt": task["answer_prompt"],
                    "raw_answer_stage": output["answer"],
                    "raw_support_stage": "deterministic_provenance_attachment",
                    **assembled,
                    "citation_valid": citation_valid,
                    "token_f1": token_f1(assembled["answer"], task["reference_answer"]),
                    "evidence_unit_count": len(task["evidence"]),
                    "evidence_character_count": sum(len(unit.text) for unit in task["evidence"]),
                    "answer_input_tokens": int(output["input_tokens"]),
                    "answer_output_tokens": int(output["output_tokens"]),
                    "support_input_tokens": 0,
                    "support_output_tokens": 0,
                    "hit_answer_token_ceiling": bool(output["hit_token_ceiling"]),
                    "stopped_on_end_of_turn": bool(output["stopped_on_requested_token"]),
                }
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            done = min(start + len(batch), len(pending))
            if done % 80 == 0 or done == len(pending):
                print(json.dumps({"generated_this_run": done, "pending_at_start": len(pending)}), flush=True)

    rows = read_jsonl(args.rows_output)
    if len(rows) != len(tasks):
        raise RuntimeError(f"generation output incomplete: {len(rows)} != {len(tasks)}")
    metrics = summarize(rows, config["evidence_policies"])
    for policy in config["evidence_policies"]:
        selected = [row for row in rows if row["evidence_policy"] == policy]
        metrics[policy]["answer_token_ceiling_rate"] = sum(
            bool(row["hit_answer_token_ceiling"]) for row in selected
        ) / len(selected)
        metrics[policy]["end_of_turn_stop_rate"] = sum(
            bool(row["stopped_on_end_of_turn"]) for row in selected
        ) / len(selected)
    selected_policy = select_policy(metrics, config)
    elapsed = time.perf_counter() - started
    summary = {
        "study": "V10 evidence and compact generation development revision 1",
        "status": "development_complete_test_not_run",
        "inputs": {
            "config_sha256": file_sha256(args.config),
            "cases_sha256": file_sha256(args.cases),
            "radgraph_sha256": file_sha256(args.radgraph),
            "ranking_rows_sha256": file_sha256(args.rankings),
        },
        "counts": {
            "validation_cases": len({str(row["case_id"]) for row in rows}),
            "question_rows": len(rows) // len(config["evidence_policies"]),
            "generation_rows": len(rows),
        },
        "metrics": metrics,
        "selected_evidence_policy": selected_policy,
        "generation_rows_sha256": file_sha256(args.rows_output),
        "runtime": {
            "elapsed_seconds_this_run": elapsed,
            "peak_gpu_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "batch_size": batch_size,
        },
        "test_outcomes_inspected": False,
        "claim_boundary": "Automated same-source report-reference consistency, not clinical correctness.",
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
