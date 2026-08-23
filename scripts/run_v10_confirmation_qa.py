from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.evaluation.v10_confirmation import case_grouped_bootstrap_difference  # noqa: E402
from medical_rag.multimodal.v9_generation import (  # noqa: E402
    MEDGEMMA_REVISION,
    MedGemmaImageGenerator,
    select_primary_image,
)
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v10_evidence import select_case_evidence  # noqa: E402
from medical_rag.similar_case.v10_generation import (  # noqa: E402
    assemble_deterministic_output,
    build_plain_answer_prompt,
    deterministic_historical_evidence,
    normalize_bounded_answer,
    parse_plain_answer,
)
from medical_rag.similar_case.v10_runtime import QUESTIONS  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from run_v10_confirmation_retrieval import verify_frozen_inputs  # noqa: E402
from run_v10_evidence_generation_development import completed_keys, read_json, read_jsonl, serialize_unit  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v10_confirmation.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v10" / "v10_cluster_disjoint_split.json"
DEFAULT_EMBEDDINGS = ROOT / "data" / "processed" / "v10_medsiglip_embeddings.npz"
DEFAULT_CALIBRATOR = ROOT / "artifacts" / "v10" / "retrieval_calibrator.json"
DEFAULT_RETRIEVAL = ROOT / "experiments" / "v10_publication" / "v10_confirmation_retrieval_rows.jsonl"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_ROWS = ROOT / "experiments" / "v10_publication" / "v10_confirmation_qa_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_confirmation_qa_summary.json"


SYSTEMS = ("g0_target_image", "g1_whole_report", "g2_hierarchical", "g3_selective")


def completed_system_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    return {
        (str(row["case_id"]), str(row["question_type"]), str(row["system"]))
        for row in read_jsonl(path)
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for system in SYSTEMS:
        selected = [row for row in rows if row["system"] == system]
        case_values: dict[str, list[float]] = defaultdict(list)
        for row in selected:
            case_values[str(row["case_id"])].append(float(row["token_f1"]))
        result[system] = {
            "row_count": len(selected),
            "case_count": len(case_values),
            "token_f1_equal_question": statistics.fmean(float(row["token_f1"]) for row in selected),
            "token_f1_case_averaged": statistics.fmean(
                statistics.fmean(values) for values in case_values.values()
            ),
            "assembled_schema_valid_rate": statistics.fmean(
                float(row["assembled_schema_valid"]) for row in selected
            ),
            "citation_valid_rate": statistics.fmean(float(row["citation_valid"]) for row in selected),
            "answer_token_ceiling_rate": statistics.fmean(
                float(row["hit_answer_token_ceiling"]) for row in selected
            ),
            "evidence_abstention_rate": statistics.fmean(
                float(row["evidence_abstained"]) for row in selected
            ),
            "by_question_type": {
                question_type: statistics.fmean(
                    float(row["token_f1"])
                    for row in selected
                    if row["question_type"] == question_type
                )
                for question_type in ("findings", "impression")
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen V10 Test QA conditions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    parser.add_argument("--retrieval-rows", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    config = read_json(args.config)
    verify_frozen_inputs(
        config,
        {
            "cases": args.cases,
            "radgraph": args.radgraph,
            "split": args.split,
            "embeddings": args.embeddings,
            "calibrator": args.calibrator,
        },
    )
    retrieval_rows = read_jsonl(args.retrieval_rows)
    if file_sha256(args.retrieval_rows) != str(config["test_retrieval_rows_sha256"]):
        raise RuntimeError("Test retrieval rows differ from the completed frozen retrieval run")
    retrieval = {
        (str(row["case_id"]), str(row["question_type"])): row
        for row in retrieval_rows
        if row["system"] == "r5_fact_attention"
    }
    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    radgraph = read_radgraph_case_records(args.radgraph)
    selected_policy = str(config["evidence_policy"])
    tasks = []
    for (case_id, question_type), ranking in sorted(retrieval.items()):
        if question_type not in {"findings", "impression"}:
            continue
        source = cases[case_id]
        question = QUESTIONS[question_type]
        top_ids = list(ranking["top_case_ids"][:3])
        for system in SYSTEMS:
            no_reliable = system == "g3_selective" and bool(ranking["no_reliable_history"])
            if system == "g0_target_image" or no_reliable:
                retrieved_ids = []
                evidence = []
            else:
                retrieved_ids = top_ids
                policy = "whole_report" if system == "g1_whole_report" else selected_policy
                evidence = []
                for retrieved_id in retrieved_ids:
                    evidence.extend(
                        select_case_evidence(
                            cases[retrieved_id],
                            query="\n".join((str(source.get("indication", "")), question)),
                            facts=radgraph[retrieved_id].facts,
                            policy=policy,
                        )
                    )
            prompt = build_plain_answer_prompt(
                indication=str(source.get("indication", "")),
                question=question,
                evidence=evidence,
                no_reliable_history=(system == "g0_target_image" or no_reliable),
            )
            tasks.append(
                {
                    "case_id": case_id,
                    "question_type": question_type,
                    "question": question,
                    "reference_answer": str(source[question_type]),
                    "system": system,
                    "retrieved_case_ids": retrieved_ids,
                    "evidence": evidence,
                    "no_reliable_history": no_reliable,
                    "target_image_path": str(select_primary_image(source, args.image_root)),
                    "answer_prompt": prompt,
                    "retrieval_confidence": ranking.get("retrieval_confidence"),
                }
            )
    expected = len(retrieval) // len(QUESTIONS) * 2 * len(SYSTEMS)
    if len(tasks) != expected:
        raise RuntimeError(f"incomplete QA matrix: {len(tasks)} != {expected}")
    completed = completed_system_keys(args.rows_output)
    pending = [
        task
        for task in tasks
        if (task["case_id"], task["question_type"], task["system"]) not in completed
    ]
    pending.sort(
        key=lambda task: (
            str(task["system"]),
            len(str(task["answer_prompt"])),
            str(task["case_id"]),
            str(task["question_type"]),
        )
    )

    generator = MedGemmaImageGenerator(
        revision=MEDGEMMA_REVISION,
        cache_dir=ROOT / ".hf_cache",
        local_files_only=True,
    )
    batch_size = int(config["generator"]["batch_size"])
    max_tokens = int(config["generator"]["answer_max_new_tokens"])
    stop_token = str(config["generator"]["stop_token"])
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with args.rows_output.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            outputs = generator.generate_batch(
                [str(task["answer_prompt"]) for task in batch],
                [Path(str(task["target_image_path"])) for task in batch],
                max_new_tokens=max_tokens,
                stop_token=stop_token,
            )
            for task, output in zip(batch, outputs, strict=True):
                parsed = parse_plain_answer(str(output["answer"]), stop_token=stop_token)
                parsed["answer"] = normalize_bounded_answer(
                    parsed["answer"], maximum_complete_sentences=2
                )
                evidence = task["evidence"]
                support = deterministic_historical_evidence(
                    evidence,
                    query="\n".join((str(task["question"]), parsed["answer"])),
                    retrieved_case_ids=task["retrieved_case_ids"],
                    maximum_units=3,
                )
                assembled = assemble_deterministic_output(
                    parsed,
                    support,
                    no_reliable_history=bool(task["no_reliable_history"] or task["system"] == "g0_target_image"),
                )
                allowed = {unit.provenance_id for unit in evidence}
                citation_valid = all(
                    item["provenance_id"] in allowed for item in assembled["historical_support"]
                )
                row = {
                    **{key: value for key, value in task.items() if key != "evidence"},
                    "evidence": [serialize_unit(unit) for unit in evidence],
                    "raw_answer_stage": output["answer"],
                    **assembled,
                    "citation_valid": citation_valid,
                    "token_f1": token_f1(assembled["answer"], task["reference_answer"]),
                    "answer_input_tokens": int(output["input_tokens"]),
                    "answer_output_tokens": int(output["output_tokens"]),
                    "hit_answer_token_ceiling": bool(output["hit_token_ceiling"]),
                    "stopped_on_end_of_turn": bool(output["stopped_on_requested_token"]),
                }
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            done = min(start + len(batch), len(pending))
            if done % 80 == 0 or done == len(pending):
                print(json.dumps({"qa_generated": done, "pending_at_start": len(pending)}), flush=True)

    rows = read_jsonl(args.rows_output)
    if len(rows) != len(tasks):
        raise RuntimeError(f"QA output incomplete: {len(rows)} != {len(tasks)}")
    metrics = summarize(rows)
    bootstrap_iterations = int(config["bootstrap_iterations"])
    summary = {
        "study": "V10 compact QA confirmation",
        "status": "confirmation_complete_no_retuning",
        "counts": {
            "test_cases": len({row["case_id"] for row in rows}),
            "qa_rows": len(rows),
            "systems": len(SYSTEMS),
        },
        "metrics": metrics,
        "primary_comparisons": {
            "g2_minus_g0_token_f1": case_grouped_bootstrap_difference(
                rows,
                left="g2_hierarchical",
                right="g0_target_image",
                metric="token_f1",
                iterations=bootstrap_iterations,
                seed=int(config["bootstrap_seed"]),
            ),
            "g2_minus_g1_token_f1": case_grouped_bootstrap_difference(
                rows,
                left="g2_hierarchical",
                right="g1_whole_report",
                metric="token_f1",
                iterations=bootstrap_iterations,
                seed=int(config["bootstrap_seed"]) + 1,
            ),
        },
        "runtime": {
            "elapsed_seconds_this_run": time.perf_counter() - started,
            "peak_gpu_memory_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "batch_size": batch_size,
        },
        "qa_rows_sha256": file_sha256(args.rows_output),
        "retuning_after_test": False,
        "claim_boundary": "Automated report-reference consistency, not physician-adjudicated correctness.",
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
