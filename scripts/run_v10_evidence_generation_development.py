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

from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.multimodal.v9_generation import (  # noqa: E402
    MEDGEMMA_REVISION,
    MedGemmaImageGenerator,
    select_primary_image,
)
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v10_evidence import (  # noqa: E402
    EvidenceUnit,
    select_case_evidence,
)
from medical_rag.similar_case.v10_generation import (  # noqa: E402
    assemble_output,
    build_answer_prompt,
    build_support_prompt,
    parse_answer_stage,
    parse_support_stage,
)
from medical_rag.similar_case.v10_runtime import QUESTIONS  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v10_evidence_generation_development.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_RANKINGS = ROOT / "experiments" / "v10_publication" / "v10_r5_multiview_integration_rows.jsonl"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_ROWS = ROOT / "experiments" / "v10_publication" / "v10_evidence_generation_validation_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_evidence_generation_development_summary.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def serialize_unit(unit: EvidenceUnit) -> dict[str, Any]:
    return {
        "case_id": unit.case_id,
        "section": unit.section,
        "unit_type": unit.unit_type,
        "unit_index": unit.unit_index,
        "text": unit.text,
        "source_sha256": unit.source_sha256,
        "score": unit.score,
        "provenance_id": unit.provenance_id,
    }


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    return {
        (str(row["case_id"]), str(row["question_type"]), str(row["evidence_policy"]))
        for row in read_jsonl(path)
    }


def build_tasks(
    cases: Mapping[str, Mapping[str, Any]],
    rankings: Sequence[Mapping[str, Any]],
    radgraph: Mapping[str, Any],
    policies: Sequence[str],
    question_types: Sequence[str],
    image_root: Path,
) -> list[dict[str, Any]]:
    tasks = []
    for ranking in rankings:
        question_type = str(ranking["question_type"])
        if question_type not in question_types:
            continue
        case_id = str(ranking["case_id"])
        source = cases[case_id]
        question = QUESTIONS[question_type]
        top_ids = list(ranking["top3"]["learned_attention"])
        if len(top_ids) != 3:
            raise RuntimeError(f"ranking for {case_id}:{question_type} does not contain Top-3")
        for policy in policies:
            evidence = []
            for retrieved_id in top_ids:
                record = radgraph[retrieved_id]
                evidence.extend(
                    select_case_evidence(
                        cases[retrieved_id],
                        query="\n".join(
                            part for part in (str(source.get("indication", "")), question) if part
                        ),
                        facts=record.facts,
                        policy=policy,
                    )
                )
            tasks.append(
                {
                    "case_id": case_id,
                    "question_type": question_type,
                    "question": question,
                    "reference_answer": str(source.get(question_type, "")),
                    "evidence_policy": policy,
                    "retrieved_case_ids": top_ids,
                    "target_image_path": str(select_primary_image(source, image_root)),
                    "evidence": evidence,
                    "answer_prompt": build_answer_prompt(
                        indication=str(source.get("indication", "")),
                        question=question,
                        evidence=evidence,
                        no_reliable_history=False,
                    ),
                }
            )
    return tasks


def summarize(rows: Sequence[Mapping[str, Any]], policies: Sequence[str]) -> dict[str, Any]:
    result = {}
    case_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        case_values[str(row["evidence_policy"])][str(row["case_id"])].append(float(row["token_f1"]))
    for policy in policies:
        selected = [row for row in rows if row["evidence_policy"] == policy]
        result[policy] = {
            "row_count": len(selected),
            "case_count": len({str(row["case_id"]) for row in selected}),
            "token_f1_equal_question": statistics.fmean(float(row["token_f1"]) for row in selected),
            "token_f1_case_averaged": statistics.fmean(
                statistics.fmean(values) for values in case_values[policy].values()
            ),
            "assembled_schema_valid_rate": statistics.fmean(
                float(row["assembled_schema_valid"]) for row in selected
            ),
            "citation_valid_rate": statistics.fmean(float(row["citation_valid"]) for row in selected),
            "answer_stage_valid_rate": statistics.fmean(
                float(row["answer_stage_valid"]) for row in selected
            ),
            "support_stage_valid_rate": statistics.fmean(
                float(row["support_stage_valid"]) for row in selected
            ),
            "mean_evidence_units": statistics.fmean(int(row["evidence_unit_count"]) for row in selected),
            "mean_evidence_characters": statistics.fmean(
                int(row["evidence_character_count"]) for row in selected
            ),
            "mean_answer_input_tokens": statistics.fmean(
                int(row["answer_input_tokens"]) for row in selected
            ),
            "mean_total_output_tokens": statistics.fmean(
                int(row["answer_output_tokens"]) + int(row["support_output_tokens"])
                for row in selected
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


def select_policy(metrics: Mapping[str, Mapping[str, float]], config: Mapping[str, Any]) -> str:
    eligible = [
        policy
        for policy, values in metrics.items()
        if values["assembled_schema_valid_rate"] >= float(config["minimum_assembled_schema_valid_rate"])
        and values["citation_valid_rate"] >= float(config["minimum_citation_valid_rate"])
    ]
    if not eligible:
        raise RuntimeError("no evidence policy met frozen structural validity constraints")
    best = max(float(metrics[policy]["token_f1_equal_question"]) for policy in eligible)
    tied = {
        policy
        for policy in eligible
        if best - float(metrics[policy]["token_f1_equal_question"])
        < float(config["material_token_f1_difference"])
    }
    return next(policy for policy in config["compactness_tie_order"] if policy in tied)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V10 compact generation development.")
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
        raise RuntimeError("generation config records forbidden outcomes")
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
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    generator = MedGemmaImageGenerator(
        revision=MEDGEMMA_REVISION,
        cache_dir=ROOT / ".hf_cache",
        local_files_only=True,
    )
    batch_size = int(config["generator"]["batch_size"])
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with args.rows_output.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            answer_outputs = generator.generate_batch(
                [str(task["answer_prompt"]) for task in batch],
                [Path(str(task["target_image_path"])) for task in batch],
                max_new_tokens=int(config["generator"]["answer_max_new_tokens"]),
            )
            answer_stages = [parse_answer_stage(str(output["answer"])) for output in answer_outputs]
            support_prompts = [
                build_support_prompt(stage["answer"], task["evidence"])
                for task, stage in zip(batch, answer_stages, strict=True)
            ]
            support_outputs = generator.generate_text_batch(
                support_prompts,
                max_new_tokens=int(config["generator"]["support_max_new_tokens"]),
            )
            for task, answer_output, answer_stage, support_prompt, support_output in zip(
                batch,
                answer_outputs,
                answer_stages,
                support_prompts,
                support_outputs,
                strict=True,
            ):
                support_stage = parse_support_stage(str(support_output["answer"]), task["evidence"])
                assembled = assemble_output(answer_stage, support_stage, no_reliable_history=False)
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
                    "support_prompt": support_prompt,
                    "raw_answer_stage": answer_output["answer"],
                    "raw_support_stage": support_output["answer"],
                    **assembled,
                    "citation_valid": citation_valid,
                    "token_f1": token_f1(assembled["answer"], task["reference_answer"]),
                    "evidence_unit_count": len(task["evidence"]),
                    "evidence_character_count": sum(len(unit.text) for unit in task["evidence"]),
                    "answer_input_tokens": int(answer_output["input_tokens"]),
                    "answer_output_tokens": int(answer_output["output_tokens"]),
                    "support_input_tokens": int(support_output["input_tokens"]),
                    "support_output_tokens": int(support_output["output_tokens"]),
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
    selected = select_policy(metrics, config)
    elapsed = time.perf_counter() - started
    summary = {
        "study": "V10 evidence and compact generation development",
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
        "selected_evidence_policy": selected,
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
