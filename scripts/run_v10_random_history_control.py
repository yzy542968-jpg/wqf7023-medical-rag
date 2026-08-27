from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.multimodal.v9_generation import (  # noqa: E402
    MEDGEMMA_REVISION,
    MedGemmaImageGenerator,
    select_primary_image,
)
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.random_history_control import select_random_history  # noqa: E402
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
from run_v10_evidence_generation_development import (  # noqa: E402
    read_json,
    read_jsonl,
    serialize_unit,
)


DEFAULT_CONFIG = ROOT / "config/v10_confirmation.json"
DEFAULT_CASES = ROOT / "data/processed/openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data/processed/v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data/splits/v10/v10_cluster_disjoint_split.json"
DEFAULT_EMBEDDINGS = ROOT / "data/processed/v10_medsiglip_embeddings.npz"
DEFAULT_CALIBRATOR = ROOT / "artifacts/v10/retrieval_calibrator.json"
DEFAULT_RETRIEVAL = ROOT / "experiments/v10_publication/v10_confirmation_retrieval_rows.jsonl"
DEFAULT_RETRIEVAL_SUMMARY = ROOT / "data/splits/v10/v10_confirmation_retrieval_summary.json"
DEFAULT_QA_ROWS = ROOT / "experiments/v10_publication/v10_confirmation_qa_rows.jsonl"
DEFAULT_QA_SUMMARY = ROOT / "data/splits/v10/v10_confirmation_qa_summary.json"
DEFAULT_IMAGE_ROOT = ROOT / "data/raw/openi_official_images"
DEFAULT_ROWS = ROOT / "experiments/v10_publication/v10_random_history_control_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "data/splits/v10/v10_random_history_generation_summary.json"
PROTOCOL_COMMIT = "183c5e8"
ASSIGNMENT_COUNT = 5
SELECTION_SEED = 7131


def completed_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.is_file():
        return set()
    keys = []
    for row in read_jsonl(path):
        keys.append(
            (str(row["case_id"]), str(row["question_type"]), int(row["assignment"]))
        )
    if len(keys) != len(set(keys)):
        raise RuntimeError("Random-history output contains duplicate task keys")
    return set(keys)


def usable_report(case: Mapping[str, Any]) -> bool:
    return bool(
        "\n".join((str(case.get("findings") or ""), str(case.get("impression") or ""))).strip()
    )


def selection_fingerprint(tasks: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "|".join(
            (
                str(task["case_id"]),
                str(task["question_type"]),
                str(task["assignment"]),
                ",".join(map(str, task["retrieved_case_ids"])),
            )
        )
        for task in sorted(
            tasks,
            key=lambda row: (
                str(row["case_id"]),
                str(row["question_type"]),
                int(row["assignment"]),
            ),
        )
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def build_tasks(
    *,
    cases: Mapping[str, Mapping[str, Any]],
    radgraph: Mapping[str, Any],
    split: Mapping[str, Any],
    retrieval_rows: Sequence[Mapping[str, Any]],
    image_root: Path,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    r4 = {
        (str(row["case_id"]), str(row["question_type"])): row
        for row in retrieval_rows
        if row["system"] == "r4_nine_feature"
    }
    r5 = {
        (str(row["case_id"]), str(row["question_type"])): row
        for row in retrieval_rows
        if row["system"] == "r5_fact_attention"
    }
    if set(r4) != set(r5):
        raise RuntimeError("R4 and R5 retrieval coverage differs")
    train_ids = tuple(
        sorted(
            case_id
            for case_id in map(str, split["partitions"]["train"]["case_ids"])
            if case_id in cases
            and case_id in radgraph
            and radgraph[case_id].status == "ok"
            and usable_report(cases[case_id])
        )
    )
    if len(train_ids) != 2506:
        raise RuntimeError(f"Unexpected technically eligible Train bank: {len(train_ids)}")

    tasks: list[dict[str, Any]] = []
    for key in sorted(r5):
        case_id, question_type = key
        if question_type not in {"findings", "impression"}:
            continue
        target = cases[case_id]
        question = QUESTIONS[question_type]
        excluded = {
            *map(str, r4[key]["top_case_ids"][:10]),
            *map(str, r5[key]["top_case_ids"][:10]),
        }
        for assignment in range(ASSIGNMENT_COUNT):
            selected_ids = select_random_history(
                train_ids,
                seed=SELECTION_SEED,
                assignment=assignment,
                target_case_id=case_id,
                question_type=question_type,
                excluded_case_ids=excluded,
                count=3,
            )
            evidence = []
            for selected_id in selected_ids:
                evidence.extend(
                    select_case_evidence(
                        cases[selected_id],
                        query="\n".join((str(target.get("indication", "")), question)),
                        facts=radgraph[selected_id].facts,
                        policy="whole_report",
                    )
                )
            tasks.append(
                {
                    "case_id": case_id,
                    "question_type": question_type,
                    "question": question,
                    "reference_answer": str(target[question_type]),
                    "system": "gr_random_history",
                    "assignment": assignment,
                    "retrieved_case_ids": list(selected_ids),
                    "excluded_r4_r5_top10_count": len(excluded),
                    "evidence": evidence,
                    "target_image_path": str(select_primary_image(target, image_root)),
                    "answer_prompt": build_plain_answer_prompt(
                        indication=str(target.get("indication", "")),
                        question=question,
                        evidence=evidence,
                        no_reliable_history=False,
                    ),
                }
            )
    expected = 568 * 2 * ASSIGNMENT_COUNT
    if len(tasks) != expected:
        raise RuntimeError(f"Random-history task matrix incomplete: {len(tasks)} != {expected}")
    return tasks, train_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the V10 random-history negative control.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    parser.add_argument("--retrieval-rows", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--retrieval-summary", type=Path, default=DEFAULT_RETRIEVAL_SUMMARY)
    parser.add_argument("--qa-rows", type=Path, default=DEFAULT_QA_ROWS)
    parser.add_argument("--qa-summary", type=Path, default=DEFAULT_QA_SUMMARY)
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
    retrieval_summary = read_json(args.retrieval_summary)
    qa_summary = read_json(args.qa_summary)
    if retrieval_summary.get("status") != "confirmation_complete_no_retuning":
        raise RuntimeError("V10 retrieval confirmation is not complete")
    if qa_summary.get("status") != "confirmation_complete_no_retuning":
        raise RuntimeError("V10 QA confirmation is not complete")
    if file_sha256(args.retrieval_rows) != retrieval_summary["retrieval_rows_sha256"]:
        raise RuntimeError("V10 retrieval rows differ from their frozen summary")
    if file_sha256(args.qa_rows) != qa_summary["qa_rows_sha256"]:
        raise RuntimeError("V10 QA rows differ from their frozen summary")

    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    radgraph = read_radgraph_case_records(args.radgraph)
    split = read_json(args.split)
    tasks, train_ids = build_tasks(
        cases=cases,
        radgraph=radgraph,
        split=split,
        retrieval_rows=read_jsonl(args.retrieval_rows),
        image_root=args.image_root,
    )
    completed = completed_keys(args.rows_output)
    pending = [
        task
        for task in tasks
        if (task["case_id"], task["question_type"], task["assignment"]) not in completed
    ]
    pending.sort(
        key=lambda task: (
            int(task["assignment"]),
            len(str(task["answer_prompt"])),
            str(task["case_id"]),
            str(task["question_type"]),
        )
    )

    started = time.perf_counter()
    peak_memory_mib = 0.0
    if pending:
        generator = MedGemmaImageGenerator(
            revision=MEDGEMMA_REVISION,
            cache_dir=ROOT / ".hf_cache",
            local_files_only=True,
        )
        batch_size = int(config["generator"]["batch_size"])
        max_tokens = int(config["generator"]["answer_max_new_tokens"])
        stop_token = str(config["generator"]["stop_token"])
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
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
                        parsed, support, no_reliable_history=False
                    )
                    allowed = {unit.provenance_id for unit in evidence}
                    row = {
                        **{key: value for key, value in task.items() if key != "evidence"},
                        "evidence": [serialize_unit(unit) for unit in evidence],
                        "raw_answer_stage": output["answer"],
                        **assembled,
                        "citation_valid": all(
                            item["provenance_id"] in allowed
                            for item in assembled["historical_support"]
                        ),
                        "token_f1": token_f1(assembled["answer"], task["reference_answer"]),
                        "answer_input_tokens": int(output["input_tokens"]),
                        "answer_output_tokens": int(output["output_tokens"]),
                        "hit_answer_token_ceiling": bool(output["hit_token_ceiling"]),
                        "stopped_on_end_of_turn": bool(output["stopped_on_requested_token"]),
                    }
                    handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                handle.flush()
                done = min(start + len(batch), len(pending))
                if done % 160 == 0 or done == len(pending):
                    print(
                        json.dumps(
                            {"random_history_generated": done, "pending_at_start": len(pending)}
                        ),
                        flush=True,
                    )
        peak_memory_mib = torch.cuda.max_memory_allocated() / 1024**2

    rows = read_jsonl(args.rows_output)
    if len(rows) != len(tasks):
        raise RuntimeError(f"Random-history output incomplete: {len(rows)} != {len(tasks)}")
    if len(completed_keys(args.rows_output)) != len(tasks):
        raise RuntimeError("Random-history output keys are incomplete")
    assignment_metrics = {}
    for assignment in range(ASSIGNMENT_COUNT):
        selected = [row for row in rows if int(row["assignment"]) == assignment]
        assignment_metrics[str(assignment)] = {
            "row_count": len(selected),
            "case_count": len({row["case_id"] for row in selected}),
            "token_f1": statistics.fmean(float(row["token_f1"]) for row in selected),
            "answer_token_ceiling_rate": statistics.fmean(
                float(row["hit_answer_token_ceiling"]) for row in selected
            ),
            "answer_contract_valid_rate": statistics.fmean(
                float(row["assembled_schema_valid"]) for row in selected
            ),
            "citation_valid_rate": statistics.fmean(
                float(row["citation_valid"]) for row in selected
            ),
        }
    summary = {
        "study": "V10 post-hoc random-history negative control generation",
        "status": "posthoc_control_generation_complete_no_retuning",
        "protocol_commit": PROTOCOL_COMMIT,
        "counts": {
            "train_bank_cases": len(train_ids),
            "test_cases": len({row["case_id"] for row in rows}),
            "assignments": ASSIGNMENT_COUNT,
            "rows": len(rows),
        },
        "selection": {
            "seed": SELECTION_SEED,
            "domain": "v10-random-history",
            "reports_per_query": 3,
            "excluded_formal_retrieval_depth_per_system": 10,
            "selection_fingerprint": selection_fingerprint(tasks),
        },
        "generator": {
            **config["generator"],
            "revision": MEDGEMMA_REVISION,
        },
        "assignment_metrics": assignment_metrics,
        "runtime": {
            "elapsed_seconds_this_run": time.perf_counter() - started,
            "peak_gpu_memory_allocated_mib": peak_memory_mib,
        },
        "source_hashes": {
            "config": file_sha256(args.config),
            "split": file_sha256(args.split),
            "retrieval_rows": file_sha256(args.retrieval_rows),
            "qa_rows": file_sha256(args.qa_rows),
            "output_rows": file_sha256(args.rows_output),
            "script": file_sha256(Path(__file__)),
        },
        "claim_boundary": (
            "Post-hoc automated random-history control; not physician-adjudicated clinical utility."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

