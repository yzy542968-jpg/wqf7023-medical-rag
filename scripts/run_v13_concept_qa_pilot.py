from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_v13_target_concepts import spectrum  # noqa: E402
from evaluate_v13_target_concepts import predict  # noqa: E402
from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.evaluation.chexbert_pathology import CHEXBERT_LABELS  # noqa: E402
from medical_rag.multimodal.v9_generation import (  # noqa: E402
    MEDGEMMA_REVISION,
    MedGemmaImageGenerator,
)
from medical_rag.similar_case.v10_evidence import sentence_units  # noqa: E402
from medical_rag.similar_case.v10_generation import (  # noqa: E402
    assemble_deterministic_output,
    deterministic_historical_evidence,
    normalize_bounded_answer,
    parse_plain_answer,
)
from medical_rag.similar_case.v10_runtime import QUESTIONS  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from medical_rag.similar_case.v11_output_contract import (  # noqa: E402
    answer_only_generation_prompt,
)
from medical_rag.similar_case.v11_question_planner import (  # noqa: E402
    plan_question,
    render_planner_instruction,
)
from run_v10_evidence_generation_development import (  # noqa: E402
    read_json,
    read_jsonl,
    serialize_unit,
)


DEFAULT_CASES = ROOT / "data/processed/openi_cases.jsonl"
DEFAULT_RANKINGS = (
    ROOT / "experiments/v12_optimization/retrieval/v12_validation_ranking_rows.jsonl"
)
DEFAULT_EMBEDDINGS = ROOT / "data/processed/v10_medsiglip_embeddings.npz"
DEFAULT_DECISION = ROOT / "data/splits/v13/v13_target_concept_decision.json"
DEFAULT_MANIFEST = ROOT / "data/splits/v13/v13_concept_qa_manifest.jsonl"
DEFAULT_MANIFEST_SUMMARY = ROOT / "data/splits/v13/v13_concept_qa_manifest_summary.json"
DEFAULT_IMAGE_ROOT = ROOT / "data/raw/openi_official_images"
DEFAULT_ROWS = ROOT / "experiments/v13_target_concept/v13_concept_qa_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "data/splits/v13/v13_concept_qa_generation_summary.json"
CONDITIONS = ("concept_off", "concept_on")
QUESTION_TYPES = ("findings", "impression")
MAX_NEW_TOKENS = 96


def whole_report_units(cases: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    units: list[Any] = []
    for case in cases:
        case_id = str(case["case_id"])
        units.extend(sentence_units(case_id, "findings", case.get("findings")))
        units.extend(sentence_units(case_id, "impression", case.get("impression")))
    return tuple(units)


def threshold_passing_concepts(
    probabilities: Sequence[float],
    thresholds: Sequence[float],
    *,
    maximum: int = 5,
) -> tuple[tuple[str, float], ...]:
    if len(probabilities) != len(CHEXBERT_LABELS) or len(thresholds) != len(
        CHEXBERT_LABELS
    ):
        raise ValueError("V13 concept vectors must follow the 14-label order")
    selected = [
        (CHEXBERT_LABELS[index], float(probability))
        for index, (probability, threshold) in enumerate(zip(probabilities, thresholds))
        if float(probability) >= float(threshold)
    ]
    if any(label != "No Finding" for label, _ in selected):
        selected = [(label, score) for label, score in selected if label != "No Finding"]
    selected.sort(key=lambda value: (-value[1], value[0]))
    return tuple(selected[:maximum])


def concept_instruction(concepts: Sequence[tuple[str, float]]) -> str:
    if not concepts:
        return (
            "Automated target-image hypothesis: no confident concept was predicted. "
            "Do not infer normality from this absence."
        )
    labels = "; ".join(label for label, _ in concepts)
    return (
        "Automated target-image hypotheses (unverified): "
        f"{labels}. Treat these only as uncertain image-derived cues."
    )


def add_concept_instruction(prompt: str, instruction: str) -> str:
    lines = str(prompt).splitlines()
    insertion = next(
        (index for index, line in enumerate(lines) if line.startswith("Indication:")),
        len(lines),
    )
    lines.insert(insertion, instruction)
    return "\n".join(lines)


def task_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["case_id"]), str(row["question_type"]), str(row["condition"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired V13 concept-on/off QA.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manifest-summary", type=Path, default=DEFAULT_MANIFEST_SUMMARY)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    manifest = read_jsonl(args.manifest)
    manifest_summary = read_json(args.manifest_summary)
    if manifest_summary.get("status") != "selection_frozen_before_generation":
        raise RuntimeError("V13 QA manifest is not frozen")
    if file_sha256(args.manifest) != manifest_summary["artifacts"]["manifest_rows_sha256"]:
        raise RuntimeError("V13 QA manifest rows differ from their summary")
    if len(manifest) != 96:
        raise RuntimeError("V13 QA manifest must contain 96 cases")

    selected_ids = [str(row["case_id"]) for row in manifest]
    rankings = {
        (str(row["case_id"]), str(row["question_type"])): row
        for row in read_jsonl(args.rankings)
        if str(row["case_id"]) in set(selected_ids)
        and str(row["question_type"]) in QUESTION_TYPES
    }
    expected_ranking_keys = {
        (case_id, question_type)
        for case_id in selected_ids
        for question_type in QUESTION_TYPES
    }
    if set(rankings) != expected_ranking_keys:
        raise RuntimeError("V13 QA ranking coverage is incomplete")

    decision = read_json(args.decision)
    checkpoint_path = ROOT / decision["selected_checkpoint"]["path"]
    if file_sha256(checkpoint_path) != decision["selected_checkpoint"]["sha256"]:
        raise RuntimeError("V13 concept checkpoint hash differs")
    with np.load(args.embeddings, allow_pickle=False) as encoded:
        embedding_ids = [str(value) for value in encoded["case_ids"].tolist()]
        embedding_matrix = np.asarray(encoded["case_image_embeddings"], dtype=np.float32)
    embedding_by_id = {
        case_id: embedding_matrix[index] for index, case_id in enumerate(embedding_ids)
    }
    selected_x = np.stack([embedding_by_id[case_id] for case_id in selected_ids])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    concept_probabilities, thresholds, _ = predict(
        checkpoint_path, selected_x, device=device
    )
    concepts_by_case = {
        case_id: threshold_passing_concepts(concept_probabilities[index], thresholds)
        for index, case_id in enumerate(selected_ids)
    }

    tasks: list[dict[str, Any]] = []
    for manifest_row in manifest:
        case_id = str(manifest_row["case_id"])
        target = cases[case_id]
        for question_type in QUESTION_TYPES:
            question = QUESTIONS[question_type]
            ranking = rankings[(case_id, question_type)]
            retrieved_ids = [
                str(value) for value in ranking["rankings"]["rrf_lambdamart"][:3]
            ]
            evidence = whole_report_units([cases[value] for value in retrieved_ids])
            plan = plan_question(question, str(target.get("indication") or ""))
            base_prompt = answer_only_generation_prompt(
                indication=str(target.get("indication") or ""),
                question=question,
                planner_instruction=render_planner_instruction(plan),
                evidence=evidence,
                abstain=not evidence,
            )
            concept_line = concept_instruction(concepts_by_case[case_id])
            for condition in CONDITIONS:
                tasks.append(
                    {
                        "case_id": case_id,
                        "spectrum": spectrum(target),
                        "question_type": question_type,
                        "question": question,
                        "condition": condition,
                        "reference_answer": str(target.get(question_type) or ""),
                        "target_image_path": str(manifest_row["target_image_path"]),
                        "retrieved_case_ids": retrieved_ids,
                        "predicted_concepts": [
                            {"label": label, "score": score}
                            for label, score in concepts_by_case[case_id]
                        ],
                        "concept_instruction": concept_line if condition == "concept_on" else "",
                        "evidence": evidence,
                        "prompt": (
                            add_concept_instruction(base_prompt, concept_line)
                            if condition == "concept_on"
                            else base_prompt
                        ),
                    }
                )

    expected = 96 * len(QUESTION_TYPES) * len(CONDITIONS)
    if len(tasks) != expected or len({task_key(task) for task in tasks}) != expected:
        raise RuntimeError("V13 concept QA task matrix is incomplete")
    completed = (
        {task_key(row) for row in read_jsonl(args.rows_output)}
        if args.rows_output.exists()
        else set()
    )
    pending = [task for task in tasks if task_key(task) not in completed]
    pending.sort(key=lambda row: (len(row["prompt"]), *task_key(row)))

    started = time.perf_counter()
    peak_memory_mib = 0.0
    if pending:
        generator = MedGemmaImageGenerator(
            revision=MEDGEMMA_REVISION,
            cache_dir=ROOT / ".hf_cache",
            local_files_only=True,
        )
        torch.cuda.reset_peak_memory_stats()
        args.rows_output.parent.mkdir(parents=True, exist_ok=True)
        with args.rows_output.open("a", encoding="utf-8", newline="\n") as handle:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                outputs = generator.generate_batch(
                    [str(task["prompt"]) for task in batch],
                    [Path(str(task["target_image_path"])) for task in batch],
                    max_new_tokens=MAX_NEW_TOKENS,
                    stop_token="<end_of_turn>",
                )
                for task, output in zip(batch, outputs, strict=True):
                    parsed = parse_plain_answer(str(output["answer"]))
                    parsed["answer"] = normalize_bounded_answer(
                        parsed["answer"], maximum_complete_sentences=2
                    )
                    support = deterministic_historical_evidence(
                        task["evidence"],
                        query="\n".join((task["question"], parsed["answer"])),
                        retrieved_case_ids=task["retrieved_case_ids"],
                        maximum_units=3,
                    )
                    assembled = assemble_deterministic_output(
                        parsed, support, no_reliable_history=False
                    )
                    allowed = {unit.provenance_id for unit in task["evidence"]}
                    row = {
                        **{
                            key: value
                            for key, value in task.items()
                            if key not in {"evidence", "prompt"}
                        },
                        "evidence": [serialize_unit(unit) for unit in task["evidence"]],
                        "raw_answer_stage": str(output["answer"]),
                        **assembled,
                        "citation_valid": all(
                            item["provenance_id"] in allowed
                            for item in assembled["historical_support"]
                        ),
                        "token_f1": token_f1(
                            assembled["answer"], task["reference_answer"]
                        ),
                        "input_tokens": int(output["input_tokens"]),
                        "output_tokens": int(output["output_tokens"]),
                        "hit_token_ceiling": bool(output["hit_token_ceiling"]),
                    }
                    handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
                handle.flush()
                done = min(start + len(batch), len(pending))
                if done % 32 == 0 or done == len(pending):
                    print(
                        json.dumps(
                            {"generated_this_run": done, "pending_at_start": len(pending)}
                        ),
                        flush=True,
                    )
        peak_memory_mib = torch.cuda.max_memory_allocated() / 1024**2

    rows = read_jsonl(args.rows_output)
    if len(rows) != expected or len({task_key(row) for row in rows}) != expected:
        raise RuntimeError("V13 concept QA rows are incomplete")
    metrics = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        metrics[condition] = {
            "rows": len(selected),
            "cases": len({row["case_id"] for row in selected}),
            "token_f1": statistics.fmean(float(row["token_f1"]) for row in selected),
            "answer_contract_valid_rate": statistics.fmean(
                float(row["assembled_schema_valid"]) for row in selected
            ),
            "citation_valid_rate": statistics.fmean(
                float(row["citation_valid"]) for row in selected
            ),
            "token_ceiling_rate": statistics.fmean(
                float(row["hit_token_ceiling"]) for row in selected
            ),
            "mean_input_tokens": statistics.fmean(
                float(row["input_tokens"]) for row in selected
            ),
            "mean_output_tokens": statistics.fmean(
                float(row["output_tokens"]) for row in selected
            ),
        }
    summary = {
        "study": "V13 paired concept-on/off QA generation",
        "status": "validation_generation_complete_no_retuning",
        "test_outcomes_inspected": False,
        "counts": {"cases": 96, "questions_per_case": 2, "rows": len(rows)},
        "conditions": list(CONDITIONS),
        "metrics": metrics,
        "model": {
            "generator": "google/medgemma-1.5-4b-it",
            "revision": MEDGEMMA_REVISION,
            "max_new_tokens": MAX_NEW_TOKENS,
            "concept_checkpoint_sha256": file_sha256(checkpoint_path),
        },
        "runtime": {
            "elapsed_seconds_this_run": time.perf_counter() - started,
            "peak_gpu_memory_allocated_mib": peak_memory_mib,
            "batch_size": args.batch_size,
        },
        "artifacts": {
            "manifest_sha256": file_sha256(args.manifest),
            "ranking_rows_sha256": file_sha256(args.rankings),
            "output_rows_sha256": file_sha256(args.rows_output),
            "script_sha256": file_sha256(Path(__file__)),
        },
        "claim_boundary": (
            "Validation-only automated answer-reference consistency; not diagnostic "
            "accuracy, clinical safety, physician utility, or confirmation evidence."
        ),
    }
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

