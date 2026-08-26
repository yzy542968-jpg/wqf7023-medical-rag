"""Run V11 MedGemma generation diagnostics on development/validation only.

This is resumable and deliberately excludes the V10 test partition. It does
not instantiate a V11 confirmation cohort or alter any frozen V10 artifact.
An explicitly selected evidence-policy subset is compared under identical
target images, questions, retrieved Top-3 cases and deterministic output
parsing.
"""

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

import torch


# This audit is intentionally reproducible from the pinned local snapshot.
# Fail fast when a required file is absent instead of allowing Transformers to
# spend several minutes retrying a network request.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.multimodal.v9_generation import (  # noqa: E402
    MEDGEMMA_REVISION,
    MedGemmaImageGenerator,
    select_primary_image,
)
from medical_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from medical_rag.similar_case.v10_evidence import sentence_units  # noqa: E402
from medical_rag.similar_case.v11_evidence import (  # noqa: E402
    evidence_profile,
    select_hierarchical_evidence,
)
from medical_rag.similar_case.v11_output_contract import (  # noqa: E402
    answer_only_generation_prompt,
    assemble_provenance_output,
    bound_complete_sentences,
    compact_generation_prompt,
    parse_compact_output,
)
from medical_rag.similar_case.v11_question_planner import (  # noqa: E402
    plan_question,
    render_planner_instruction,
)
from medical_rag.similar_case.v11_selective import compute_retrieval_confidence  # noqa: E402
from medical_rag.similar_case.v11_qrel import report_index_spectrum  # noqa: E402


QUESTIONS = {
    "findings": "What are the main radiographic findings?",
    "impression": "What is the most likely radiographic impression?",
    "acute": "Is there an acute cardiopulmonary abnormality? Explain briefly.",
}
POLICIES = ("whole_report", "sentence_only", "case_to_fact", "case_to_fact_plus_selective_gate")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialize_units(units: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "provenance_id": unit.provenance_id,
            "case_id": unit.case_id,
            "section": unit.section,
            "unit_type": unit.unit_type,
            "unit_index": unit.unit_index,
            "text": unit.text,
            "source_sha256": unit.source_sha256,
            "score": unit.score,
        }
        for unit in units
    ]


def _whole_report_units(case: Mapping[str, Any]) -> list[Any]:
    case_id = str(case["case_id"])
    return sentence_units(case_id, "findings", case.get("findings")) + sentence_units(
        case_id, "impression", case.get("impression")
    )


def _reference(case: Mapping[str, Any], question_type: str) -> tuple[str, bool]:
    if question_type in {"findings", "impression"}:
        return str(case.get(question_type, "")), False
    # The source cases do not contain a separately adjudicated acute-answer
    # field. Keep the row for generation diagnostics but mark its report-field
    # reference as a proxy rather than presenting it as a gold answer.
    return str(case.get("impression") or case.get("findings") or ""), True


def build_tasks(
    cases: Mapping[str, Mapping[str, Any]],
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    image_root: Path,
    facts_by_case: Mapping[str, Sequence[str]],
    selective_threshold: float,
    policies: Sequence[str] = POLICIES,
) -> list[dict[str, Any]]:
    train_cases = [cases[case_id] for case_id in train_ids]
    retriever = BM25Retriever().fit(train_cases)
    tasks: list[dict[str, Any]] = []
    for case_id in validation_ids:
        source = cases[case_id]
        scores = retriever.score_all(
            "\n".join(part for part in (str(source.get("indication", "")), "radiology report") if part)
        )
        order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
        top_cases = [train_cases[index] for index in order[:3]]
        plan_cache = {name: plan_question(question, str(source.get("indication", ""))) for name, question in QUESTIONS.items()}
        for question_type, question in QUESTIONS.items():
            plan = plan_cache[question_type]
            query = "\n".join(part for part in (str(source.get("indication", "")), question) if part)
            # No facts are needed for the sentence-only condition. This uses
            # the same deterministic within-case ranking budget as the V11
            # evidence selector while keeping the condition sentence-only.
            sentence_only = select_hierarchical_evidence(
                top_cases,
                query=query,
                facts_by_case={},
                plan=plan,
                maximum_units_per_case=3,
                maximum_total_units=9,
                maximum_characters=2000,
            )
            case_to_fact = select_hierarchical_evidence(
                top_cases,
                query=query,
                facts_by_case=facts_by_case,
                plan=plan,
            )
            # The generation audit is intentionally independent of the
            # RadGraph qrel extraction run; it evaluates output contracts and
            # evidence budgets with report/sentence units available locally.
            whole_units = tuple(unit for case in top_cases for unit in _whole_report_units(case))
            confidence = compute_retrieval_confidence(
                scores,
                evidence_coverage=min(1.0, len(case_to_fact.units) / 6.0),
            )
            evidence_by_policy = {
                "whole_report": whole_units,
                "sentence_only": sentence_only.units,
                "case_to_fact": case_to_fact.units,
                "case_to_fact_plus_selective_gate": (
                    case_to_fact.units if confidence.confidence >= selective_threshold else tuple()
                ),
            }
            reference, reference_proxy = _reference(source, question_type)
            for policy in policies:
                evidence = evidence_by_policy[policy]
                no_history = not evidence
                tasks.append(
                    {
                        "case_id": case_id,
                        "question_type": question_type,
                        "question": question,
                        "reference_answer": reference,
                        "reference_is_proxy": reference_proxy,
                        "policy": policy,
                        "retrieved_case_ids": [str(case["case_id"]) for case in top_cases],
                        "target_image_path": str(select_primary_image(source, image_root)),
                        "evidence": evidence,
                        "planner_intent": plan.intent,
                        "planner_instruction": render_planner_instruction(plan),
                        "retrieval_confidence": confidence.confidence,
                        "selective_history_suppressed": policy.endswith("selective_gate") and no_history,
                        "prompt": answer_only_generation_prompt(
                            indication=str(source.get("indication", "")),
                            question=question,
                            planner_instruction=render_planner_instruction(plan),
                            evidence=evidence,
                            abstain=no_history,
                        ),
                    }
                )
    return tasks


def _completed(path: Path) -> set[tuple[str, str, str]]:
    if not path.is_file():
        return set()
    return {(str(row["case_id"]), str(row["question_type"]), str(row["policy"])) for row in read_jsonl(path)}


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values) if values else 0.0


def select_development_cases(
    validation_ids: Sequence[str],
    cases: Mapping[str, Mapping[str, Any]],
    max_cases: int,
    *,
    stratify_spectrum: bool,
) -> list[str]:
    """Select a deterministic development subset without inspecting outcomes."""

    ordered = sorted(
        (str(case_id) for case_id in validation_ids),
        key=lambda case_id: hashlib.sha256(f"v11-medgemma-development|{case_id}".encode("utf-8")).hexdigest(),
    )
    if max_cases <= 0 or max_cases >= len(ordered):
        return ordered
    if not stratify_spectrum:
        return ordered[:max_cases]
    strata: dict[str, list[str]] = defaultdict(list)
    for case_id in ordered:
        strata[report_index_spectrum(cases[case_id])].append(case_id)
    labels = ("report_indexed_normal", "report_indexed_abnormal", "report_index_indeterminate")
    selected: list[str] = []
    quotas = {label: max_cases // 2 for label in labels[:2]}
    remaining = max_cases - sum(quotas.values())
    for label in labels:
        take = min(len(strata[label]), quotas.get(label, 0) + (remaining if label == "report_indexed_normal" else 0))
        selected.extend(strata[label][:take])
        remaining -= max(0, take - quotas.get(label, 0))
    if len(selected) < max_cases:
        used = set(selected)
        selected.extend(case_id for case_id in ordered if case_id not in used)
    return sorted(selected[:max_cases])


def summarize(rows: Sequence[Mapping[str, Any]], policies: Sequence[str] = POLICIES) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        result[policy] = {
            "row_count": len(selected),
            "case_count": len({str(row["case_id"]) for row in selected}),
            "token_f1_all_rows": _mean(selected, "token_f1"),
            "token_f1_non_proxy_questions": _mean([row for row in selected if not row["reference_is_proxy"]], "token_f1"),
            "structured_output_valid_rate": _mean(selected, "structured_output_valid"),
            "raw_json_valid_rate": _mean(selected, "raw_json_valid"),
            "parser_repaired_rate": _mean(selected, "parser_repaired"),
            "normalized_output_usable_rate": _mean(selected, "normalized_output_usable"),
            "answer_only_contract_valid_rate": _mean(selected, "answer_only_contract_valid"),
            "evidence_provenance_valid_rate": _mean(selected, "evidence_provenance_valid"),
            "abstention_rate": _mean(selected, "abstain"),
            "token_ceiling_rate": _mean(selected, "hit_token_ceiling"),
            "mean_input_tokens": _mean(selected, "input_tokens"),
            "mean_output_tokens": _mean(selected, "output_tokens"),
            "mean_evidence_units": _mean(selected, "evidence_unit_count"),
            "mean_evidence_characters": _mean(selected, "evidence_character_count"),
            "mean_latency_seconds": _mean(selected, "latency_seconds"),
            "max_peak_gpu_memory_mib": max((float(row["peak_gpu_memory_mib"]) for row in selected), default=0.0),
            "by_question_type": {
                question_type: {
                    "rows": sum(row["question_type"] == question_type for row in selected),
                    "token_f1": _mean([row for row in selected if row["question_type"] == question_type], "token_f1"),
                    "structured_output_valid_rate": _mean([row for row in selected if row["question_type"] == question_type], "structured_output_valid"),
                }
                for question_type in QUESTIONS
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--facts", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/openi_official_images")
    parser.add_argument("--rows-output", type=Path, default=ROOT / "experiments/v11_development/v11_medgemma_generation_rows.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "data/splits/v11/v11_medgemma_generation_summary.json")
    parser.add_argument("--max-cases", type=int, default=0, help="Optional deterministic prefix for smoke/preflight runs.")
    parser.add_argument("--stratify-spectrum", action="store_true", help="Use a deterministic report-indexed normal/abnormal development sample.")
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=POLICIES,
        default=list(POLICIES),
        help="Explicit generation policies to evaluate; keep the list fixed for a clean audit run.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    split = read_json(args.split)
    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    fact_rows = read_jsonl(args.facts)
    facts_by_case = {
        str(row["case_id"]): tuple(row.get("facts", ()))
        for row in fact_rows
        if row.get("status") == "ok"
    }
    train_ids = [str(value) for value in split["partitions"]["train"]["case_ids"]]
    validation_ids = [str(value) for value in split["partitions"]["validation"]["case_ids"]]
    selected_case_ids = select_development_cases(
        validation_ids,
        cases,
        args.max_cases,
        stratify_spectrum=args.stratify_spectrum,
    )
    validation_ids = selected_case_ids
    prior_summary = ROOT / "data/splits/v11/v11_development_evidence_ablation_summary.json"
    selective_threshold = 0.7142591215182613
    if prior_summary.is_file():
        selective_threshold = float(read_json(prior_summary)["selective_gate"]["fit"]["threshold"])
    policies = tuple(args.policies)
    tasks = build_tasks(
        cases,
        train_ids,
        validation_ids,
        args.image_root,
        facts_by_case,
        selective_threshold,
        policies=policies,
    )
    expected = len(validation_ids) * len(QUESTIONS) * len(policies)
    if len(tasks) != expected:
        raise RuntimeError(f"incomplete V11 generation matrix: {len(tasks)} != {expected}")
    completed = _completed(args.rows_output)
    pending = [task for task in tasks if (task["case_id"], task["question_type"], task["policy"]) not in completed]
    for task in pending:
        image_path = Path(task["target_image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"target image missing for {task['case_id']}: {image_path}")
    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    previous_runtime = None
    if not pending and args.summary_output.is_file():
        previous_runtime = read_json(args.summary_output).get("runtime")
    started = time.perf_counter()
    generator = None
    if pending:
        generator = MedGemmaImageGenerator(
            revision=MEDGEMMA_REVISION,
            cache_dir=ROOT / ".hf_cache",
            local_files_only=True,
        )
        torch.cuda.reset_peak_memory_stats()
    with args.rows_output.open("a", encoding="utf-8", newline="\n") as handle:
        for start in range(0, len(pending), max(1, args.batch_size)):
            batch = pending[start : start + max(1, args.batch_size)]
            batch_start = time.perf_counter()
            outputs = generator.generate_batch(
                [str(task["prompt"]) for task in batch],
                [Path(str(task["target_image_path"])) for task in batch],
                max_new_tokens=args.max_new_tokens,
                stop_token="<end_of_turn>",
            )
            batch_latency = (time.perf_counter() - batch_start) / max(len(batch), 1)
            peak_mib = torch.cuda.max_memory_allocated() / 1024**2
            for task, output in zip(batch, outputs, strict=True):
                evidence = task["evidence"]
                raw_answer = str(output["answer"])
                # V11 development uses answer-only generation and deterministic
                # provenance attachment. Keep the JSON parser diagnostics in
                # the row so the compact-JSON alternative remains auditable.
                parsed = parse_compact_output(raw_answer, evidence, no_history_expected=not evidence)
                answer = bound_complete_sentences(raw_answer)
                if parsed["parser_repaired"] and parsed["answer"]:
                    answer = parsed["answer"]
                answer_only_valid = bool(answer)
                assembled = assemble_provenance_output(
                    answer,
                    evidence=[] if not evidence else evidence[:3],
                    uncertainty="high" if not evidence else "medium",
                    abstain=not evidence,
                )
                allowed = {unit.provenance_id for unit in evidence}
                evidence_valid = all(item in allowed for item in parsed["evidence"])
                row = {
                    "case_id": task["case_id"],
                    "question_type": task["question_type"],
                    "question": task["question"],
                    "reference_answer": task["reference_answer"],
                    "reference_is_proxy": task["reference_is_proxy"],
                    "policy": task["policy"],
                    "retrieved_case_ids": task["retrieved_case_ids"],
                    "target_image_path": task["target_image_path"],
                    "planner_intent": task["planner_intent"],
                    "retrieval_confidence": task["retrieval_confidence"],
                    "selective_history_suppressed": task["selective_history_suppressed"],
                    "evidence": serialize_units(evidence),
                    "raw_output": output["answer"],
                    "parsed_output": parsed,
                    "assembled_output": assembled,
                    "structured_output_valid": float(parsed["structured_output_valid"]),
                    "answer_only_contract_valid": float(answer_only_valid),
                    "raw_json_valid": float(parsed["raw_json_valid"]),
                    "parser_repaired": float(parsed["parser_repaired"]),
                    "normalized_output_usable": float(parsed["normalized_output_usable"]),
                    "evidence_provenance_valid": float(evidence_valid),
                    "abstain": float(not evidence),
                    "token_f1": token_f1(answer, task["reference_answer"]),
                    "evidence_unit_count": len(evidence),
                    "evidence_character_count": sum(len(unit.text) for unit in evidence),
                    "input_tokens": int(output["input_tokens"]),
                    "output_tokens": int(output["output_tokens"]),
                    "hit_token_ceiling": float(output["hit_token_ceiling"]),
                    "stopped_on_requested_token": float(output["stopped_on_requested_token"]),
                    "latency_seconds": batch_latency,
                    "peak_gpu_memory_mib": peak_mib,
                }
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            done = min(start + len(batch), len(pending))
            if done % 20 == 0 or done == len(pending):
                print(json.dumps({"generated_this_run": done, "pending_at_start": len(pending)}), flush=True)

    rows = read_jsonl(args.rows_output)
    if len(rows) != len(tasks):
        raise RuntimeError(f"generation output incomplete: {len(rows)} != {len(tasks)}")
    summary = {
        "study": "v11_medgemma_generation_development",
        "status": "development_only_no_confirmation",
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "facts_sha256": file_sha256(args.facts),
            "split_sha256": file_sha256(args.split),
            "train_case_count": len(train_ids),
            "validation_case_count": len(validation_ids),
            "validation_case_ids_sha256": hashlib.sha256("\n".join(sorted(validation_ids)).encode("utf-8")).hexdigest(),
            "case_selection": {
                "rule": "sha256(v11-medgemma-development|case_id) within report-indexed spectrum strata",
                "stratified": bool(args.stratify_spectrum),
                "report_indexed_spectrum_counts": {
                    spectrum: sum(report_index_spectrum(cases[case_id]) == spectrum for case_id in validation_ids)
                    for spectrum in ("report_indexed_normal", "report_indexed_abnormal", "report_index_indeterminate")
                },
            },
            "selective_threshold_source": str(prior_summary.relative_to(ROOT)) if prior_summary.is_file() else "protocol_default",
            "model": {
                "name": "google/medgemma-1.5-4b-it",
                "revision": MEDGEMMA_REVISION,
                "local_files_only": True,
            },
        },
        "counts": {"rows": len(rows), "questions_per_case": len(QUESTIONS), "policies": list(policies)},
        "metrics": summarize(rows, policies=policies),
        "generation_rows_sha256": file_sha256(args.rows_output),
        "runtime": previous_runtime or {
            "elapsed_seconds_this_run": time.perf_counter() - started,
            "max_peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
        },
        "output_mode": "answer_only_generation_plus_deterministic_provenance",
        "claim_boundary": "Development-only answer-contract and same-source report-reference diagnostics; not clinical correctness, safety, human review, or external validation.",
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
