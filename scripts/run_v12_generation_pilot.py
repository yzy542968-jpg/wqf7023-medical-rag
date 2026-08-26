"""Run a bounded V12 generation pilot on the preselected V11 Validation cases.

This experiment is deliberately isolated from the frozen V10/V11 results.  It
reuses the already selected 48-case Validation sample, replaces the historical
Top-3 source with the saved V12 LambdaMART ranking, and compares whole-report
against case-to-fact evidence under an answer-only output contract.  It never
reads or instantiates the Test partition.
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
from medical_rag.similar_case.v10_evidence import sentence_units  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from medical_rag.similar_case.v11_evidence import select_hierarchical_evidence  # noqa: E402
from medical_rag.similar_case.v11_output_contract import (  # noqa: E402
    answer_only_generation_prompt,
    bound_complete_sentences,
)
from medical_rag.similar_case.v11_question_planner import (  # noqa: E402
    plan_question,
    render_planner_instruction,
)
from medical_rag.similar_case.v11_selective import compute_retrieval_confidence  # noqa: E402


QUESTIONS = {
    "findings": "What are the main radiographic findings?",
    "impression": "What is the most likely radiographic impression?",
    "acute": "Is there an acute cardiopulmonary abnormality? Explain briefly.",
}
POLICIES = ("whole_report", "case_to_fact")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_ids(values: Sequence[str]) -> str:
    payload = "\n".join(sorted({str(value).strip() for value in values}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def reference_for(case: Mapping[str, Any], question_type: str) -> tuple[str, bool]:
    if question_type in {"findings", "impression"}:
        return str(case.get(question_type, "")), False
    # The source does not provide an independently adjudicated acute answer.
    return str(case.get("impression") or case.get("findings") or ""), True


def whole_report_units(cases: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    units: list[Any] = []
    for case in cases:
        case_id = str(case["case_id"])
        units.extend(sentence_units(case_id, "findings", case.get("findings")))
        units.extend(sentence_units(case_id, "impression", case.get("impression")))
    return tuple(units)


def clean_answer(raw: str) -> str:
    answer = bound_complete_sentences(str(raw or ""))
    return answer.strip()


def output_contract_diagnostic(raw: str, answer: str, *, hit_ceiling: bool) -> dict[str, float]:
    text = str(raw or "").strip()
    lower = text.lower()
    forbidden = any(token in lower for token in ("\"answer\"", "\"evidence\"", "uncertainty:", "analysis:"))
    return {
        "answer_nonempty": float(bool(answer)),
        "answer_only_contract_valid": float(bool(answer) and not forbidden),
        "serialization_leak_detected": float(forbidden),
        "hit_token_ceiling": float(hit_ceiling),
    }


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    first_policy: str,
    second_policy: str,
    metric: str,
    *,
    iterations: int = 10_000,
    seed: int = 1212,
) -> dict[str, float | int | bool]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["case_id"])][str(row["policy"])].append(float(row[metric]))
    case_ids = sorted(grouped)
    differences = [
        mean(grouped[case_id][first_policy]) - mean(grouped[case_id][second_policy])
        for case_id in case_ids
        if grouped[case_id].get(first_policy) and grouped[case_id].get(second_policy)
    ]
    if not differences:
        raise RuntimeError("paired generation comparison has no complete cases")
    import numpy as np

    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(iterations, len(values)))].mean(axis=1)
    low = float(np.quantile(sampled, 0.025))
    high = float(np.quantile(sampled, 0.975))
    return {
        "case_count": len(values),
        "mean_difference": float(values.mean()),
        "ci_95_low": low,
        "ci_95_high": high,
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "iterations": iterations,
        "seed": seed,
    }


def build_tasks(
    cases: Mapping[str, Mapping[str, Any]],
    selected_case_ids: Sequence[str],
    ranking_rows: Mapping[tuple[str, str], Mapping[str, Any]],
    facts_by_case: Mapping[str, Sequence[str]],
    image_root: Path,
    *,
    policies: Sequence[str],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for case_id in selected_case_ids:
        source = cases[case_id]
        for question_type, question in QUESTIONS.items():
            ranking_row = ranking_rows[(case_id, question_type)]
            retrieved_ids = [str(value) for value in ranking_row["rankings"]["rrf_lambdamart"][:3]]
            top_cases = [cases[value] for value in retrieved_ids]
            plan = plan_question(question, str(source.get("indication", "")))
            query = "\n".join(part for part in (str(source.get("indication", "")), question) if part)
            case_to_fact = select_hierarchical_evidence(
                top_cases,
                query=query,
                facts_by_case=facts_by_case,
                plan=plan,
            )
            whole = whole_report_units(top_cases)
            confidence = compute_retrieval_confidence(
                [float(index) for index in range(3)],
                evidence_coverage=min(1.0, len(case_to_fact.units) / 6.0),
            )
            evidence_by_policy = {
                "whole_report": whole,
                "case_to_fact": case_to_fact.units,
            }
            reference, reference_proxy = reference_for(source, question_type)
            for policy in policies:
                evidence = evidence_by_policy[policy]
                answer_prompt = answer_only_generation_prompt(
                    indication=str(source.get("indication", "")),
                    question=question,
                    planner_instruction=render_planner_instruction(plan),
                    evidence=evidence,
                    abstain=not evidence,
                )
                tasks.append(
                    {
                        "case_id": case_id,
                        "question_type": question_type,
                        "question": question,
                        "reference_answer": reference,
                        "reference_is_proxy": reference_proxy,
                        "policy": policy,
                        "max_new_tokens": max_new_tokens,
                        "retrieved_case_ids": retrieved_ids,
                        "target_image_path": str(select_primary_image(source, image_root)),
                        "evidence": evidence,
                        "planner_intent": plan.intent,
                        "retrieval_confidence": confidence.confidence,
                        "prompt": answer_prompt,
                    }
                )
    return tasks


def summarize(rows: Sequence[Mapping[str, Any]], policies: Sequence[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for policy in policies:
        selected = [row for row in rows if row["policy"] == policy]
        summary[policy] = {
            "row_count": len(selected),
            "case_count": len({str(row["case_id"]) for row in selected}),
            "token_f1_all_rows": mean([float(row["token_f1"]) for row in selected]),
            "token_f1_non_proxy": mean([float(row["token_f1"]) for row in selected if not row["reference_is_proxy"]]),
            "answer_only_contract_valid_rate": mean([float(row["answer_only_contract_valid"]) for row in selected]),
            "serialization_leak_rate": mean([float(row["serialization_leak_detected"]) for row in selected]),
            "evidence_provenance_valid_rate": mean([float(row["evidence_provenance_valid"]) for row in selected]),
            "token_ceiling_rate": mean([float(row["hit_token_ceiling"]) for row in selected]),
            "mean_input_tokens": mean([float(row["input_tokens"]) for row in selected]),
            "mean_output_tokens": mean([float(row["output_tokens"]) for row in selected]),
            "mean_evidence_characters": mean([float(row["evidence_character_count"]) for row in selected]),
            "mean_latency_seconds": mean([float(row["latency_seconds"]) for row in selected]),
            "by_question_type": {
                question_type: {
                    "token_f1": mean([float(row["token_f1"]) for row in selected if row["question_type"] == question_type]),
                    "rows": sum(row["question_type"] == question_type for row in selected),
                }
                for question_type in QUESTIONS
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--facts", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--ranking-rows", type=Path, default=ROOT / "experiments/v12_optimization/retrieval/v12_validation_ranking_rows.jsonl")
    parser.add_argument("--selection-rows", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_selection_rows.jsonl")
    parser.add_argument("--image-root", type=Path, default=ROOT / "data/raw/openi_official_images")
    parser.add_argument("--rows-output", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_rows.jsonl")
    parser.add_argument("--summary-output", type=Path, default=ROOT / "experiments/v12_optimization/generation/v12_generation_summary.json")
    parser.add_argument("--max-new-tokens", type=int, choices=(96, 128), default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    args = parser.parse_args()

    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    facts_by_case = {
        str(row["case_id"]): tuple(row.get("facts", ()))
        for row in read_jsonl(args.facts)
        if row.get("status") == "ok"
    }
    selection_rows = read_jsonl(args.selection_rows)
    selected_case_ids = sorted({str(row["case_id"]) for row in selection_rows})
    if len(selected_case_ids) != 48:
        raise RuntimeError(f"expected the preselected 48-case V12 Validation manifest, found {len(selected_case_ids)}")
    ranking_rows = {
        (str(row["case_id"]), str(row["question_type"])): row
        for row in read_jsonl(args.ranking_rows)
        if str(row["case_id"]) in set(selected_case_ids)
    }
    expected_keys = {(case_id, question_type) for case_id in selected_case_ids for question_type in QUESTIONS}
    if set(ranking_rows) != expected_keys:
        raise RuntimeError("V12 ranking rows do not cover the fixed 48-case x 3-question matrix")
    tasks = build_tasks(
        cases,
        selected_case_ids,
        ranking_rows,
        facts_by_case,
        args.image_root,
        policies=tuple(args.policies),
        max_new_tokens=args.max_new_tokens,
    )
    expected = len(selected_case_ids) * len(QUESTIONS) * len(args.policies)
    if len(tasks) != expected:
        raise RuntimeError(f"incomplete V12 generation matrix: {len(tasks)} != {expected}")

    def key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
        return (str(row["case_id"]), str(row["question_type"]), str(row["policy"]), int(row["max_new_tokens"]))

    completed = {key(row) for row in read_jsonl(args.rows_output)} if args.rows_output.is_file() else set()
    pending = [task for task in tasks if key(task) not in completed]
    for task in pending:
        image_path = Path(task["target_image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"target image missing for {task['case_id']}: {image_path}")

    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generator = None
    if pending:
        generator = MedGemmaImageGenerator(
            revision=MEDGEMMA_REVISION,
            cache_dir=ROOT / ".hf_cache",
            local_files_only=True,
        )
        if torch.cuda.is_available():
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
            latency = (time.perf_counter() - batch_start) / max(len(batch), 1)
            peak_mib = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
            for task, output in zip(batch, outputs, strict=True):
                raw_answer = str(output["answer"])
                answer = clean_answer(raw_answer)
                diagnostics = output_contract_diagnostic(
                    raw_answer,
                    answer,
                    hit_ceiling=bool(output["hit_token_ceiling"]),
                )
                allowed = {unit.provenance_id for unit in task["evidence"]}
                provenance_valid = bool(task["evidence"]) and all(
                    unit.provenance_id in allowed for unit in task["evidence"]
                ) or not task["evidence"]
                row = {
                    "case_id": task["case_id"],
                    "question_type": task["question_type"],
                    "question": task["question"],
                    "reference_answer": task["reference_answer"],
                    "reference_is_proxy": task["reference_is_proxy"],
                    "policy": task["policy"],
                    "max_new_tokens": task["max_new_tokens"],
                    "retrieved_case_ids": task["retrieved_case_ids"],
                    "target_image_path": task["target_image_path"],
                    "planner_intent": task["planner_intent"],
                    "retrieval_confidence": task["retrieval_confidence"],
                    "evidence": [
                        {
                            "provenance_id": unit.provenance_id,
                            "case_id": unit.case_id,
                            "section": unit.section,
                            "text": unit.text,
                            "source_sha256": unit.source_sha256,
                        }
                        for unit in task["evidence"]
                    ],
                    "raw_output": raw_answer,
                    "answer": answer,
                    "token_f1": token_f1(answer, task["reference_answer"]),
                    "evidence_unit_count": len(task["evidence"]),
                    "evidence_character_count": sum(len(unit.text) for unit in task["evidence"]),
                    "evidence_provenance_valid": float(provenance_valid),
                    "input_tokens": int(output["input_tokens"]),
                    "output_tokens": int(output["output_tokens"]),
                    "latency_seconds": latency,
                    "peak_gpu_memory_mib": peak_mib,
                    **diagnostics,
                }
                handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            done = min(start + len(batch), len(pending))
            if done % 20 == 0 or done == len(pending):
                print(json.dumps({"generated_this_run": done, "pending_at_start": len(pending)}), flush=True)

    rows = read_jsonl(args.rows_output)
    if len(rows) != len(tasks):
        raise RuntimeError(f"generation output incomplete: {len(rows)} != {len(tasks)}")
    output = {
        "study": "V12 validation-only generation pilot",
        "status": "development_only_no_confirmation",
        "no_test_evaluation": True,
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "facts_sha256": file_sha256(args.facts),
            "ranking_rows_sha256": file_sha256(args.ranking_rows),
            "selection_rows_sha256": file_sha256(args.selection_rows),
            "selected_case_count": len(selected_case_ids),
            "selected_case_ids_sha256": sha256_ids(selected_case_ids),
            "selection_source": "predeclared V12 48-case Validation manifest selected by spectrum-stratified SHA-256 ordering; no outcome-based replacement",
            "model": {"name": "google/medgemma-1.5-4b-it", "revision": MEDGEMMA_REVISION, "local_files_only": True},
        },
        "retrieval_source": "saved V12 LambdaMART rrf_lambdamart Top-3",
        "max_new_tokens": args.max_new_tokens,
        "policies": list(args.policies),
        "counts": {"rows": len(rows), "cases": len(selected_case_ids), "questions_per_case": len(QUESTIONS)},
        "metrics": summarize(rows, tuple(args.policies)),
        "paired_case_bootstrap": {
            "case_to_fact_minus_whole_report": {
                "token_f1": paired_bootstrap(rows, "case_to_fact", "whole_report", "token_f1"),
                "answer_only_contract_valid": paired_bootstrap(rows, "case_to_fact", "whole_report", "answer_only_contract_valid"),
            }
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - started,
            "batch_size": args.batch_size,
            "max_peak_gpu_memory_mib": max((float(row["peak_gpu_memory_mib"]) for row in rows), default=0.0),
        },
        "generation_rows_sha256": file_sha256(args.rows_output),
        "claim_boundary": (
            "Token-F1 is automated overlap with a same-source report reference, not clinical correctness. "
            "This pilot is development-only, uses the preselected Validation sample, and does not establish "
            "clinical safety, external validation, or physician agreement."
        ),
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
