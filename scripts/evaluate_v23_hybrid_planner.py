from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from medical_rag.agentic.closed_loop_agent import (
    ClosedLoopEvidenceAgent,
    infer_report_intent,
)
from medical_rag.agentic.hybrid_planner import select_hybrid_plan
from medical_rag.evaluation.selective_prediction import apply_platt_scaler
from medical_rag.retrieval.scoped_chunk_retriever import ScopedBM25ChunkRetriever
from scripts.evaluate_case_scoped_hard_v21 import _system_metrics
from scripts.evaluate_v22_semantic_planner import (
    LABEL_TO_INTENT,
    expected_planner_label,
    parse_planner_label,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _git_introduction_commit(relative_path: str) -> str:
    commits = subprocess.check_output(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", relative_path],
        cwd=ROOT,
        text=True,
    ).splitlines()
    if not commits:
        raise RuntimeError(f"Cannot locate introduction commit for {relative_path}")
    return commits[-1].strip()


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run_systems(
    prompt_rows: list[dict[str, Any]],
    generations: dict[str, dict[str, Any]],
    questions: dict[str, dict[str, Any]],
    agent: ClosedLoopEvidenceAgent,
    set_name: str,
) -> dict[str, list[dict[str, Any]]]:
    outputs = {"lexical": [], "semantic": [], "hybrid": []}
    for prompt_row in prompt_rows:
        qid = str(prompt_row["qid"])
        source_qid = str(prompt_row["source_qid"])
        source = questions[source_qid]
        generation = generations[qid]
        planner_label = parse_planner_label(str(generation.get("answer", "")))
        semantic_intent = LABEL_TO_INTENT.get(planner_label, "unknown")
        hybrid_plan = select_hybrid_plan(str(prompt_row["question"]), semantic_intent)
        intents = {
            "lexical": None,
            "semantic": semantic_intent,
            "hybrid": hybrid_plan.selected_intent,
        }
        for system, planned_intent in intents.items():
            result = asdict(
                agent.run(
                    str(prompt_row["question"]),
                    str(source["scope_case_id"]),
                    planned_intent,
                )
            )
            outputs[system].append(
                {
                    **source,
                    **result,
                    "qid": qid,
                    "source_qid": source_qid,
                    "question": prompt_row["question"],
                    "evaluation_set": set_name,
                    "planner_label": planner_label,
                    "expected_planner_label": expected_planner_label(source_qid),
                    "lexical_intent": infer_report_intent(
                        str(prompt_row["question"])
                    ),
                    "semantic_intent": semantic_intent,
                    "hybrid_selected_intent": hybrid_plan.selected_intent,
                    "hybrid_planner_source": hybrid_plan.planner_source,
                    "system": f"v23_{system}",
                }
            )
    return outputs


def _calibrated_metrics(
    rows: list[dict[str, Any]], model: dict[str, float | int]
) -> dict[str, Any]:
    probabilities = apply_platt_scaler(
        [float(row["answer_probability"]) for row in rows], model
    )
    calibrated_rows = [
        {**row, "answer_probability": probability}
        for row, probability in zip(rows, probabilities, strict=True)
    ]
    return _system_metrics(calibrated_rows, 0.5)


def _summarize_set(
    systems: dict[str, list[dict[str, Any]]],
    threshold: float,
    platt: dict[str, float | int],
) -> dict[str, Any]:
    source_counts = Counter(
        row["hybrid_planner_source"] for row in systems["hybrid"]
    )
    planner_confusion: dict[str, dict[str, int]] = {}
    for row in systems["hybrid"]:
        expected = str(row["expected_planner_label"])
        predicted = str(row["planner_label"])
        planner_confusion.setdefault(expected, {}).setdefault(predicted, 0)
        planner_confusion[expected][predicted] += 1
    semantic_call_rate = source_counts["semantic_fallback"] / len(systems["hybrid"])
    system_diagnostics = {
        name: _failure_taxonomy(rows, threshold) for name, rows in systems.items()
    }
    return {
        "systems": {
            name: {
                "raw": _system_metrics(rows, threshold),
                "calibrated": _calibrated_metrics(rows, platt),
            }
            for name, rows in systems.items()
        },
        "hybrid_policy_usage": {
            "counts": dict(sorted(source_counts.items())),
            "semantic_planner_call_rate": semantic_call_rate,
            "avoided_semantic_planner_call_rate": 1.0 - semantic_call_rate,
        },
        "semantic_planner": {
            "label_accuracy": sum(
                row["planner_label"] == row["expected_planner_label"]
                for row in systems["hybrid"]
            )
            / len(systems["hybrid"]),
            "parse_failure_count": sum(
                row["planner_label"] == "PARSE_FAILURE"
                for row in systems["hybrid"]
            ),
            "label_confusion": planner_confusion,
        },
        "failure_taxonomy": system_diagnostics,
        "paired_case_bootstrap": _paired_case_bootstrap(
            systems["lexical"], systems["hybrid"], threshold
        ),
    }


def _failure_taxonomy(
    rows: list[dict[str, Any]], threshold: float
) -> dict[str, int]:
    outcomes: Counter[str] = Counter()
    for row in rows:
        predicts_answer = float(row["answer_probability"]) >= threshold
        hit = bool(set(row["relevant_chunk_ids"]) & set(row["retrieved_chunk_ids"]))
        if not row["is_answerable"]:
            category = (
                "false_answer_unanswerable" if predicts_answer else "correct_abstention"
            )
        elif not predicts_answer:
            category = "missed_answerable"
        elif not hit:
            category = "retrieval_miss"
        elif row["final_intent"] != row["expected_intent"]:
            category = "wrong_section_route_with_evidence_hit"
        else:
            category = "correct_answer_action"
        outcomes[category] += 1
    return dict(sorted(outcomes.items()))


def _binary_metrics(rows: list[dict[str, Any]], threshold: float) -> tuple[float, float]:
    truth = [bool(row["is_answerable"]) for row in rows]
    predicted = [float(row["answer_probability"]) >= threshold for row in rows]
    tp = sum(actual and guess for actual, guess in zip(truth, predicted))
    tn = sum(not actual and not guess for actual, guess in zip(truth, predicted))
    fp = sum(not actual and guess for actual, guess in zip(truth, predicted))
    fn = sum(actual and not guess for actual, guess in zip(truth, predicted))

    def f1(true_positive: int, false_positive: int, false_negative: int) -> float:
        denominator = 2 * true_positive + false_positive + false_negative
        return 2 * true_positive / denominator if denominator else 0.0

    macro_f1 = (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2
    false_answer_rate = fp / (fp + tn) if fp + tn else 0.0
    return macro_f1, false_answer_rate


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def _paired_case_bootstrap(
    lexical_rows: list[dict[str, Any]],
    hybrid_rows: list[dict[str, Any]],
    threshold: float,
    *,
    resamples: int = 5000,
    seed: int = 20260816,
) -> dict[str, Any]:
    lexical_by_case: dict[str, list[dict[str, Any]]] = {}
    hybrid_by_case: dict[str, list[dict[str, Any]]] = {}
    for row in lexical_rows:
        lexical_by_case.setdefault(str(row["scope_case_id"]), []).append(row)
    for row in hybrid_rows:
        hybrid_by_case.setdefault(str(row["scope_case_id"]), []).append(row)
    case_ids = sorted(lexical_by_case)
    if case_ids != sorted(hybrid_by_case):
        raise ValueError("Paired bootstrap requires identical case IDs.")
    rng = random.Random(seed)
    macro_deltas: list[float] = []
    false_answer_deltas: list[float] = []
    for _ in range(resamples):
        sampled = [rng.choice(case_ids) for _ in case_ids]
        lexical_sample = [row for case_id in sampled for row in lexical_by_case[case_id]]
        hybrid_sample = [row for case_id in sampled for row in hybrid_by_case[case_id]]
        lexical_macro, lexical_false = _binary_metrics(lexical_sample, threshold)
        hybrid_macro, hybrid_false = _binary_metrics(hybrid_sample, threshold)
        macro_deltas.append(hybrid_macro - lexical_macro)
        false_answer_deltas.append(hybrid_false - lexical_false)
    lexical_macro, lexical_false = _binary_metrics(lexical_rows, threshold)
    hybrid_macro, hybrid_false = _binary_metrics(hybrid_rows, threshold)
    return {
        "unit": "case_id",
        "case_count": len(case_ids),
        "resamples": resamples,
        "seed": seed,
        "macro_f1_delta_hybrid_minus_lexical": {
            "observed": hybrid_macro - lexical_macro,
            "ci95": [
                _percentile(macro_deltas, 0.025),
                _percentile(macro_deltas, 0.975),
            ],
            "bootstrap_probability_positive": sum(
                value > 0 for value in macro_deltas
            )
            / resamples,
        },
        "false_answer_rate_delta_hybrid_minus_lexical": {
            "observed": hybrid_false - lexical_false,
            "ci95": [
                _percentile(false_answer_deltas, 0.025),
                _percentile(false_answer_deltas, 0.975),
            ],
            "bootstrap_probability_positive": sum(
                value > 0 for value in false_answer_deltas
            )
            / resamples,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate preregistered V2.3 hybrid routing.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_scoped_hard_v21.json",
    )
    parser.add_argument(
        "--v21-summary",
        type=Path,
        default=ROOT / "experiments" / "post_submission_v21" / "summary.json",
    )
    parser.add_argument(
        "--v22-pack",
        type=Path,
        default=ROOT / "data" / "processed" / "prompt_packs" / "v22_semantic_planner.jsonl",
    )
    parser.add_argument(
        "--v22-generations",
        type=Path,
        default=ROOT / "experiments" / "post_submission_v22" / "planner_generations_qwen15.jsonl",
    )
    parser.add_argument(
        "--v23-pack",
        type=Path,
        default=ROOT / "data" / "processed" / "prompt_packs" / "v23_hybrid_transfer2_planner.jsonl",
    )
    parser.add_argument(
        "--v23-generations",
        type=Path,
        default=ROOT / "experiments" / "post_submission_v23" / "planner_generations_qwen15.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments" / "post_submission_v23" / "preregistration_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "post_submission_v23",
    )
    parser.add_argument(
        "--runtime-profile",
        type=Path,
        default=ROOT
        / "experiments"
        / "post_submission_v23"
        / "generation_runtime_profile.json",
    )
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    v21 = json.loads(args.v21_summary.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    questions = {str(row["qid"]): row for row in benchmark["questions"]}
    v22_pack = _read_jsonl(args.v22_pack)
    v23_pack = _read_jsonl(args.v23_pack)
    generations = {
        str(row["qid"]): row
        for row in _read_jsonl(args.v22_generations)
        + _read_jsonl(args.v23_generations)
    }
    required = {str(row["qid"]) for row in v22_pack + v23_pack}
    if set(generations) != required:
        raise ValueError("Planner generations must exactly cover both frozen packs.")

    retriever = ScopedBM25ChunkRetriever().fit(benchmark["chunks"])
    agent = ClosedLoopEvidenceAgent(retriever, first_pass_k=3, retry_k=3)
    sets = {
        "original_test": [
            row
            for row in v22_pack
            if not row["transfer"] and row["source_split"] == "test"
        ],
        "reserved_wording_set_1": [row for row in v22_pack if row["transfer"]],
        "reserved_wording_set_2": v23_pack,
    }
    threshold = float(
        v21["threshold_selection"]["closed_loop_agent_v2"]["threshold"]
    )
    platt = v21["posthoc_probability_calibration"]["closed_loop_agent_v2"][
        "model"
    ]
    all_rows: list[dict[str, Any]] = []
    summaries = {}
    for set_name, prompt_rows in sets.items():
        system_rows = _run_systems(
            prompt_rows, generations, questions, agent, set_name
        )
        for rows in system_rows.values():
            all_rows.extend(rows)
        summaries[set_name] = _summarize_set(system_rows, threshold, platt)

    original_hybrid = summaries["original_test"]["systems"]["hybrid"]["raw"]
    original_frozen = v21["systems"]["closed_loop_agent_v2"]["test"]
    if original_hybrid != original_frozen:
        raise AssertionError("V2.3 must reproduce the frozen V2.1 original-test metrics.")

    output = {
        "experiment": "v23_preregistered_hybrid_planner",
        "protocol": {
            **manifest,
            "preregistration_git_commit": _git_introduction_commit(
                "experiments/post_submission_v23/preregistration_manifest.json"
            ),
            "preregistration_manifest_canonical_sha256": _canonical_json_sha256(
                manifest
            ),
            "evaluation_after_preregistration_commit": True,
            "answerability_threshold": threshold,
            "probability_calibration_reused_without_refit": True,
        },
        "evaluation_sets": summaries,
    }
    if args.runtime_profile.exists():
        runtime_profile = json.loads(args.runtime_profile.read_text(encoding="utf-8"))
        if runtime_profile.get("records_generated") != len(v23_pack):
            raise ValueError("Runtime profile does not cover the complete V2.3 pack.")
        output["runtime_profile"] = runtime_profile
    output["cost_profile"] = {
        set_name: {
            "semantic_planner_call_rate": result["hybrid_policy_usage"][
                "semantic_planner_call_rate"
            ],
            "semantic_planner_calls_per_100_questions": 100
            * result["hybrid_policy_usage"]["semantic_planner_call_rate"],
            "mean_evidence_retrieval_calls": result["systems"]["hybrid"]["raw"][
                "mean_retrieval_calls"
            ],
            "mean_retrieved_chunks": result["systems"]["hybrid"]["raw"][
                "mean_retrieved_chunks"
            ],
        }
        for set_name, result in summaries.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
