from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
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


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


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
    semantic_call_rate = source_counts["semantic_fallback"] / len(systems["hybrid"])
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
            "preregistration_git_commit": _git_commit(),
            "evaluation_after_preregistration_commit": True,
            "answerability_threshold": threshold,
            "probability_calibration_reused_without_refit": True,
        },
        "evaluation_sets": summaries,
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
