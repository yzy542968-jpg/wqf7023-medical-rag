from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.closed_loop_agent import ClosedLoopEvidenceAgent
from medical_rag.evaluation.answer_metrics import token_f1
from medical_rag.evaluation.case_scoped_hard_benchmark import (
    build_case_scoped_hard_benchmark,
    dumps_json,
)
from medical_rag.evaluation.radqa_agent import (
    answerability_metrics,
    select_answerability_threshold,
)
from medical_rag.evaluation.selective_prediction import (
    apply_platt_scaler,
    calibration_metrics,
    fit_platt_scaler,
    risk_coverage_curve,
)
from medical_rag.retrieval.scoped_chunk_retriever import ScopedBM25ChunkRetriever
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = _read_json(path)
    ids = {
        str(row["case_id"])
        for row in payload.get("questions", [])
        if row.get("case_id") is not None
    }
    for part in payload.get("split", {}).values():
        ids.update(str(value) for value in part.get("case_ids", []))
    return ids


def _bounded_score(score: float) -> float:
    return 1.0 - math.exp(-max(0.0, score))


def _baseline_row(
    question: dict[str, Any], retriever: ScopedBM25ChunkRetriever, max_chunks: int
) -> dict[str, Any]:
    rows = retriever.search(
        question["question"], top_k=max_chunks, case_id=question["scope_case_id"]
    )
    top_score = float(rows[0]["score"]) if rows else 0.0
    return {
        **question,
        "system": "fixed_report_bm25",
        "answer_probability": _bounded_score(top_score),
        "planned_intent": "none",
        "final_intent": "none",
        "retrieved_chunk_ids": [str(row["chunk_id"]) for row in rows],
        "retrieved_sections": [str(row["section"]) for row in rows],
        "retrieved_texts": [str(row["text"]) for row in rows],
        "retrieval_calls": 1,
        "retrieved_chunk_count": len(rows),
        "retried": False,
        "trace": [],
    }


def _agent_row(
    question: dict[str, Any], agent: ClosedLoopEvidenceAgent, system: str
) -> dict[str, Any]:
    result = asdict(agent.run(question["question"], question["scope_case_id"]))
    return {**question, **result, "system": system}


def _threshold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**row, "top1_score": float(row["answer_probability"])} for row in rows
    ]


def _system_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    binary = answerability_metrics(_threshold_rows(rows), threshold)
    correctness: list[bool] = []
    prediction_confidences: list[float] = []
    retrieval_hits: list[bool] = []
    answer_f1s: list[float] = []
    route_hits: list[bool] = []

    for row in rows:
        probability = float(row["answer_probability"])
        predicts_answer = probability >= threshold
        relevant = set(row["relevant_chunk_ids"])
        retrieved = set(row["retrieved_chunk_ids"])
        retrieval_hit = bool(relevant & retrieved) if row["is_answerable"] else False
        if row["is_answerable"]:
            retrieval_hits.append(retrieval_hit)
            correct = predicts_answer and retrieval_hit
            answer = (
                " ".join(row["retrieved_texts"]) if predicts_answer else "NOT ANSWERABLE"
            )
            answer_f1s.append(token_f1(answer, row["reference_answer"]))
            if row["system"] in {
                "route_only_agent",
                "closed_loop_agent_v2",
                "semantic_planner_agent_v22",
            }:
                route_hits.append(row["final_intent"] == row["expected_intent"])
        else:
            correct = not predicts_answer
        correctness.append(correct)
        prediction_confidences.append(probability if predicts_answer else 1.0 - probability)

    curve = risk_coverage_curve(prediction_confidences, correctness)
    curve_points = curve.pop("points")
    curve["deciles"] = [
        curve_points[min(len(curve_points) - 1, math.ceil(len(curve_points) * fraction) - 1)]
        for fraction in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    ]
    binary.update(
        {
            "retrieval_hit_rate_answerable": mean(retrieval_hits) if retrieval_hits else None,
            "end_to_end_action_accuracy": mean(correctness),
            "answerable_token_f1": mean(answer_f1s) if answer_f1s else None,
            "route_accuracy_answerable": mean(route_hits) if route_hits else None,
            "mean_retrieval_calls": mean(float(row["retrieval_calls"]) for row in rows),
            "mean_retrieved_chunks": mean(
                float(row["retrieved_chunk_count"]) for row in rows
            ),
            "retry_rate": mean(bool(row["retried"]) for row in rows),
            "answerability_calibration": calibration_metrics(
                [float(row["answer_probability"]) for row in rows],
                [bool(row["is_answerable"]) for row in rows],
            ),
            "risk_coverage": curve,
        }
    )
    return binary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and evaluate the leakage-resistant v2.1 hard benchmark."
    )
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "data" / "processed" / "openi_cases.jsonl"
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_scoped_hard_v21.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "post_submission_v21",
    )
    parser.add_argument("--max-cases", type=int, default=240)
    parser.add_argument("--seed", type=int, default=27023)
    parser.add_argument("--max-chunks", type=int, default=6)
    args = parser.parse_args()

    prior_paths = [
        ROOT / "data" / "processed" / "openi_case_qa_seed_clean.json",
        ROOT / "data" / "processed" / "openi_case_scoped_benchmark_v2.json",
        ROOT / "data" / "processed" / "openi_case_scoped_confirmation_v2.json",
    ]
    excluded: set[str] = set()
    for path in prior_paths:
        excluded.update(_case_ids(path))

    benchmark = build_case_scoped_hard_benchmark(
        load_cases_jsonl(args.cases),
        excluded_case_ids=excluded,
        max_cases=args.max_cases,
        seed=args.seed,
    )
    args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    args.benchmark_output.write_text(dumps_json(benchmark), encoding="utf-8")

    retriever = ScopedBM25ChunkRetriever().fit(benchmark["chunks"])
    agent = ClosedLoopEvidenceAgent(
        retriever,
        first_pass_k=args.max_chunks // 2,
        retry_k=args.max_chunks - args.max_chunks // 2,
    )
    route_only_agent = ClosedLoopEvidenceAgent(
        retriever,
        first_pass_k=args.max_chunks,
        retry_k=0,
        retry_threshold=0.0,
    )
    baseline_rows = [
        _baseline_row(question, retriever, args.max_chunks)
        for question in benchmark["questions"]
    ]
    route_only_rows = [
        _agent_row(question, route_only_agent, "route_only_agent")
        for question in benchmark["questions"]
    ]
    agent_rows = [
        _agent_row(question, agent, "closed_loop_agent_v2")
        for question in benchmark["questions"]
    ]

    systems = {
        "fixed_report_bm25": baseline_rows,
        "route_only_agent": route_only_rows,
        "closed_loop_agent_v2": agent_rows,
    }
    selections = {
        system: select_answerability_threshold(
            _threshold_rows([row for row in rows if row["split"] == "development"])
        )
        for system, rows in systems.items()
    }
    thresholds = {
        system: selection["selected"]["threshold"]
        for system, selection in selections.items()
    }
    summaries: dict[str, Any] = {}
    for system, rows in systems.items():
        threshold = thresholds[system]
        summaries[system] = {
            split: _system_metrics(
                [row for row in rows if row["split"] == split], threshold
            )
            for split in ("development", "calibration", "test")
        }

    posthoc_calibration: dict[str, Any] = {}
    for system, rows in systems.items():
        calibration_rows = [row for row in rows if row["split"] == "calibration"]
        model = fit_platt_scaler(
            [float(row["answer_probability"]) for row in calibration_rows],
            [bool(row["is_answerable"]) for row in calibration_rows],
        )
        split_results: dict[str, Any] = {}
        for split in ("calibration", "test"):
            split_rows = [row for row in rows if row["split"] == split]
            calibrated = apply_platt_scaler(
                [float(row["answer_probability"]) for row in split_rows], model
            )
            calibrated_rows = [
                {**row, "answer_probability": probability}
                for row, probability in zip(split_rows, calibrated, strict=True)
            ]
            split_results[split] = _system_metrics(calibrated_rows, threshold=0.5)
        posthoc_calibration[system] = {
            "fit_split": "calibration",
            "test_was_not_used_for_fitting": True,
            "model": model,
            "fixed_decision_threshold": 0.5,
            "splits": split_results,
        }

    summary = {
        "experiment": "case_scoped_hard_v21",
        "benchmark_version": benchmark["version"],
        "benchmark_content_fingerprint_sha256": benchmark[
            "content_fingerprint_sha256"
        ],
        "case_count": benchmark["case_count"],
        "question_count": benchmark["question_count"],
        "excluded_prior_case_count": len(excluded),
        "fair_compute_budget": {
            "maximum_retrieved_chunks_per_question": args.max_chunks,
            "baseline_maximum_calls": 1,
            "agent_maximum_calls": 2,
        },
        "threshold_selection": {
            "data": "development only",
            **{
                system: selection["selected"]
                for system, selection in selections.items()
            },
        },
        "systems": summaries,
        "posthoc_probability_calibration": posthoc_calibration,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for system, rows in systems.items():
        with (args.output_dir / f"{system}_rows.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
