from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from medical_rag.agentic.closed_loop_agent import ClosedLoopEvidenceAgent
from medical_rag.evaluation.radqa_agent import select_answerability_threshold
from medical_rag.evaluation.selective_prediction import (
    apply_platt_scaler,
    fit_platt_scaler,
)
from medical_rag.retrieval.scoped_chunk_retriever import ScopedBM25ChunkRetriever
from scripts.evaluate_case_scoped_hard_v21 import _system_metrics, _threshold_rows


LABEL_TO_INTENT = {
    "FINDINGS": "findings",
    "IMPRESSION": "impression",
    "REPORT_FACT": "unknown",
    "OUTSIDE_REPORT": "unavailable",
}


def parse_planner_label(answer: str) -> str:
    first_line = answer.strip().upper().splitlines()[0] if answer.strip() else ""
    normalized = re.sub(r"[^A-Z_]", " ", first_line)
    tokens = normalized.split()
    for label in ("OUTSIDE_REPORT", "REPORT_FACT", "IMPRESSION", "FINDINGS"):
        if label in tokens or first_line.startswith(label):
            return label
    return "PARSE_FAILURE"


def expected_planner_label(qid: str) -> str:
    if "_v21_observation" in qid:
        return "FINDINGS"
    if "_v21_conclusion" in qid:
        return "IMPRESSION"
    if "_v21_fact_probe" in qid or "_v21_near_domain_negative" in qid:
        return "REPORT_FACT"
    if "_v21_unanswerable_" in qid:
        return "OUTSIDE_REPORT"
    raise ValueError(f"Unknown planner QID: {qid}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _calibrated_rows(
    rows: list[dict[str, Any]], model: dict[str, float | int]
) -> list[dict[str, Any]]:
    probabilities = apply_platt_scaler(
        [float(row["answer_probability"]) for row in rows], model
    )
    return [
        {**row, "answer_probability": probability}
        for row, probability in zip(rows, probabilities, strict=True)
    ]


def _diagnostics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    confusion: dict[str, dict[str, int]] = {}
    outcomes: Counter[str] = Counter()
    for row in rows:
        expected_label = str(row["expected_planner_label"])
        planner_label = str(row["planner_label"])
        confusion.setdefault(expected_label, {}).setdefault(planner_label, 0)
        confusion[expected_label][planner_label] += 1
        predicts_answer = float(row["answer_probability"]) >= threshold
        hit = bool(set(row["relevant_chunk_ids"]) & set(row["retrieved_chunk_ids"]))
        if not row["is_answerable"]:
            outcome = "false_answer_unanswerable" if predicts_answer else "correct_abstention"
        elif not predicts_answer:
            outcome = "missed_answerable"
        elif not hit:
            outcome = "retrieval_miss"
        else:
            outcome = "correct_answer_action"
        outcomes[outcome] += 1
    return {
        "planner_label_confusion": confusion,
        "action_outcome_counts": dict(sorted(outcomes.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen Qwen semantic planner with the bounded evidence loop."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_scoped_hard_v21.json",
    )
    parser.add_argument(
        "--planner-pack",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "prompt_packs"
        / "v22_semantic_planner.jsonl",
    )
    parser.add_argument(
        "--planner-generations",
        type=Path,
        default=ROOT
        / "experiments"
        / "post_submission_v22"
        / "planner_generations_qwen15.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "experiments"
        / "post_submission_v22"
        / "planner_pack_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "post_submission_v22",
    )
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pack = {str(row["qid"]): row for row in _read_jsonl(args.planner_pack)}
    generations = _read_jsonl(args.planner_generations)
    if len(generations) != len(pack) or len({row["qid"] for row in generations}) != len(pack):
        raise ValueError("Planner generations must exactly cover the frozen prompt pack once.")

    question_by_qid = {str(row["qid"]): row for row in benchmark["questions"]}
    retriever = ScopedBM25ChunkRetriever().fit(benchmark["chunks"])
    agent = ClosedLoopEvidenceAgent(retriever, first_pass_k=3, retry_k=3)
    rows: list[dict[str, Any]] = []
    planner_correct: list[bool] = []
    parse_failures = 0
    for generation in generations:
        qid = str(generation["qid"])
        pack_row = pack[qid]
        source_qid = str(pack_row["source_qid"])
        source = question_by_qid[source_qid]
        label = parse_planner_label(str(generation.get("answer", "")))
        parse_failures += int(label == "PARSE_FAILURE")
        planner_correct.append(label == expected_planner_label(source_qid))
        intent = LABEL_TO_INTENT.get(label, "unknown")
        result = asdict(
            agent.run(str(pack_row["question"]), str(source["scope_case_id"]), intent)
        )
        rows.append(
            {
                **source,
                **result,
                "qid": qid,
                "source_qid": source_qid,
                "question": pack_row["question"],
                "transfer": bool(pack_row["transfer"]),
                "planner_label": label,
                "expected_planner_label": expected_planner_label(source_qid),
                "planner_correct": planner_correct[-1],
                "system": "semantic_planner_agent_v22",
            }
        )

    original = [row for row in rows if not row["transfer"]]
    development = [row for row in original if row["split"] == "development"]
    calibration = [row for row in original if row["split"] == "calibration"]
    original_test = [row for row in original if row["split"] == "test"]
    transfer_test = [row for row in rows if row["transfer"]]
    selection = select_answerability_threshold(_threshold_rows(development))
    threshold = float(selection["selected"]["threshold"])
    platt = fit_platt_scaler(
        [float(row["answer_probability"]) for row in calibration],
        [bool(row["is_answerable"]) for row in calibration],
    )

    output = {
        "experiment": "v22_semantic_planner_agent",
        "protocol": {
            **manifest,
            "planner_model": generations[0]["model"],
            "planner_temperature": 0.0,
            "planner_max_new_tokens": 8,
            "answerability_threshold_fit_split": "development",
            "probability_calibration_fit_split": "calibration",
            "test_or_transfer_tuning": False,
        },
        "planner": {
            "n": len(rows),
            "parse_failure_count": parse_failures,
            "overall_label_accuracy": sum(planner_correct) / len(planner_correct),
            "original_test_label_accuracy": sum(
                row["planner_correct"] for row in original_test
            )
            / len(original_test),
            "transfer_test_label_accuracy": sum(
                row["planner_correct"] for row in transfer_test
            )
            / len(transfer_test),
        },
        "answerability_threshold_selection": selection["selected"],
        "platt_model": platt,
        "original_test": {
            "raw": _system_metrics(original_test, threshold),
            "calibrated": _system_metrics(
                _calibrated_rows(original_test, platt), 0.5
            ),
            "diagnostics": _diagnostics(original_test, threshold),
        },
        "transfer_test": {
            "raw": _system_metrics(transfer_test, threshold),
            "calibrated": _system_metrics(
                _calibrated_rows(transfer_test, platt), 0.5
            ),
            "diagnostics": _diagnostics(transfer_test, threshold),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "semantic_planner_agent_rows.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
