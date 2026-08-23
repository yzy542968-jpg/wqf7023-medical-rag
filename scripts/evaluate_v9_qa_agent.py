from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_v6_development_confirmation_separation import file_sha256, read_json  # noqa: E402
from medical_rag.agentic.semantic_evidence_checker import MedicalNLIPredictor  # noqa: E402
from medical_rag.agentic.v9_historical_evidence_agent import (  # noqa: E402
    run_bounded_historical_evidence_agent,
)
from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402


DEFAULT_CONFIG = ROOT / "config" / "v9_qa_agent_confirmation.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_ROWS = ROOT / "experiments" / "post_submission_v9" / "v9_qa_raw_rows.jsonl"
DEFAULT_RAW_SUMMARY = ROOT / "experiments" / "post_submission_v9" / "v9_qa_raw_summary.json"
DEFAULT_RANKINGS = ROOT / "experiments" / "post_submission_v9" / "v9_qa_top3_rankings.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "post_submission_v9"
DEFAULT_PUBLIC_INDEX = ROOT / "data" / "splits" / "v9" / "v9_qualitative_case_index.csv"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bootstrap_difference(
    values: Mapping[str, float], *, iterations: int, seed: int
) -> dict[str, float]:
    observed = np.asarray(list(values.values()), dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        samples[index] = float(rng.choice(observed, len(observed), replace=True).mean())
    return {
        "difference": float(observed.mean()),
        "ci_95_low": float(np.quantile(samples, 0.025)),
        "ci_95_high": float(np.quantile(samples, 0.975)),
    }


def case_pack_selection(
    raw_rows: Sequence[Mapping[str, Any]], agent_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, str]]:
    by_case_system: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in raw_rows:
        by_case_system[str(row["case_id"])][str(row["system"])].append(float(row["token_f1"]))
    gains = {
        case_id: statistics.fmean(values["g3_learned_multimodal_rag"])
        - statistics.fmean(values["g0_no_retrieval"])
        for case_id, values in by_case_system.items()
    }
    agent_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in agent_rows:
        agent_by_case[str(row["case_id"])].append(row)
    categories = [
        (
            "largest_g3_minus_g0_gain",
            [case_id for case_id, _ in sorted(gains.items(), key=lambda item: (-item[1], item[0]))],
        ),
        (
            "largest_g3_minus_g0_loss",
            [case_id for case_id, _ in sorted(gains.items(), key=lambda item: (item[1], item[0]))],
        ),
        (
            "agent_retry_recovered",
            sorted(
                case_id
                for case_id, rows in agent_by_case.items()
                if any(row["retried"] and not row["historical_evidence_abstained"] for row in rows)
            ),
        ),
        (
            "agent_historical_evidence_abstention",
            sorted(
                case_id
                for case_id, rows in agent_by_case.items()
                if any(row["historical_evidence_abstained"] for row in rows)
            ),
        ),
    ]
    selected: list[dict[str, str]] = []
    used: set[str] = set()
    revised_pool = sorted(
        (case_id for case_id, rows in agent_by_case.items() if any(row["historical_support_revised"] for row in rows)),
        key=lambda case_id: (
            hashlib.sha256(f"v9-qualitative|7033|{case_id}".encode()).hexdigest(),
            case_id,
        ),
    )
    for category, candidates in categories:
        picked = [case_id for case_id in candidates if case_id not in used][:6]
        for case_id in picked:
            selected.append({"case_id": case_id, "selection_category": category, "selection_reason": category})
            used.add(case_id)
        while len(picked) < 6:
            fill = next((case_id for case_id in revised_pool if case_id not in used), None)
            if fill is None:
                fill = next(case_id for case_id in sorted(gains) if case_id not in used)
            selected.append(
                {
                    "case_id": fill,
                    "selection_category": category,
                    "selection_reason": "deterministic_revised_case_fill",
                }
            )
            used.add(fill)
            picked.append(fill)
    if len(selected) != 24 or len(used) != 24:
        raise RuntimeError("V9 qualitative selection did not produce 24 unique cases.")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen V9 bounded evidence-control agent.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--raw-summary", type=Path, default=DEFAULT_RAW_SUMMARY)
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--public-index", type=Path, default=DEFAULT_PUBLIC_INDEX)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    config = read_json(args.config)
    raw_summary = read_json(args.raw_summary)
    if raw_summary["status"] != "formal_test_qa_outcomes_frozen_no_retuning":
        raise RuntimeError("V9 raw QA outcomes are not frozen.")
    if file_sha256(args.rows) != raw_summary["outputs"]["rows_sha256"]:
        raise RuntimeError("V9 raw QA rows changed after freeze.")
    raw_rows = read_jsonl(args.rows)
    g3_rows = [row for row in raw_rows if row["system"] == "g3_learned_multimodal_rag"]
    if len(g3_rows) != int(config["qa_frame"]["question_count"]):
        raise RuntimeError("The G3 row frame is incomplete.")
    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    ranking_rows = read_jsonl(args.rankings)
    rankings = {(str(row["qid"]), str(row["system"])): list(row["top_case_ids"]) for row in ranking_rows}
    agent_config = config["agent"]
    predictor = MedicalNLIPredictor(
        str(agent_config["historical_support_checker"]),
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=True,
    )

    output_path = args.output_dir / "v9_agent_rows.jsonl"
    summary_path = args.output_dir / "v9_agent_summary.json"
    local_pack_path = args.output_dir / "v9_qualitative_review_pack.jsonl"
    if summary_path.exists():
        raise RuntimeError("V9 agent summary already exists; refusing rerun.")
    started = time.perf_counter()
    agent_rows = []
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(g3_rows, start=1):
            qid = str(row["qid"])
            result = run_bounded_historical_evidence_agent(
                answer_row=row,
                primary_case_ids=rankings[(qid, "r4_learned_mlp")],
                retry_case_ids=rankings[(qid, "r1_image_image")],
                cases=cases,
                predictor=predictor,
                minimum_support_rate=float(agent_config["minimum_support_rate"]),
                minimum_combined_support=float(agent_config["minimum_combined_support"]),
                entailment_threshold=float(agent_config["entailment_threshold"]),
                contradiction_threshold=float(agent_config["contradiction_threshold"]),
            )
            result.update(
                {
                    "system": "g4_bounded_agentic_multimodal_rag",
                    "case_id": row["case_id"],
                    "qid": qid,
                    "question_type": row["question_type"],
                    "reference_answer": row["reference_answer"],
                    "token_f1": token_f1(result["agent_answer"], str(row["reference_answer"])),
                }
            )
            agent_rows.append(result)
            handle.write(json.dumps(result, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()
            if index % 100 == 0 or index == len(g3_rows):
                print(f"agent_rows={index}/{len(g3_rows)}", flush=True)

    by_case_initial: dict[str, list[float]] = defaultdict(list)
    by_case_final: dict[str, list[float]] = defaultdict(list)
    by_case_qa: dict[str, list[float]] = defaultdict(list)
    for row in agent_rows:
        by_case_initial[str(row["case_id"])].append(float(row["initial_unsupported"]))
        by_case_final[str(row["case_id"])].append(float(row["final_unsupported"]))
        by_case_qa[str(row["case_id"])].append(float(row["token_f1"]))
    unsupported_difference = {
        case_id: statistics.fmean(by_case_final[case_id]) - statistics.fmean(by_case_initial[case_id])
        for case_id in by_case_initial
    }
    g3_by_case: dict[str, list[float]] = defaultdict(list)
    for row in g3_rows:
        g3_by_case[str(row["case_id"])].append(float(row["token_f1"]))
    qa_difference = {
        case_id: statistics.fmean(by_case_qa[case_id]) - statistics.fmean(g3_by_case[case_id])
        for case_id in by_case_qa
    }

    selection = case_pack_selection(raw_rows, agent_rows)
    raw_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        raw_by_case[str(row["case_id"])].append(row)
    agent_by_case: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in agent_rows:
        agent_by_case[str(row["case_id"])].append(row)
    with local_pack_path.open("w", encoding="utf-8", newline="\n") as handle:
        for selected in selection:
            case_id = selected["case_id"]
            handle.write(
                json.dumps(
                    {
                        **selected,
                        "source_case": cases[case_id],
                        "generation_rows": raw_by_case[case_id],
                        "agent_rows": agent_by_case[case_id],
                        "assistant_proposed_labels_v1_0": [],
                        "researcher_reviewed_labels_v1_0": [],
                        "review_status": "pending_researcher_review",
                        "review_note": "",
                        "reviewer_initials": "",
                        "review_date": "",
                    },
                    ensure_ascii=True,
                    sort_keys=True,
                )
                + "\n"
            )
    args.public_index.parent.mkdir(parents=True, exist_ok=True)
    with args.public_index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "selection_category",
                "selection_reason",
                "assistant_proposed_labels_v1_0",
                "researcher_reviewed_labels_v1_0",
                "review_status",
                "review_note",
                "reviewer_initials",
                "review_date",
            ],
        )
        writer.writeheader()
        for row in selection:
            writer.writerow(
                {
                    **row,
                    "assistant_proposed_labels_v1_0": "",
                    "researcher_reviewed_labels_v1_0": "",
                    "review_status": "pending_researcher_review",
                    "review_note": "",
                    "reviewer_initials": "",
                    "review_date": "",
                }
            )

    iterations = int(config["bootstrap"]["iterations"])
    seed = int(config["bootstrap"]["seed"])
    unsupported_stats = bootstrap_difference(unsupported_difference, iterations=iterations, seed=seed)
    qa_stats = bootstrap_difference(qa_difference, iterations=iterations, seed=seed + 1)
    summary = {
        "study": "V9 bounded historical-evidence agent",
        "status": "agent_evaluation_complete_no_retuning",
        "config_sha256": file_sha256(args.config),
        "raw_qa_summary_sha256": file_sha256(args.raw_summary),
        "raw_qa_rows_sha256": file_sha256(args.rows),
        "ranking_rows_sha256": file_sha256(args.rankings),
        "implementation_sha256": file_sha256(Path(__file__)),
        "row_count": len(agent_rows),
        "case_count": len(by_case_initial),
        "metrics": {
            "g3_initial_unsupported_historical_rate": statistics.fmean(float(row["initial_unsupported"]) for row in agent_rows),
            "g4_final_unsupported_historical_rate": statistics.fmean(float(row["final_unsupported"]) for row in agent_rows),
            "historical_claim_presence_rate": statistics.fmean(float(row["historical_claim_present"]) for row in agent_rows),
            "retry_rate": statistics.fmean(float(row["retried"]) for row in agent_rows),
            "historical_evidence_abstention_rate": statistics.fmean(float(row["historical_evidence_abstained"]) for row in agent_rows),
            "historical_support_revision_rate": statistics.fmean(float(row["historical_support_revised"]) for row in agent_rows),
            "mean_retrieval_calls": statistics.fmean(int(row["retrieval_calls"]) for row in agent_rows),
            "g4_token_f1": statistics.fmean(float(row["token_f1"]) for row in agent_rows),
        },
        "paired_case_bootstrap": {
            "g4_minus_g3_unsupported_historical_rate": unsupported_stats,
            "g4_minus_g3_token_f1": qa_stats,
        },
        "success": {
            "unsupported_historical_rate_reduced": unsupported_stats["ci_95_high"] < 0,
            "qa_noninferiority_margin": config["agent_success"]["qa_noninferiority_margin"],
            "qa_noninferiority_met": qa_stats["ci_95_low"] >= float(config["agent_success"]["qa_noninferiority_margin"]),
        },
        "runtime_seconds": time.perf_counter() - started,
        "outputs": {
            "agent_rows_path": str(output_path.relative_to(ROOT)).replace("\\", "/"),
            "agent_rows_sha256": file_sha256(output_path),
            "local_review_pack_path": str(local_pack_path.relative_to(ROOT)).replace("\\", "/"),
            "local_review_pack_sha256": file_sha256(local_pack_path),
            "public_case_index_path": str(args.public_index.relative_to(ROOT)).replace("\\", "/"),
            "public_case_index_sha256": file_sha256(args.public_index),
            "large_outputs_committed": False,
        },
        "target_image_answer_verified_by_agent": False,
        "human_clinical_adjudication": False,
        "retuning_after_agent_outcomes": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
