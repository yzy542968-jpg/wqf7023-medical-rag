from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.semantic_evidence_checker import (  # noqa: E402
    MedicalNLIPredictor,
    check_semantic_evidence_support,
)
from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1  # noqa: E402
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl  # noqa: E402


PROTOCOL_COMMIT = "eee7405"
COHORT_COMMIT = "43fe1a0"
RAW_QA_OUTCOME_COMMIT = "5aa8a6b"

DEFAULT_GENERATIONS = (
    ROOT / "experiments" / "post_submission_v6" / "confirmation_qa_factorial_rows.jsonl"
)
DEFAULT_GENERATION_SUMMARY = (
    ROOT / "experiments" / "post_submission_v6" / "confirmation_qa_factorial_summary.json"
)
DEFAULT_CONFIG = ROOT / "config" / "v6_confirmation.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_SEMANTIC_CONFIG = (
    ROOT
    / "experiments"
    / "final_optimized"
    / "semantic_agent"
    / "semantic_agent_selection.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "post_submission_v6"
EXPECTED_SYSTEMS = {
    "bm25_qwen2_5",
    "medsiglip_qwen2_5",
    "bm25_medgemma_1_5",
    "medsiglip_medgemma_1_5",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commit_exists(commit: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def evidence_text(case: dict[str, Any] | None) -> str:
    if not case:
        return ""
    return "\n".join(
        [
            f"Case ID: {case['case_id']}",
            f"Findings: {case.get('findings', '')}",
            f"Impression: {case.get('impression', '')}",
        ]
    )


def validate_factorial_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [(str(row["system"]), str(row["qid"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate (system, qid) rows found in the V6 factorial.")
    by_system: dict[str, set[str]] = defaultdict(set)
    for system, qid in keys:
        by_system[system].add(qid)
    if set(by_system) != EXPECTED_SYSTEMS:
        raise ValueError(f"Unexpected V6 factorial systems: {sorted(by_system)}")
    qid_sets = list(by_system.values())
    if any(len(qids) != 360 for qids in qid_sets):
        raise ValueError("Each V6 factorial system must contain exactly 360 qids.")
    if any(qids != qid_sets[0] for qids in qid_sets[1:]):
        raise ValueError("V6 factorial systems do not contain identical qid sets.")
    return {
        "row_count": len(rows),
        "system_count": len(by_system),
        "qids_per_system": 360,
        "unique_case_count": len({str(row["case_id"]) for row in rows}),
        "duplicate_system_qid_count": 0,
        "identical_qid_sets": True,
    }


def validate_frozen_inputs(
    *,
    generation_rows_path: Path,
    generation_summary_path: Path,
    config_path: Path,
    semantic_config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for commit in (PROTOCOL_COMMIT, COHORT_COMMIT, RAW_QA_OUTCOME_COMMIT):
        if not commit_exists(commit):
            raise RuntimeError(f"Required frozen commit is unavailable: {commit}")
    generation_summary = read_json(generation_summary_path)
    config = read_json(config_path)
    semantic_selection = read_json(semantic_config_path)

    if generation_summary["status"] != "formal_confirmation_raw_qa_outcomes_frozen":
        raise RuntimeError("The formal raw QA outcome is not frozen.")
    if generation_summary["protocol_commit"] != PROTOCOL_COMMIT:
        raise RuntimeError("The formal raw QA protocol commit changed.")
    if generation_summary["cohort_commit"] != COHORT_COMMIT:
        raise RuntimeError("The formal raw QA cohort commit changed.")
    if generation_summary["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("The formal raw QA config hash changed.")
    if generation_summary["outputs"]["rows_sha256"] != file_sha256(
        generation_rows_path
    ):
        raise RuntimeError("The local raw QA rows do not match their frozen summary.")
    if generation_summary["outputs"]["row_count"] != 1440:
        raise RuntimeError("The formal raw QA matrix is incomplete.")

    verification = config["verification"]
    selected = semantic_selection["selected_config"]
    expected_values = {
        "lexical_weight": selected["lexical_weight"],
        "support_threshold": selected["support_threshold"],
        "entailment_threshold": selected["entailment_threshold"],
        "contradiction_threshold": selected["contradiction_threshold"],
    }
    if semantic_selection["nli_model"] != verification["model"]:
        raise RuntimeError("The frozen verifier model changed.")
    if file_sha256(semantic_config_path) != verification["config_sha256"]:
        raise RuntimeError("The frozen verifier config hash changed.")
    if any(float(verification[key]) != float(value) for key, value in expected_values.items()):
        raise RuntimeError("The frozen verifier thresholds changed.")
    if verification["changed_from_v5"] is not False:
        raise RuntimeError("The confirmation verifier is no longer the frozen V5 verifier.")
    return generation_summary, config, semantic_selection


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["system"])].append(row)
    result = {}
    for system, system_rows in sorted(grouped.items()):
        result[system] = {
            "row_count": len(system_rows),
            "case_count": len({str(row["case_id"]) for row in system_rows}),
            "draft_token_f1": mean(float(row["draft_token_f1"]) for row in system_rows),
            "verified_token_f1": mean(float(row["final_token_f1"]) for row in system_rows),
            "evidence_support_rate": mean(float(row["support_rate"]) for row in system_rows),
            "final_abstention_rate": mean(float(row["agent_abstained"]) for row in system_rows),
            "revision_rate": mean(float(row["revised"]) for row in system_rows),
            "retrieval_top1_accuracy": mean(
                float(row["retrieval_rank_correct"]) for row in system_rows
            ),
            "nli_contradiction_count": sum(
                int(row["nli_contradiction_count"]) for row in system_rows
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the frozen V5 semantic verifier to V6 confirmation QA."
    )
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument(
        "--generation-summary", type=Path, default=DEFAULT_GENERATION_SUMMARY
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--semantic-config", type=Path, default=DEFAULT_SEMANTIC_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "confirmation_qa_factorial_verified_rows.jsonl"
    summary_path = args.output_dir / "confirmation_qa_factorial_verified_summary.json"
    if summary_path.exists():
        raise RuntimeError("Formal confirmation verifier summary exists; refusing rerun.")

    generation_summary, config, semantic_selection = validate_frozen_inputs(
        generation_rows_path=args.generations,
        generation_summary_path=args.generation_summary,
        config_path=args.config,
        semantic_config_path=args.semantic_config,
    )
    generation_rows = read_jsonl(args.generations)
    integrity = validate_factorial_rows(generation_rows)
    cases = {str(case["case_id"]): case for case in load_cases_jsonl(args.cases)}
    verifier_config = semantic_selection["selected_config"]
    predictor = MedicalNLIPredictor(
        semantic_selection["nli_model"],
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=True,
    )

    existing_rows = read_jsonl(output_path) if output_path.is_file() else []
    completed = {(str(row["system"]), str(row["qid"])) for row in existing_rows}
    if len(completed) != len(existing_rows):
        raise RuntimeError("The partial verified output contains duplicate keys.")
    pending = [
        row
        for row in generation_rows
        if (str(row["system"]), str(row["qid"])) not in completed
    ]

    started = time.perf_counter()
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(pending, start=1):
            selected_case = cases.get(str(row["selected_case_id"]))
            draft = extract_final_answer(str(row.get("answer", "")))
            check = check_semantic_evidence_support(
                draft,
                evidence_text(selected_case),
                predictor,
                min_combined_support=float(verifier_config["support_threshold"]),
                entailment_threshold=float(verifier_config["entailment_threshold"]),
                contradiction_threshold=float(verifier_config["contradiction_threshold"]),
                lexical_weight=float(verifier_config["lexical_weight"]),
            )
            reference = str(row["reference_answer"])
            contradiction_count = sum(
                value.contradiction_probability
                >= float(verifier_config["contradiction_threshold"])
                for value in check.sentence_checks
            )
            output = {
                **row,
                "draft_answer": draft,
                "final_answer": check.revised_answer,
                "draft_token_f1": token_f1(draft, reference),
                "final_token_f1": token_f1(check.revised_answer, reference),
                "support_rate": check.support_rate,
                "agent_abstained": check.abstained,
                "revised": check.revised_answer.strip() != draft.strip(),
                "nli_contradiction_count": contradiction_count,
                "sentence_checks": [asdict(value) for value in check.sentence_checks],
            }
            handle.write(json.dumps(output, ensure_ascii=True) + "\n")
            handle.flush()
            if index % 100 == 0 or index == len(pending):
                print(
                    json.dumps(
                        {
                            "verified_this_run": index,
                            "pending_at_start": len(pending),
                        }
                    ),
                    flush=True,
                )
    elapsed = time.perf_counter() - started

    final_rows = read_jsonl(output_path)
    verified_integrity = validate_factorial_rows(final_rows)
    if len(final_rows) != 1440:
        raise RuntimeError("The formal V6 confirmation verified matrix is incomplete.")
    summary = {
        "experiment": "V6 confirmation QA with frozen V5 semantic verifier",
        "status": "formal_confirmation_verified_qa_outcomes_frozen",
        "protocol_commit": PROTOCOL_COMMIT,
        "cohort_commit": COHORT_COMMIT,
        "raw_qa_outcome_commit": RAW_QA_OUTCOME_COMMIT,
        "config_sha256": file_sha256(args.config),
        "input_integrity": integrity,
        "output_integrity": verified_integrity,
        "verifier": {
            "model": semantic_selection["nli_model"],
            "config": verifier_config,
            "config_sha256": file_sha256(args.semantic_config),
            "semantic_checker_sha256": file_sha256(
                ROOT / "src" / "medical_rag" / "agentic" / "semantic_evidence_checker.py"
            ),
            "lexical_checker_sha256": file_sha256(
                ROOT / "src" / "medical_rag" / "agentic" / "evidence_checker.py"
            ),
            "evidence_scope": "top1_selected_case_findings_and_impression",
            "changed_from_v5": False,
            "human_validated_clinical_correctness": False,
        },
        "metrics": summarize(final_rows),
        "runtime": {
            "device": args.device,
            "batch_size": args.batch_size,
            "pending_at_start": len(pending),
            "verified_this_run": len(pending),
            "verification_seconds": elapsed,
            "records_per_second": len(pending) / elapsed if elapsed else None,
        },
        "outputs": {
            "input_rows_sha256": file_sha256(args.generations),
            "input_summary_sha256": file_sha256(args.generation_summary),
            "verified_rows_sha256": file_sha256(output_path),
            "verified_row_count": len(final_rows),
            "verified_rows": str(output_path.relative_to(ROOT)).replace("\\", "/"),
            "summary": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "input_summary_status": generation_summary["status"],
        "claim_boundary": (
            "Automated semantic verification is not human clinical adjudication; "
            "verified Token-F1 measures frozen-reference consistency after automated filtering."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
