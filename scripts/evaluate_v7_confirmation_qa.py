from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.semantic_evidence_checker import (  # noqa: E402
    MedicalNLIPredictor,
    check_semantic_evidence_support,
)
from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1  # noqa: E402
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl  # noqa: E402


PROTOCOL_COMMIT = "4821f38"
COHORT_COMMIT = "25a39d8"
RETRIEVAL_RESULT_COMMIT = "ff629f4"

DEFAULT_CONFIG = ROOT / "config" / "v7_confirmation.json"
DEFAULT_V6_CONFIG = ROOT / "config" / "v6_confirmation.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_SEMANTIC_CONFIG = (
    ROOT
    / "experiments"
    / "final_optimized"
    / "semantic_agent"
    / "semantic_agent_selection.json"
)
DEFAULT_GENERATIONS = (
    ROOT / "experiments" / "post_submission_v7" / "v7_confirmation_qa_raw_rows.jsonl"
)
DEFAULT_GENERATION_SUMMARY = (
    ROOT / "experiments" / "post_submission_v7" / "v7_confirmation_qa_raw_summary.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "post_submission_v7"
EXPECTED_SYSTEMS = {
    "bm25_medgemma_1_5",
    "global_alpha_0_52_medgemma_1_5",
    "adaptive_alpha_q_medgemma_1_5",
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


def evidence_text(case: Mapping[str, Any] | None) -> str:
    if not case:
        return ""
    return "\n".join(
        [
            f"Case ID: {case.get('case_id', '')}",
            f"Findings: {case.get('findings', '')}",
            f"Impression: {case.get('impression', '')}",
        ]
    )


def source_balanced(values: Sequence[Mapping[str, Any]], metric: str) -> float:
    by_case_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in values:
        by_case_type[str(row["case_id"])][str(row["question_type"])].append(float(row[metric]))
    case_scores = []
    for type_values in by_case_type.values():
        findings = mean(type_values["case_scoped_findings"])
        impression = mean(type_values["case_scoped_impression"])
        summary = mean(type_values["case_scoped_summary"])
        case_scores.append(0.50 * findings + 0.25 * impression + 0.25 * summary)
    return mean(case_scores) if case_scores else 0.0


def validate_inputs(
    *,
    config_path: Path,
    v6_config_path: Path,
    semantic_config_path: Path,
    generation_path: Path,
    generation_summary_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    for commit in (PROTOCOL_COMMIT, COHORT_COMMIT, RETRIEVAL_RESULT_COMMIT):
        if not commit_exists(commit):
            raise RuntimeError(f"Required frozen commit is unavailable: {commit}")
    config = read_json(config_path)
    v6_config = read_json(v6_config_path)
    semantic_config = read_json(semantic_config_path)
    generation_summary = read_json(generation_summary_path)

    if generation_summary["status"] != "formal_confirmation_raw_qa_outcomes_frozen":
        raise RuntimeError("The V7 raw QA outcome is not frozen.")
    if generation_summary["protocol_commit"] != PROTOCOL_COMMIT:
        raise RuntimeError("The V7 QA protocol commit changed.")
    if generation_summary["cohort_commit"] != COHORT_COMMIT:
        raise RuntimeError("The V7 QA cohort commit changed.")
    if generation_summary["retrieval_result_commit"] != RETRIEVAL_RESULT_COMMIT:
        raise RuntimeError("The V7 retrieval result commit changed.")
    if generation_summary["outputs"]["rows_sha256"] != file_sha256(generation_path):
        raise RuntimeError("The V7 raw QA rows do not match their frozen summary.")
    if generation_summary["outputs"]["row_count"] != 1080:
        raise RuntimeError("The V7 raw QA matrix is incomplete.")
    if generation_summary["config_sha256"] != file_sha256(config_path):
        raise RuntimeError("The raw QA used a different V7 config.")

    v6_verification = v6_config["verification"]
    selected = semantic_config["selected_config"]
    if semantic_config["nli_model"] != v6_verification["model"]:
        raise RuntimeError("The frozen verifier model changed.")
    if file_sha256(semantic_config_path) != v6_verification["config_sha256"]:
        raise RuntimeError("The frozen verifier config hash changed.")
    for key in ("lexical_weight", "support_threshold", "entailment_threshold", "contradiction_threshold"):
        if float(v6_verification[key]) != float(selected[key]):
            raise RuntimeError(f"The frozen verifier setting changed: {key}")
    if v6_verification["changed_from_v5"] is not False:
        raise RuntimeError("The V7 verifier is no longer the frozen V5 verifier.")
    if config["secondary_qa"]["verifier"] != "reuse_v6_verifier":
        raise RuntimeError("The V7 verifier policy changed.")
    return config, v6_config, semantic_config, generation_summary


def validate_generation_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    keys = [(str(row["system"]), str(row["qid"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate (system, qid) rows found in V7 raw QA.")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["system"])].append(row)
    if set(grouped) != EXPECTED_SYSTEMS or any(len(values) != 360 for values in grouped.values()):
        raise RuntimeError("The V7 raw QA matrix has unexpected system counts.")
    qid_sets = [set(str(row["qid"]) for row in values) for values in grouped.values()]
    if any(current != qid_sets[0] for current in qid_sets[1:]):
        raise RuntimeError("V7 raw QA systems do not contain identical qid sets.")
    return grouped


def completed_keys(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    rows = read_jsonl(path)
    keys = [(str(row["system"]), str(row["qid"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("The partial V7 verified output contains duplicate keys.")
    return set(keys)


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["system"])].append(row)
    result: dict[str, Any] = {}
    for system, system_rows in sorted(grouped.items()):
        result[system] = {
            "row_count": len(system_rows),
            "case_count": len({str(row["case_id"]) for row in system_rows}),
            "draft_token_f1": mean(float(row["draft_token_f1"]) for row in system_rows),
            "verified_token_f1": mean(float(row["final_token_f1"]) for row in system_rows),
            "source_balanced_verified_token_f1": source_balanced(system_rows, "final_token_f1"),
            "evidence_support_rate": mean(float(row["support_rate"]) for row in system_rows),
            "final_abstention_rate": mean(float(row["agent_abstained"]) for row in system_rows),
            "revision_rate": mean(float(row["revised"]) for row in system_rows),
            "exact_match": mean(float(row["exact_match"]) for row in system_rows),
            "retrieval_top1_accuracy": mean(
                float(row["retrieval_rank_correct"]) for row in system_rows
            ),
            "nli_contradiction_count": sum(
                int(row["nli_contradiction_count"]) for row in system_rows
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the frozen V6 semantic verifier to V7 QA.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v6-config", type=Path, default=DEFAULT_V6_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--semantic-config", type=Path, default=DEFAULT_SEMANTIC_CONFIG)
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--generation-summary", type=Path, default=DEFAULT_GENERATION_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "v7_confirmation_qa_verified_rows.jsonl"
    summary_path = args.output_dir / "v7_confirmation_qa_verified_summary.json"
    if summary_path.exists():
        raise RuntimeError("The V7 verified QA summary already exists; refusing rerun.")

    config, v6_config, semantic_config, generation_summary = validate_inputs(
        config_path=args.config,
        v6_config_path=args.v6_config,
        semantic_config_path=args.semantic_config,
        generation_path=args.generations,
        generation_summary_path=args.generation_summary,
    )
    generation_rows = read_jsonl(args.generations)
    grouped = validate_generation_rows(generation_rows)
    cases = {str(case["case_id"]): case for case in load_cases_jsonl(args.cases)}
    verifier_config = semantic_config["selected_config"]
    predictor = MedicalNLIPredictor(
        semantic_config["nli_model"],
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=True,
    )

    existing = completed_keys(output_path)
    expected = {(str(row["system"]), str(row["qid"])) for row in generation_rows}
    if not existing.issubset(expected):
        raise RuntimeError("The partial V7 verified output contains an unexpected task.")
    pending = [
        row
        for row in generation_rows
        if (str(row["system"]), str(row["qid"])) not in existing
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
                value.contradiction_probability >= float(verifier_config["contradiction_threshold"])
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
    final_grouped = validate_generation_rows(final_rows)
    if len(final_rows) != 1080:
        raise RuntimeError("The V7 verified QA matrix is incomplete.")
    summary = {
        "experiment": "V7 secondary MedGemma QA with frozen V6 semantic verifier",
        "status": "formal_confirmation_verified_qa_outcomes_frozen",
        "protocol_commit": PROTOCOL_COMMIT,
        "cohort_commit": COHORT_COMMIT,
        "retrieval_result_commit": RETRIEVAL_RESULT_COMMIT,
        "config_sha256": file_sha256(args.config),
        "v6_config_sha256": file_sha256(args.v6_config),
        "semantic_config_sha256": file_sha256(args.semantic_config),
        "semantic_checker_sha256": file_sha256(
            ROOT / "src" / "medical_rag" / "agentic" / "semantic_evidence_checker.py"
        ),
        "generation_summary_sha256": file_sha256(args.generation_summary),
        "generation_rows_sha256": file_sha256(args.generations),
        "implementation_sha256": file_sha256(Path(__file__)),
        "input_row_count": len(generation_rows),
        "output_row_count": len(final_rows),
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
            "verified_rows": str(output_path.relative_to(ROOT)).replace("\\", "/"),
            "verified_rows_sha256": file_sha256(output_path),
            "verified_row_count": len(final_rows),
            "summary": str(summary_path.relative_to(ROOT)).replace("\\", "/"),
        },
        "input_summary_status": generation_summary["status"],
        "system_counts": {system: len(rows) for system, rows in sorted(final_grouped.items())},
        "verifier": {
            "model": semantic_config["nli_model"],
            "config": verifier_config,
            "config_sha256": file_sha256(args.semantic_config),
            "changed_from_v5": False,
            "human_validated_clinical_correctness": False,
        },
        "claim_boundary": (
            "Automated semantic verification is not human clinical adjudication; verified Token-F1 "
            "measures frozen-reference consistency after automated filtering."
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
