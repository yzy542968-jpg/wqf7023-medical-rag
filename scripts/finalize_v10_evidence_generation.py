from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.evaluation.answer_metrics import token_f1  # noqa: E402
from medical_rag.similar_case.v10_evidence import EvidenceUnit  # noqa: E402
from medical_rag.similar_case.v10_generation import (  # noqa: E402
    assemble_deterministic_output,
    deterministic_historical_evidence,
    normalize_bounded_answer,
    parse_plain_answer,
)
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from run_v10_evidence_generation_development import (  # noqa: E402
    read_json,
    read_jsonl,
    select_policy,
    summarize,
)


DEFAULT_CONFIG = ROOT / "config" / "v10_evidence_answer_normalization.json"
DEFAULT_REVISION_CONFIG = ROOT / "config" / "v10_evidence_generation_revision1.json"
DEFAULT_ROWS = ROOT / "experiments" / "v10_publication" / "v10_evidence_generation_revision1_rows.jsonl"
DEFAULT_OUTPUT = ROOT / "experiments" / "v10_publication" / "v10_evidence_generation_final_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_evidence_generation_final_summary.json"


def unit_from_row(row: dict[str, object]) -> EvidenceUnit:
    return EvidenceUnit(
        case_id=str(row["case_id"]),
        section=str(row["section"]),
        unit_type=str(row["unit_type"]),
        unit_index=int(row["unit_index"]),
        text=str(row["text"]),
        source_sha256=str(row["source_sha256"]),
        score=float(row["score"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize bounded V10 evidence answers.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--revision-config", type=Path, default=DEFAULT_REVISION_CONFIG)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    config = read_json(args.config)
    revision = read_json(args.revision_config)
    if config["validation_aggregate_metrics_inspected"] or config["test_outcomes_inspected"]:
        raise RuntimeError("normalization config records forbidden outcomes")
    rows = read_jsonl(args.rows)
    expected = int(config["expected_revision1_rows"])
    if len(rows) != expected:
        raise RuntimeError(f"Revision 1 rows are incomplete: {len(rows)} != {expected}")

    finalized = []
    for row in rows:
        parsed = parse_plain_answer(str(row["raw_answer_stage"]), stop_token="<end_of_turn>")
        bounded = normalize_bounded_answer(
            parsed["answer"],
            maximum_complete_sentences=int(config["maximum_complete_sentences"]),
        )
        answer_stage = parse_plain_answer(bounded)
        evidence = [unit_from_row(value) for value in row["evidence"]]
        support = deterministic_historical_evidence(
            evidence,
            query="\n".join((str(row["question"]), answer_stage["answer"])),
            retrieved_case_ids=row["retrieved_case_ids"],
            maximum_units=int(revision["provenance_attachment"]["maximum_units"]),
        )
        assembled = assemble_deterministic_output(
            answer_stage,
            support,
            no_reliable_history=False,
        )
        allowed = {unit.provenance_id for unit in evidence}
        citation_valid = all(
            item["provenance_id"] in allowed for item in assembled["historical_support"]
        )
        finalized.append(
            {
                **row,
                **assembled,
                "citation_valid": citation_valid,
                "token_f1": token_f1(assembled["answer"], str(row["reference_answer"])),
                "answer_normalization": "first_two_complete_sentences_else_available_text",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in finalized) + "\n",
        encoding="utf-8",
    )
    policies = revision["evidence_policies"]
    metrics = summarize(finalized, policies)
    for policy in policies:
        selected = [row for row in finalized if row["evidence_policy"] == policy]
        metrics[policy]["answer_token_ceiling_rate"] = sum(
            bool(row["hit_answer_token_ceiling"]) for row in selected
        ) / len(selected)
        metrics[policy]["end_of_turn_stop_rate"] = sum(
            bool(row["stopped_on_end_of_turn"]) for row in selected
        ) / len(selected)
    selected_policy = select_policy(metrics, revision)
    summary = {
        "study": "V10 finalized evidence and compact generation development",
        "status": "development_complete_test_not_run",
        "inputs": {
            "normalization_config_sha256": file_sha256(args.config),
            "revision_config_sha256": file_sha256(args.revision_config),
            "revision1_rows_sha256": file_sha256(args.rows),
        },
        "counts": {
            "validation_cases": len({str(row["case_id"]) for row in finalized}),
            "question_rows": len(finalized) // len(policies),
            "finalized_rows": len(finalized),
        },
        "metrics": metrics,
        "selected_evidence_policy": selected_policy,
        "final_rows_sha256": file_sha256(args.output),
        "test_outcomes_inspected": False,
        "claim_boundary": "Automated same-source report-reference consistency, not clinical correctness.",
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
