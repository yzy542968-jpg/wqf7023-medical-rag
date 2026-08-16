from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.benchmark_v2_validity import (
    audit_cohort,
    audit_human_evaluation,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit structural validity of benchmark V2.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments"
        / "benchmark_v2"
        / "validity_audit"
        / "benchmark_v2_validity_audit.json",
    )
    args = parser.parse_args()

    main_benchmark = json.loads(
        (ROOT / "data" / "processed" / "openi_case_scoped_benchmark_v2.json").read_text(
            encoding="utf-8"
        )
    )
    confirmation_benchmark = json.loads(
        (
            ROOT / "data" / "processed" / "openi_case_scoped_confirmation_v2.json"
        ).read_text(encoding="utf-8")
    )
    test = audit_cohort(
        main_benchmark,
        "test",
        read_jsonl(
            ROOT
            / "data"
            / "processed"
            / "prompt_packs"
            / "benchmark_v2"
            / "test_case_scoped_routed.jsonl"
        ),
        read_jsonl(
            ROOT
            / "experiments"
            / "benchmark_v2"
            / "retrieval"
            / "test_retrieval_rows.jsonl"
        ),
        read_jsonl(
            ROOT
            / "experiments"
            / "benchmark_v2"
            / "final_test_evaluation"
            / "test_generation_rows.jsonl"
        ),
    )
    confirmation = audit_cohort(
        confirmation_benchmark,
        "confirmation",
        read_jsonl(
            ROOT
            / "data"
            / "processed"
            / "prompt_packs"
            / "benchmark_v2"
            / "confirmation_case_scoped_routed.jsonl"
        ),
        read_jsonl(
            ROOT
            / "experiments"
            / "benchmark_v2"
            / "confirmation_retrieval"
            / "confirmation_retrieval_rows.jsonl"
        ),
        read_jsonl(
            ROOT
            / "experiments"
            / "benchmark_v2"
            / "confirmation_evaluation"
            / "test_generation_rows.jsonl"
        ),
    )
    human = audit_human_evaluation(
        ROOT
        / "experiments"
        / "benchmark_v2"
        / "human_evaluation"
        / "v2_confirmation_blinded_human_evaluation_36.csv"
    )
    payload = {
        "audit": "Benchmark V2 structural validity audit",
        "version": "1.0",
        "initial_test_status": (
            "diagnostic after inspection; it was viewed before verifier action calibration"
        ),
        "confirmation_status": (
            "disjoint outcome-independent confirmation cohort and primary final automated evidence"
        ),
        "test": test,
        "confirmation": confirmation,
        "human_evaluation": human,
        "implemented_security_scope": (
            "explicit case-ID metadata filtering only; no authentication or access-control layer"
        ),
        "valid_claims": [
            "Patient/case metadata filtering prevents cross-case retrieval in the V2 workflow.",
            "The locked top-k usually covers the report section selected by the deterministic route.",
            "Advisory verification preserves output when automatic rewriting is not calibration-safe.",
        ],
        "invalid_or_unsupported_claims": [
            "Routed Hit@1 does not demonstrate semantic retrieval because routed candidates equal qrels.",
            "The generator does not improve this benchmark over returning retrieved evidence.",
            "The deterministic type-to-section rule is not learned or autonomous planning.",
            "Clinical correctness and safety remain unsupported until blinded human evaluation is complete.",
        ],
        "priority_decision": (
            "Retain V2 as a controlled workflow and safety benchmark. Use a new independently "
            "authored QA benchmark with hard negatives and unanswerable questions for the main thesis claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

