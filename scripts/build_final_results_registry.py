from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.benchmark_v2_validity import audit_human_evaluation


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def sha256(relative: str) -> str:
    digest = hashlib.sha256()
    with (ROOT / relative).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_system(rows: list[dict[str, Any]], system: str) -> dict[str, Any]:
    return next(row for row in rows if row["system"] == system)


def find_pair(
    rows: list[dict[str, Any]], system_a: str, system_b: str
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["system_a"] == system_a and row["system_b"] == system_b
    )


def assert_same(left: float, right: float, label: str) -> None:
    if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Result mismatch for {label}: {left} != {right}")


def build_registry() -> dict[str, Any]:
    decisions = load_json("config/submission_decisions.json")
    human_policy = decisions["human_evaluation"]
    v1 = load_json("experiments/final_optimized/final_test/final_optimized_test_summary.json")
    statistics = load_json(
        "experiments/final_optimized/statistics/held_out_test_grouped_bootstrap.json"
    )
    v1_ci = find_system(statistics["summary"], v1["system"])
    assert_same(v1["verified_token_f1"], v1_ci["mean_token_f1"], "V1 final F1")
    pair = find_pair(
        statistics["pairwise"],
        "final_adaptive_direct_semantic_agent",
        "case_bm25_top1_semantic_agent",
    )
    hybrid = load_json(
        "experiments/final_optimized/retrieval/hybrid_alpha_selection.json"
    )
    adaptive = load_json(
        "experiments/final_optimized/adaptive_retrieval/adaptive_policy_selection.json"
    )
    v1_validity = load_json(
        "experiments/final_optimized/validity_audit/research_validity_audit.json"
    )
    oracle = load_json(
        "experiments/final_optimized/oracle_test/final_optimized_test_summary.json"
    )
    stress = load_json(
        "experiments/final_optimized/verifier_stress_test/development_polarity_stress_test.json"
    )
    v2 = load_json(
        "experiments/benchmark_v2/confirmation_evaluation/test_generation_summary.json"
    )
    v2_validity = load_json(
        "experiments/benchmark_v2/validity_audit/benchmark_v2_validity_audit.json"
    )
    assert_same(
        v2["verified_token_f1"],
        v2_validity["confirmation"]["qwen_verified_token_f1"],
        "V2 confirmation Qwen F1",
    )
    v2_benchmark = load_json("data/processed/openi_case_scoped_benchmark_v2.json")
    v2_confirmation = load_json("data/processed/openi_case_scoped_confirmation_v2.json")

    with (ROOT / "data/raw/indiana_projections.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        image_rows = sum(1 for _ in csv.DictReader(handle))
    openi_cases = sum(
        1
        for line in (ROOT / "data/processed/openi_cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
    v1_human = audit_human_evaluation(
        ROOT
        / "experiments/final_optimized/human_evaluation/held_out_blinded_human_evaluation_36.csv"
    )
    v2_human = audit_human_evaluation(
        ROOT
        / "experiments/benchmark_v2/human_evaluation/v2_confirmation_blinded_human_evaluation_36.csv"
    )

    locked_artifacts = [
        "experiments/final_optimized/final_test/final_optimized_test_summary.json",
        "experiments/final_optimized/statistics/held_out_test_grouped_bootstrap.json",
        "experiments/final_optimized/validity_audit/research_validity_audit.json",
        "experiments/benchmark_v2/confirmation_evaluation/test_generation_summary.json",
        "experiments/benchmark_v2/validity_audit/benchmark_v2_validity_audit.json",
        "data/processed/openi_case_scoped_benchmark_v2.json",
        "data/processed/openi_case_scoped_confirmation_v2.json",
    ]
    return {
        "registry": "WQF7023 final results registry",
        "updated": human_policy["decision_date"],
        "dataset": {
            "openi_report_cases": openi_cases,
            "openi_image_mapping_rows": image_rows,
            "modeled_input": "radiology report text only",
        },
        "v1_open_corpus_stress_test": {
            "case_count": v1_ci["case_count"],
            "question_count": v1["n"],
            "verified_token_f1": v1["verified_token_f1"],
            "case_bootstrap_95_ci": [v1_ci["ci_low_95"], v1_ci["ci_high_95"]],
            "evidence_support_rate": v1["evidence_support_rate"],
            "final_abstention_rate": v1["final_abstention_rate"],
            "selected_hybrid_alpha": hybrid["selected_alpha"],
            "retrieval_test_selective_accuracy": adaptive["held_out_test"][
                "selective_accuracy"
            ],
            "retrieval_test_abstention_rate": adaptive["held_out_test"]["abstention_rate"],
            "oracle_verified_token_f1": oracle["verified_token_f1"],
            "ambiguous_query_rate": v1_validity["benchmark_ambiguity"]["held_out_test"][
                "ambiguous_question_rate"
            ],
            "final_minus_case_bm25": {
                "difference": pair["mean_difference"],
                "ci_95": [pair["ci_low"], pair["ci_high"]],
                "paired_randomization_p": pair["paired_randomization_p"],
                "holm_adjusted_p": pair["holm_adjusted_randomization_p"],
            },
            "development_polarity_stress_rejection_rate": stress[
                "semantic_contradiction_rejected_rate"
            ],
            "interpretation": "retrieval-limited open-corpus failure analysis",
        },
        "v2_controlled_case_scoped_workflow": {
            "main_case_count": v2_benchmark["case_count"],
            "confirmation_case_count": v2["case_count"],
            "confirmation_question_count": v2["n"],
            "locked_top_k": v2["top_k"],
            "qwen_token_f1": v2["verified_token_f1"],
            "qwen_case_bootstrap_95_ci": v2[
                "verified_token_f1_case_bootstrap_95_ci"
            ],
            "extractive_context_token_f1": v2_validity["confirmation"][
                "extractive_retrieved_context_token_f1"
            ],
            "qwen_minus_extractive": v2_validity["confirmation"][
                "qwen_minus_extractive_token_f1"
            ],
            "mean_evidence_recall": v2["mean_retrieval_recall"],
            "routed_candidate_pool_equals_qrels_rate": v2_validity["confirmation"][
                "routed_candidate_pool_equals_qrels_rate"
            ],
            "verifier_action": "audit_only",
            "content_fingerprint_main": v2_benchmark["content_fingerprint_sha256"],
            "content_fingerprint_confirmation": v2_confirmation[
                "content_fingerprint_sha256"
            ],
            "interpretation": "controlled case-isolation and workflow-safety benchmark",
        },
        "human_evaluation": {"v1": v1_human, "v2": v2_human},
        "human_evaluation_policy": human_policy,
        "claim_controls": {
            "v1_v2_are_paired": False,
            "image_pixels_are_model_inputs": False,
            "authentication_is_implemented": False,
            "routing_is_learned_or_autonomous": False,
            "clinical_validation_is_complete": False,
        },
        "locked_artifact_sha256": {
            relative: sha256(relative) for relative in locked_artifacts
        },
    }


def render_markdown(registry: dict[str, Any]) -> str:
    v1 = registry["v1_open_corpus_stress_test"]
    v2 = registry["v2_controlled_case_scoped_workflow"]
    human = registry["human_evaluation"]
    human_policy = registry["human_evaluation_policy"]
    return f"""# Final Results Registry

Generated from locked artifacts on {registry['updated']}.

## Dataset

- OpenI report cases: {registry['dataset']['openi_report_cases']:,}
- Image mapping rows: {registry['dataset']['openi_image_mapping_rows']:,}
- Modeled input: {registry['dataset']['modeled_input']}

## V1 Open-Corpus Stress Test

| Measure | Locked value |
|---|---:|
| Held-out cases / questions | {v1['case_count']} / {v1['question_count']} |
| Verified Token-F1 | {v1['verified_token_f1']:.3f} |
| Case-bootstrap 95% CI | [{v1['case_bootstrap_95_ci'][0]:.3f}, {v1['case_bootstrap_95_ci'][1]:.3f}] |
| Oracle-retrieval verified Token-F1 | {v1['oracle_verified_token_f1']:.3f} |
| Ambiguous held-out queries | {v1['ambiguous_query_rate']:.1%} |
| Retrieval abstention | {v1['retrieval_test_abstention_rate']:.1%} |
| Final minus Case-BM25 | {v1['final_minus_case_bm25']['difference']:+.3f} |
| Holm-adjusted p | {v1['final_minus_case_bm25']['holm_adjusted_p']:.4f} |

Interpretation: {v1['interpretation']}.

## V2 Controlled Case-Scoped Workflow

| Measure | Locked value |
|---|---:|
| Confirmation cases / questions | {v2['confirmation_case_count']} / {v2['confirmation_question_count']} |
| Locked top-k | {v2['locked_top_k']} |
| Evidence recall | {v2['mean_evidence_recall']:.1%} |
| Extractive-context Token-F1 | {v2['extractive_context_token_f1']:.3f} |
| Qwen Token-F1 | {v2['qwen_token_f1']:.3f} |
| Qwen minus extractive | {v2['qwen_minus_extractive']:+.3f} |
| Routed candidates equal qrels | {v2['routed_candidate_pool_equals_qrels_rate']:.0%} |

Interpretation: {v2['interpretation']}. Routed Hit@1 is not a semantic-retrieval claim.

## Human-Evaluation Disposition

- V1 completed blinded rows: {human['v1']['completed_rows']}/{human['v1']['rows']}
- V2 completed blinded rows: {human['v2']['completed_rows']}/{human['v2']['rows']}
- Status: {human_policy['status']}
- Decision date: {human_policy['decision_date']}
- Reason: {human_policy['reason']}

No human result is claimed or inferred from automatic metrics. The empty blinded packages are retained as an unexecuted protocol, and the absence of independent review is reported as a limitation.
"""


def main() -> None:
    registry = build_registry()
    output_dir = ROOT / "experiments/final_submission"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_results_registry.json").write_text(
        json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ROOT / "docs/FINAL_RESULTS_REGISTRY.md").write_text(
        render_markdown(registry), encoding="utf-8"
    )
    print(json.dumps(registry, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
