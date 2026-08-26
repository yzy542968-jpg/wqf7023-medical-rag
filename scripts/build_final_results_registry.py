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
    base = f"""# Final Results Registry

Generated from locked artifacts and development summaries on 2026-08-26.

## Dataset

- OpenI report cases: {registry['dataset']['openi_report_cases']:,}
- Image mapping rows: {registry['dataset']['openi_image_mapping_rows']:,}
- Final V10 modeled input: target chest image(s), indication and question; retrieved evidence comes from other-case historical image-report pairs

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
    return base.rstrip() + "\n\n" + render_v9_v11_sections()


def render_v9_v11_sections() -> str:
    v9 = {
        "split": load_json("data/splits/v9/v9_full_source_split_freeze.json"),
        "retrieval": load_json("data/splits/v9/v9_retrieval_confirmation_summary.json"),
        "qa": load_json("data/splits/v9/v9_qa_confirmation_summary.json"),
    }
    v10 = {
        "split": load_json("data/splits/v10/v10_cluster_disjoint_split_freeze.json"),
        "retrieval": load_json("data/splits/v10/v10_confirmation_retrieval_summary.json"),
        "qa": load_json("data/splits/v10/v10_confirmation_qa_summary.json"),
        "radgraph": load_json("data/splits/v10/v10_radgraph_metrics_summary.json"),
        "qrel": load_json("data/splits/v10/v10_qrel_sensitivity_summary.json"),
    }
    v11 = {
        "evidence": load_json("data/splits/v11/v11_development_evidence_ablation_summary.json"),
        "candidates_100": load_json("data/splits/v11/v11_candidate_generation_audit_summary.json"),
        "candidates_200": load_json("data/splits/v11/v11_candidate_generation_audit_k200_summary.json"),
        "planner": load_json("data/splits/v11/v11_question_planner_benchmark_summary.json"),
        "planner_reserved": load_json("data/splits/v11/v11_question_planner_reserved_summary.json"),
        "generation": load_json("data/splits/v11/v11_medgemma_generation_48_clean_summary.json"),
        "statistics": load_json("data/splits/v11/v11_medgemma_generation_48_statistical_summary.json"),
    }
    v9_parts = v9["split"]["partition_fingerprints"]
    v9_retrieval = v9["retrieval"]
    v9_qa = v9["qa"]
    v10_parts = v10["split"]["partition_fingerprints"]
    retrieval = v10["retrieval"]
    qa = v10["qa"]
    radgraph = v10["radgraph"]
    qrel_abnormal = v10["qrel"]["metrics"]
    evidence = v11["evidence"]
    candidates_100 = v11["candidates_100"]
    candidates_200 = v11["candidates_200"]
    planner = v11["planner"]
    planner_reserved = v11["planner_reserved"]
    generation = v11["generation"]
    statistics = v11["statistics"]["primary_case_to_fact_minus_whole_report"]
    v9_pair = v9_retrieval["primary_comparison_r4_minus_r1"]
    v10_pair = retrieval["primary_r5_minus_r4"]
    g2_g0 = qa["primary_comparisons"]["g2_minus_g0_token_f1"]
    g2_g1 = qa["primary_comparisons"]["g2_minus_g1_token_f1"]
    combined_abnormal = qrel_abnormal["combined_0.6_label_0.4_fact"][
        "r5_minus_r4_by_report_index_class"
    ]["abnormal"]
    label_abnormal = qrel_abnormal["label_only"]["r5_minus_r4_by_report_index_class"][
        "abnormal"
    ]
    token_difference = statistics["token_f1"]
    graph_difference = statistics["f1_radgraph_complete"]
    return f"""## V9 Historical Final Study

| Measure | Frozen value |
|---|---:|
| Source cases | 3,851 |
| Train / Validation / Test | {v9_parts['train']['case_count']:,} / {v9_parts['validation']['case_count']:,} / {v9_parts['test']['case_count']:,} |
| Learned reranker nDCG@10 | {v9_retrieval['metrics']['r4_learned_mlp']['ndcg@10']:.4f} |
| Image-only nDCG@10 | {v9_retrieval['metrics']['r1_image_image']['ndcg@10']:.4f} |
| Learned minus image-only, 95% CI | {v9_pair['difference']:+.5f} [{v9_pair['ci_95_low']:.5f}, {v9_pair['ci_95_high']:.5f}] |
| Learned multimodal RAG Token-F1 | {v9_qa['metrics']['g3_learned_multimodal_rag']['token_f1']:.4f} |
| Learned RAG minus no retrieval, 95% CI | {v9_qa['primary_comparison_g3_minus_g0']['difference']:+.5f} [{v9_qa['primary_comparison_g3_minus_g0']['ci_95_low']:.5f}, {v9_qa['primary_comparison_g3_minus_g0']['ci_95_high']:.5f}] |

V9 is retained as historical evidence. Its post-hoc similarity audit motivated duplicate clustering before the V10 split.

## V10 Final Primary Study

| Measure | Frozen value |
|---|---:|
| Source cases / duplicate clusters | 3,851 / 3,013 |
| Train / Calibration / Validation / Test | {v10_parts['train']['case_count']:,} / {v10_parts['calibration']['case_count']:,} / {v10_parts['validation']['case_count']:,} / {v10_parts['test']['case_count']:,} |
| Technically eligible Test cases | {retrieval['counts']['test_cases']} |
| R4 nDCG@10 | {retrieval['metrics']['r4_nine_feature']['ndcg@10']:.5f} |
| R5 nDCG@10 | {retrieval['metrics']['r5_fact_attention']['ndcg@10']:.5f} |
| R5 minus R4, case-bootstrap 95% CI | {v10_pair['mean_difference']:+.5f} [{v10_pair['ci_95_low']:.5f}, {v10_pair['ci_95_high']:.5f}] |
| Post-hoc abnormal combined-qrel R5 minus R4 | {combined_abnormal['difference']:+.5f} [{combined_abnormal['ci95_case_bootstrap'][0]:.5f}, {combined_abnormal['ci95_case_bootstrap'][1]:.5f}] |
| Post-hoc abnormal label-only R5 minus R4 | {label_abnormal['difference']:+.5f} [{label_abnormal['ci95_case_bootstrap'][0]:.5f}, {label_abnormal['ci95_case_bootstrap'][1]:.5f}] |
| Correct-image / shuffled mean nDCG@10 | {retrieval['alignment_control']['aligned_mean_ndcg@10']:.5f} / {retrieval['alignment_control']['shuffled_mean_ndcg@10']:.5f} |
| Shuffled-image plus-one Monte Carlo p | {retrieval['alignment_control']['plus_one_monte_carlo_p']:.5f} |
| Retrieval confidence Brier / ECE / AUROC | {retrieval['calibration']['metrics']['brier']:.5f} / {retrieval['calibration']['metrics']['ece_10']:.5f} / {retrieval['calibration']['metrics']['auroc']:.5f} |
| No-history Token-F1 | {qa['metrics']['g0_target_image']['token_f1_equal_question']:.5f} |
| R4 whole-report RAG Token-F1 | {qa['metrics']['g1_whole_report']['token_f1_equal_question']:.5f} |
| R5 whole-report historical RAG Token-F1 | {qa['metrics']['g2_hierarchical']['token_f1_equal_question']:.5f} |
| R5 minus no history, case-bootstrap 95% CI | {g2_g0['mean_difference']:+.5f} [{g2_g0['ci_95_low']:.5f}, {g2_g0['ci_95_high']:.5f}] |
| R5 minus R4, case-bootstrap 95% CI | {g2_g1['mean_difference']:+.5f} [{g2_g1['ci_95_low']:.5f}, {g2_g1['ci_95_high']:.5f}] |
| R5 complete F1RadGraph | {radgraph['systems']['g2_hierarchical']['metrics']['f1_radgraph_complete']['mean']:.5f} |
| Schema / provenance integrity | 100% / 100% |

Interpretation: correctly aligned images improved report-derived similar-case retrieval, fact-aware reranking produced a small confirmed aggregate gain, and historical retrieval transferred to report-reference-consistent generation. Post-hoc qrel sensitivity showed spectrum-dependent behavior and feature-metric coupling; it does not support a uniform clinical-similarity claim. These are automated within-source results, not physician-adjudicated diagnostic accuracy.

## V11 Development Extension

| Measure | Development-only value |
|---|---:|
| Hierarchical evidence audit cases | {evidence['inputs']['validation_case_count']} |
| Mean context reduction | {evidence['evidence_compression']['character_reduction_fraction']:.2%} |
| Provenance completeness | {evidence['evidence_compression']['case_to_fact']['provenance_complete_rate']:.0%} |
| RRF K=100 nDCG@10 / relevant presence | {candidates_100['metrics']['rrf_union']['ndcg10']:.4f} / {candidates_100['metrics']['rrf_union']['has_relevant_in_pool']:.2%} |
| RRF K=200 relevant-item recall | {candidates_200['metrics']['rrf_union']['relevant_recall_at_k']:.2%} |
| Original planner examples / accuracy | {planner['example_count']} / {planner['accuracy']:.4f} |
| Reserved planner examples / accuracy | {planner_reserved['example_count']} / {planner_reserved['accuracy']:.4f} |
| Reserved planner macro-F1 / indication invariance | {planner_reserved['macro_f1']:.4f} / {planner_reserved['indication_invariance_rate']:.4f} |
| Clean MedGemma cases / generations | {generation['metrics']['whole_report']['case_count']} / {generation['counts']['rows']} |
| Whole-report Token-F1 | {generation['metrics']['whole_report']['token_f1_all_rows']:.5f} |
| Sentence-only Token-F1 | {generation['metrics']['sentence_only']['token_f1_all_rows']:.5f} |
| Case-to-fact Token-F1 | {generation['metrics']['case_to_fact']['token_f1_all_rows']:.5f} |
| Case-to-fact minus whole report, 95% CI | {token_difference['mean_difference']:+.5f} [{token_difference['ci_95_low']:.5f}, {token_difference['ci_95_high']:.5f}] |
| Case-to-fact minus whole report complete F1RadGraph, 95% CI | {graph_difference['mean_difference']:+.5f} [{graph_difference['ci_95_low']:.5f}, {graph_difference['ci_95_high']:.5f}] |
| Case-to-fact mean input tokens / evidence characters | {generation['metrics']['case_to_fact']['mean_input_tokens']:.1f} / {generation['metrics']['case_to_fact']['mean_evidence_characters']:.1f} |
| Whole-report mean input tokens / evidence characters | {generation['metrics']['whole_report']['mean_input_tokens']:.1f} / {generation['metrics']['whole_report']['mean_evidence_characters']:.1f} |

Interpretation: V11 supports efficiency, auditability and planner wording robustness. Both primary 48-case quality intervals cross zero, so V11 does not claim confirmed generation superiority. It did not instantiate a confirmation cohort and did not modify V10.

## Final Evidence Boundary

- Independent radiologist review: Future Work; no scores are reported.
- Authorized external patient-level validation: Future Work; the adapter/runbook exists but no result is claimed.
- Retrieval confidence: report-derived research signal, not diagnostic confidence.
- Patient-level independence: not verified because reliable patient identifiers were unavailable in the processed OpenI artifact.
- Clinical safety, treatment utility and deployment performance: outside the completed evidence.
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
