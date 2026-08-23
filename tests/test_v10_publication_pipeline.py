from medical_rag.similar_case.v10_calibration import (
    RetrievalCalibrator,
    calibration_metrics,
    risk_coverage_curve,
    threshold_for_coverage,
)
from medical_rag.similar_case.v10_evidence import (
    evidence_diagnostics,
    select_case_evidence,
)
from medical_rag.similar_case.v10_generation import (
    assemble_deterministic_output,
    assemble_output,
    build_answer_prompt,
    build_plain_answer_prompt,
    deterministic_historical_evidence,
    parse_answer_stage,
    parse_plain_answer,
    parse_support_stage,
)
from medical_rag.similar_case.v10_reranker import (
    FactAwareFeatureIndex,
    augment_r4_features,
    sample_fact_aware_pairs,
)
import numpy as np


CASE = {
    "case_id": "CXR1",
    "findings": "Mild cardiomegaly. No focal consolidation. No pleural effusion.",
    "impression": "Mild cardiomegaly without acute pulmonary disease.",
}
FACTS = {
    "entity|cardiomegaly|observation::definitely present",
    "entity|consolidation|observation::definitely absent",
    "relation|effusion|observation::definitely absent|located_at|pleural",
}


def test_hierarchical_selector_keeps_case_and_section_provenance() -> None:
    units = select_case_evidence(
        CASE,
        query="Is there cardiomegaly?",
        facts=FACTS,
        policy="sentence_top2_fact_top5",
    )
    assert units
    assert all(unit.case_id == "CXR1" for unit in units)
    assert all(unit.provenance_id.startswith("CXR1:") for unit in units)
    assert any("cardiomegaly" in unit.text.lower() for unit in units)
    diagnostics = evidence_diagnostics(units)
    assert diagnostics["unit_count"] == len(units)
    assert 0.0 <= diagnostics["redundancy"] <= 1.0


def test_compact_two_stage_generation_always_assembles_valid_schema() -> None:
    units = select_case_evidence(
        CASE,
        query="Is there cardiomegaly?",
        facts=FACTS,
        policy="sentence_top3",
    )
    prompt = build_answer_prompt(
        indication="Preoperative evaluation",
        question="Is there cardiomegaly?",
        evidence=units,
        no_reliable_history=False,
    )
    assert "analogy only" in prompt
    answer = parse_answer_stage('{"a":"Mild cardiomegaly is present.","u":"medium"}')
    support = parse_support_stage(
        '{"s":[{"p":"' + units[0].provenance_id + '","t":"A similar case reports cardiomegaly."}]}',
        units,
    )
    result = assemble_output(answer, support, no_reliable_history=False)
    assert result["assembled_schema_valid"] is True
    assert result["supporting_case_ids"] == ["CXR1"]


def test_low_confidence_output_removes_historical_support_deterministically() -> None:
    answer = parse_answer_stage("truncated but clinically cautious answer")
    support = {"historical_support": [{"case_id": "CXR1"}], "support_stage_valid": False}
    result = assemble_output(answer, support, no_reliable_history=True)
    assert result["assembled_schema_valid"] is True
    assert result["historical_support"] == []
    assert result["evidence_abstained"] is True


def test_plain_answer_and_deterministic_provenance_are_complete() -> None:
    units = select_case_evidence(
        CASE,
        query="Is there cardiomegaly?",
        facts=FACTS,
        policy="sentence_top3",
    )
    prompt = build_plain_answer_prompt(
        indication="Preoperative evaluation",
        question="Is there cardiomegaly?",
        evidence=units,
        no_reliable_history=False,
    )
    assert "Return only the concise answer" in prompt
    answer = parse_plain_answer("Mild cardiomegaly.<end_of_turn><unused94>thought")
    support = deterministic_historical_evidence(
        units,
        query="cardiomegaly",
        retrieved_case_ids=["CXR1"],
    )
    result = assemble_deterministic_output(answer, support, no_reliable_history=False)
    assert result["answer"] == "Mild cardiomegaly."
    assert result["supporting_case_ids"] == ["CXR1"]
    assert result["assembled_schema_valid"] is True


def test_retrieval_calibration_and_risk_coverage_are_deterministic() -> None:
    rows = [
        {"top1_score": 0.9, "top1_top2_margin": 0.4},
        {"top1_score": 0.8, "top1_top2_margin": 0.3},
        {"top1_score": 0.2, "top1_top2_margin": 0.01},
        {"top1_score": 0.1, "top1_top2_margin": 0.00},
    ]
    labels = [1, 1, 0, 0]
    calibrator = RetrievalCalibrator().fit(rows, labels)
    probabilities = calibrator.predict_proba(rows)
    assert probabilities[0] > probabilities[-1]
    assert threshold_for_coverage(probabilities, 0.5) >= min(probabilities)
    curve = risk_coverage_curve(labels, probabilities)
    assert len(curve) == len(rows)
    metrics = calibration_metrics(labels, probabilities)
    assert 0.0 <= metrics["brier"] <= 1.0
    assert 0.0 <= metrics["ece_10"] <= 1.0


def test_fact_aware_index_and_pair_sampler_use_inference_available_features() -> None:
    cases = {
        "CXR1": CASE,
        "CXR2": {
            "case_id": "CXR2",
            "findings": "Pulmonary edema and bilateral pleural effusions.",
            "impression": "Congestive heart failure.",
        },
        "CXR3": {
            "case_id": "CXR3",
            "findings": "The lungs are clear.",
            "impression": "No acute disease.",
        },
    }
    facts = {
        "CXR1": FACTS,
        "CXR2": {"entity|edema|observation::definitely present"},
        "CXR3": {"entity|edema|observation::definitely absent"},
    }
    index = FactAwareFeatureIndex.build(["CXR1", "CXR2", "CXR3"], cases, facts)
    added = index.query_features("Is there pulmonary edema?")
    combined = augment_r4_features(np.zeros((3, 9), dtype=np.float32), added)
    assert combined.shape == (3, 17)
    assert added[1, 0] > added[0, 0]
    high, low, weights = sample_fact_aware_pairs(
        combined,
        np.asarray([0.7, 0.9, 0.1], dtype=np.float32),
        [np.asarray([0.5, 0.8, 0.9]), np.asarray([0.4, 0.7, 0.8])],
        config={
            "component_top_k": 3,
            "relevance_top_k": 3,
            "relevance_bottom_k": 3,
            "hard_negative_k": 2,
            "hard_negative_max_gain": 0.25,
            "high_candidates": 2,
            "low_candidates": 2,
            "minimum_gain_difference": 0.05,
        },
    )
    assert high.shape[1] == low.shape[1] == 17
    assert len(weights) > 0
