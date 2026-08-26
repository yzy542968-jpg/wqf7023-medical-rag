from __future__ import annotations

from medical_rag.similar_case.v11_evidence import evidence_profile, select_case_facts, select_hierarchical_evidence
from medical_rag.similar_case.v11_output_contract import answer_only_generation_prompt, parse_compact_output
from medical_rag.similar_case.v11_qrel import prepare_qrel_case, qrel_v2_profile, qrel_v2_profile_prepared, report_index_spectrum
from medical_rag.similar_case.v11_question_planner import plan_question
from medical_rag.similar_case.v11_selective import compute_retrieval_confidence, fit_proxy_threshold
from medical_rag.similar_case.v11_training import build_pairwise_examples, hard_negative_indices, pairwise_training_mask, target_inside_shortlist
from medical_rag.retrieval.candidate_generation import reciprocal_rank_fusion_union, top_k_union
from medical_rag.evaluation.grouped_bootstrap import grouped_bootstrap_ci


def _case(case_id: str, problems: str = "Pneumonia") -> dict[str, object]:
    return {
        "case_id": case_id,
        "indication": "fever and cough",
        "problems": problems,
        "findings": "There is a mild right lower lobe opacity. No pleural effusion.",
        "impression": "Right lower lobe pneumonia is possible.",
    }


def test_qrel_v2_exposes_structured_components_and_no_empty_credit() -> None:
    query = _case("Q")
    candidate = _case("C")
    facts = {
        "Q": ("entity|pneumonia|observation::definitely present", "entity|right lower lobe|anatomy::definitely present"),
        "C": ("entity|pneumonia|observation::definitely present", "entity|right lower lobe|anatomy::definitely present"),
    }
    profile = qrel_v2_profile(query, candidate, facts)
    assert profile["qrel_v2"] > 0.5
    assert {"lesion_type", "anatomy", "severity", "polarity", "uncertainty", "indication", "report_label"} <= set(profile)
    assert 0.0 < float(profile["availability_fraction"]) <= 1.0
    assert 0.0 <= float(profile["qrel_v2_available_normalized"]) <= 1.0
    assert report_index_spectrum(_case("N", "normal")) == "report_indexed_normal"
    assert report_index_spectrum(_case("U", "no indexing")) == "report_index_indeterminate"
    empty_query = {**_case("E", "no indexing"), "indication": ""}
    empty_candidate = {**_case("F", "no indexing"), "indication": ""}
    empty = qrel_v2_profile(empty_query, empty_candidate, {"E": (), "F": ()})
    assert empty["qrel_v2"] == 0.0
    assert empty["qrel_v2_available_normalized"] == 0.0
    assert empty["availability_fraction"] == 0.0
    prepared = qrel_v2_profile_prepared(
        prepare_qrel_case(query, facts),
        prepare_qrel_case(candidate, facts),
    )
    assert prepared["qrel_v2"] == profile["qrel_v2"]


def test_question_planner_and_hierarchical_selector_preserve_provenance() -> None:
    plan = plan_question("Is there a new left lower lobe opacity?", "shortness of breath")
    assert plan.intent in {"presence", "location", "comparison"}
    case_a = _case("A")
    case_b = _case("B")
    facts = {"A": ("entity|pneumonia|observation::definitely present",), "B": ()}
    selected = select_hierarchical_evidence([case_a, case_b], query="left lower lobe opacity", facts_by_case=facts, plan=plan)
    profile = evidence_profile(selected.units)
    assert selected.units
    assert profile["provenance_complete_rate"] == 1.0
    assert all(unit.case_id in {"A", "B"} and unit.section for unit in selected.units)
    assert plan_question("What is the diagnosis?", "device present").intent == "summary"
    assert plan_question("Is there enough information to answer?").intent == "insufficient_information"


def test_compact_output_rejects_unknown_evidence_ids() -> None:
    case = _case("A")
    plan = plan_question("What are the findings?")
    selected = select_case_facts(case, query="findings", facts=(), plan=plan)
    output = parse_compact_output('{"answer":"There is an opacity.","uncertainty":"medium","abstain":false,"evidence":["unknown"]}', selected)
    assert output["structured_output_valid"] is False
    assert output["evidence"] == []


def test_answer_only_prompt_excludes_serialization_burden() -> None:
    prompt = answer_only_generation_prompt(
        indication="cough", question="What are the findings?", planner_instruction="Intent=summary", evidence=(), abstain=True
    )
    assert "Return only the concise answer" in prompt
    assert "Do not output JSON" in prompt


def test_compact_output_marks_yaml_repair_without_calling_it_json_valid() -> None:
    case = _case("A")
    plan = plan_question("What are the findings?")
    selected = select_case_facts(case, query="findings", facts=(), plan=plan)
    raw = "answer: There is an opacity.\nuncertainty: medium\nabstain: false\nevidence: []"
    output = parse_compact_output(raw, selected)
    assert output["raw_json_valid"] is False
    assert output["parser_repaired"] is True
    assert output["normalized_output_usable"] is True
    assert output["structured_output_valid"] is False


def test_selective_gate_is_deterministic() -> None:
    confidence = compute_retrieval_confidence([0.9, 0.4, 0.1], component_agreement=0.67, evidence_coverage=0.8)
    assert 0.0 <= confidence.confidence <= 1.0
    assert confidence.score_range > 0.0
    assert 0.0 <= confidence.normalized_top_score <= 1.0
    assert confidence.confidence < 0.999999
    result = fit_proxy_threshold([0.2, 0.8, 0.9], [False, True, True], minimum_coverage=0.8)
    assert result["threshold"] <= 0.9
    assert result["selection_basis"] == "development_proxy_relevance_only"


def test_shortlist_without_positive_is_kept_for_evaluation_but_not_training() -> None:
    assert target_inside_shortlist([0.2, 0.4], positive_threshold=0.5) is False
    assert not pairwise_training_mask([0.2, 0.4], positive_threshold=0.5).any()
    positive, negative = build_pairwise_examples([[1.0], [2.0]], [0.2, 0.4])
    assert positive.shape == (0, 1)
    assert negative.shape == (0, 1)
    assert hard_negative_indices([0.9, 0.8, 0.7], [0.1, 0.8, 0.2], top_k=2) == [0, 2]


def test_candidate_union_and_rrf_are_deterministic_and_budgeted() -> None:
    rankings = [["b", "a", "c"], ["a", "c", "d"], ["d", "b", "e"]]
    assert top_k_union(rankings, 2) == ["b", "a", "c", "d"]
    assert reciprocal_rank_fusion_union(rankings, source_top_k=3, output_k=3) == ["a", "b", "d"]


def test_grouped_bootstrap_is_reproducible() -> None:
    first = grouped_bootstrap_ci({"a": 0.1, "b": 0.2, "c": 0.3}, repetitions=200, seed=7)
    second = grouped_bootstrap_ci({"c": 0.3, "a": 0.1, "b": 0.2}, repetitions=200, seed=7)
    assert first == second
    assert abs(float(first["estimate"]) - 0.2) < 1e-12


def test_planner_covers_author_defined_intents() -> None:
    assert plan_question("Has the opacity changed from the prior study?").intent == "comparison"
    assert plan_question("How severe is the opacity?").intent == "severity"
    assert plan_question("Is the central line properly positioned?").intent == "device"
    assert plan_question("Summarize the chest radiograph.").intent == "summary"
