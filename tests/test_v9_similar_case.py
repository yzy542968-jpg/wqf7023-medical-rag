from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.generate_v9_radgraph_annotations import (
    annotation_record,
    normalized_report_text,
)
from scripts.run_v9_development_medsiglip import (
    select_fusion_weights,
    select_report_policy,
)
from scripts.train_v9_learned_reranker import (
    exact_leave_one_out_bm25_scores,
    feature_matrix,
)

from medical_rag.evaluation.graded_retrieval import (
    evaluate_graded_retrieval,
    evaluate_grouped_graded_retrieval,
    ndcg_at_k,
)
from medical_rag.similar_case.bank import build_candidate_bank
from medical_rag.similar_case.chexpert_plus_adapter import (
    CHEXBERT_LABEL_COLUMNS,
    parse_chexpert_patient_study,
    read_chexpert_plus_cases,
)
from medical_rag.similar_case.openi_adapter import (
    openi_row_to_paired_case,
    read_openi_paired_cases,
)
from medical_rag.similar_case.prompt import build_evidence_constrained_prompt
from medical_rag.similar_case.radgraph_adapter import (
    match_radgraph_facts,
    read_radgraph_case_records,
    read_radgraph_facts_by_text,
)
from medical_rag.similar_case.relevance import (
    active_label_similarity,
    report_relevance_gain,
)
from medical_rag.similar_case.retrieval import fuse_component_scores
from medical_rag.similar_case.retrieval import cosine_score_map
from medical_rag.similar_case.schema import PairedCase
from medical_rag.similar_case.text_baseline import SimilarCaseBM25Retriever


ROOT = Path(__file__).resolve().parents[1]


def make_case(
    study_id: str,
    patient_id: str | None,
    *,
    labels: dict[str, object] | None = None,
    facts: set[str] | None = None,
    findings: str = "Target-hidden finding.",
    impression: str = "Target-hidden impression.",
) -> PairedCase:
    return PairedCase(
        study_id=study_id,
        patient_id=patient_id,
        image_paths=(f"{study_id}.png",),
        indication="Shortness of breath",
        findings=findings,
        impression=impression,
        labels=labels or {},
        radgraph_facts=frozenset(facts or set()),
        source="synthetic",
    )


def test_candidate_bank_excludes_target_study_and_same_patient() -> None:
    query = make_case("s1", "p1")
    source = [query, make_case("s2", "p1"), make_case("s3", "p2")]

    bank, audit = build_candidate_bank(query, source)

    assert [case.study_id for case in bank] == ["s3"]
    assert audit.excluded_same_study_count == 1
    assert audit.excluded_same_patient_count == 1
    assert audit.post_filter_same_patient_count == 0
    assert audit.patient_level_exclusion_verified is True


def test_candidate_bank_requires_patient_ids_for_formal_use() -> None:
    query = make_case("s1", None)
    with pytest.raises(ValueError, match="query patient ID"):
        build_candidate_bank(query, [make_case("s2", None)])


def test_negative_label_agreement_receives_no_credit() -> None:
    query = {"pneumonia": 1, "effusion": 0, "edema": 0}
    candidate = {"pneumonia": 0, "effusion": 0, "edema": 0}
    assert active_label_similarity(query, candidate) == 0.0


def test_report_relevance_combines_active_labels_and_facts() -> None:
    query = make_case(
        "s1",
        "p1",
        labels={"effusion": 1, "edema": 0},
        facts={"effusion located_at right"},
    )
    candidate = make_case(
        "s2",
        "p2",
        labels={"effusion": 1, "edema": 0},
        facts={"effusion located_at right"},
    )
    assert report_relevance_gain(query, candidate) == pytest.approx(1.0)


def test_graded_ndcg_rewards_better_ordering() -> None:
    qrels = {"a": 1.0, "b": 0.6, "c": 0.0}
    assert ndcg_at_k(qrels, ["a", "b", "c"], 3) == pytest.approx(1.0)
    assert ndcg_at_k(qrels, ["c", "b", "a"], 3) < 1.0
    aggregate = evaluate_graded_retrieval(
        {"q1": qrels}, {"q1": ["a", "b", "c"]}, k_values=(1, 3)
    )
    assert aggregate["ndcg@3"] == pytest.approx(1.0)
    assert aggregate["mrr"] == pytest.approx(1.0)


def test_patient_grouped_metric_prevents_multi_query_patient_overweighting() -> None:
    qrels = {
        "p1:q1": {"relevant": 1.0},
        "p1:q2": {"relevant": 1.0},
        "p2:q1": {"relevant": 1.0},
    }
    rankings = {
        "p1:q1": ["relevant"],
        "p1:q2": ["relevant"],
        "p2:q1": ["irrelevant"],
    }
    grouped = evaluate_grouped_graded_retrieval(
        qrels,
        rankings,
        {"p1:q1": "p1", "p1:q2": "p1", "p2:q1": "p2"},
        k_values=(1,),
    )
    ungrouped = evaluate_graded_retrieval(qrels, rankings, k_values=(1,))

    assert grouped["ndcg@1"] == pytest.approx(0.5)
    assert ungrouped["ndcg@1"] == pytest.approx(2.0 / 3.0)


def test_paired_score_fusion_normalizes_components_and_breaks_ties() -> None:
    rows = fuse_component_scores(
        {
            "bm25": {"b": 1.0, "a": 2.0},
            "image_image": {"b": 3.0, "a": 1.0},
            "image_report": {"b": 2.0, "a": 2.0},
        },
        {"bm25": 0.5, "image_image": 0.5, "image_report": 0.0},
    )
    assert [row.study_id for row in rows] == ["a", "b"]
    assert rows[0].score == pytest.approx(rows[1].score)


def test_cosine_score_map_supports_image_and_report_embedding_channels() -> None:
    scores = cosine_score_map(
        query_embedding=[1.0, 0.0],
        candidate_embeddings=[[1.0, 0.0], [0.0, 2.0]],
        candidate_ids=["same-direction", "orthogonal"],
    )
    assert scores == pytest.approx({"same-direction": 1.0, "orthogonal": 0.0})


def test_similar_case_bm25_uses_only_query_indication_and_question() -> None:
    bank = [
        make_case(
            "edema",
            "p2",
            findings="Pulmonary edema is present.",
            impression="Pulmonary edema.",
        ),
        make_case(
            "clear",
            "p3",
            findings="The lungs are clear.",
            impression="No acute disease.",
        ),
    ]
    query = make_case(
        "query",
        "p1",
        findings="This hidden report must not be searched.",
        impression="This hidden reference must not be searched.",
    )
    retriever = SimilarCaseBM25Retriever().fit(bank)
    rows = retriever.search(query, "Is pulmonary edema present?", top_k=2)

    assert rows[0].study_id == "edema"
    assert set(rows[0].component_scores) == {"bm25"}


def test_similar_case_bm25_rejects_same_patient_bank_leakage() -> None:
    query = make_case("query", "p1")
    retriever = SimilarCaseBM25Retriever().fit([make_case("historical", "p1")])
    with pytest.raises(ValueError, match="target-patient"):
        retriever.search(query, "What is the finding?")


def test_chexpert_plus_adapter_groups_views_at_study_level(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    relative_paths = [
        "train/patient00001/study2/view1_frontal.jpg",
        "train/patient00001/study2/view2_lateral.jpg",
    ]
    for relative_path in relative_paths:
        path = image_root.joinpath(*relative_path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    csv_path = tmp_path / "df_chexpert_plus_240401.csv"
    csv_path.write_text(
        "path_to_image,section_history,section_findings,section_impression\n"
        + "\n".join(
            f"{path},Dyspnea,Bilateral edema.,Pulmonary edema."
            for path in relative_paths
        )
        + "\n",
        encoding="utf-8",
    )
    labels_path = tmp_path / "findings_fixed.jsonl"
    label_rows = []
    for relative_path in relative_paths:
        row = {label: None for label in CHEXBERT_LABEL_COLUMNS}
        row.update({"path_to_image": relative_path, "Edema": 1.0})
        label_rows.append(json.dumps(row))
    labels_path.write_text("\n".join(label_rows) + "\n", encoding="utf-8")

    cases = read_chexpert_plus_cases(
        csv_path,
        image_root=image_root,
        chexbert_labels_path=labels_path,
        radgraph_facts_by_findings={
            "Bilateral edema.": ["edema located_at bilateral lung"]
        },
    )

    assert len(cases) == 1
    case = cases[0]
    assert case.study_id == "patient00001/study2"
    assert case.patient_id == "patient00001"
    assert len(case.image_paths) == 2
    assert case.labels["Edema"] == 1.0
    assert case.radgraph_facts == frozenset({"edema located_at bilateral lung"})
    assert case.metadata["view_count"] == 2


def test_chexpert_plus_adapter_blocks_qrels_until_radgraph_is_available(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text(
        "path_to_image,section_findings,section_impression\n"
        "train/patient1/study1/view1_frontal.jpg,Clear lungs.,No acute disease.\n"
        "train/patient2/study1/view1_frontal.jpg,Clear lungs.,No acute disease.\n",
        encoding="utf-8",
    )
    labels_path = tmp_path / "labels.jsonl"
    rows = []
    for patient in ("patient1", "patient2"):
        row = {label: None for label in CHEXBERT_LABEL_COLUMNS}
        row.update(
            {
                "path_to_image": f"train/{patient}/study1/view1_frontal.jpg",
                "No Finding": 1.0,
            }
        )
        rows.append(json.dumps(row))
    labels_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    cases = read_chexpert_plus_cases(
        csv_path,
        image_root=tmp_path,
        chexbert_labels_path=labels_path,
        require_image_files=False,
    )

    with pytest.raises(ValueError, match="unavailable RadGraph annotations"):
        report_relevance_gain(cases[0], cases[1])


def test_chexpert_path_parser_rejects_noncanonical_ids() -> None:
    assert parse_chexpert_patient_study(
        "train/patient42142/study5/view1_frontal.jpg"
    ) == ("patient42142", "patient42142/study5")
    with pytest.raises(ValueError, match="Cannot parse patient/study"):
        parse_chexpert_patient_study("train/unknown/view.jpg")
    with pytest.raises(ValueError, match="safe relative path"):
        parse_chexpert_patient_study("../patient1/study1/view.jpg")


def test_official_radgraph_xl_annotations_are_flattened_deterministically(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "section_findings.json"
    annotation_path.write_text(
        json.dumps(
            [
                {
                    "0": {
                        "text": "Bilateral edema .",
                        "entities": {
                            "1": {
                                "tokens": "edema",
                                "label": "OBS-DP",
                                "relations": [["located_at", "2"]],
                            },
                            "2": {
                                "tokens": "lungs",
                                "label": "ANAT-DP",
                                "relations": [],
                            },
                        },
                    }
                }
            ]
        ),
        encoding="utf-8",
    )

    facts_by_text = read_radgraph_facts_by_text(annotation_path)
    facts, available = match_radgraph_facts("Bilateral edema.", facts_by_text)

    assert available is True
    assert facts == frozenset(
        {
            "relation|edema|obs-dp|located_at|lungs",
            "entity|lungs|anat-dp",
        }
    )


def test_prompt_separates_target_observation_from_historical_analogy() -> None:
    historical = make_case(
        "h1",
        "p2",
        findings="Small right pleural effusion.",
        impression="Right pleural effusion.",
    )
    prompt = build_evidence_constrained_prompt(
        indication="Dyspnea",
        question="What is the main finding?",
        retrieved_cases=[historical],
    )
    assert "analogies only" in prompt
    assert "not proof" in prompt
    assert "h1" in prompt
    assert "Target-hidden" not in prompt


def test_openi_adapter_is_explicitly_not_patient_level() -> None:
    case = openi_row_to_paired_case(
        {
            "case_id": "CXR2",
            "indication": "Preoperative study",
            "findings": "Borderline cardiomegaly.",
            "impression": "No acute pulmonary finding.",
            "problems": "Cardiomegaly;Pulmonary Artery",
            "images": [{"filename": "2.png", "projection": "Frontal"}],
        }
    )
    assert case.patient_id is None
    assert case.source == "openi_engineering_smoke_only"
    assert set(case.labels) == {"cardiomegaly", "pulmonary artery"}
    assert case.metadata["report_index_class"] == "abnormal"
    assert case.metadata["label_annotation_available"] is True


def test_openi_formal_mode_uses_source_design_patient_keys() -> None:
    rows = [
        openi_row_to_paired_case(
            {
                "case_id": case_id,
                "findings": "Clear lungs.",
                "impression": "No acute disease.",
                "problems": "normal",
                "images": [{"filename": f"{case_id}.png"}],
            },
            source_unique_patient=True,
        )
        for case_id in ("CXR1", "CXR2")
    ]

    assert rows[0].patient_id == "openi-source-unique:CXR1"
    assert rows[0].source == "openi_iu_xray_primary_source"
    assert rows[0].metadata["released_patient_identifier_available"] is False
    assert rows[0].metadata["source_collection_one_study_per_patient"] is True
    bank, audit = build_candidate_bank(rows[0], rows, require_patient_ids=True)
    assert [case.study_id for case in bank] == ["CXR2"]
    assert audit.patient_level_exclusion_verified is True


def test_openi_normal_and_unindexed_reports_are_not_conflated() -> None:
    common = {
        "indication": "Screening",
        "findings": "No focal opacity.",
        "impression": "No acute disease.",
        "images": [{"filename": "image.png"}],
    }
    normal = openi_row_to_paired_case(
        {**common, "case_id": "normal", "problems": "normal"}
    )
    unindexed = openi_row_to_paired_case(
        {**common, "case_id": "unindexed", "problems": "no indexing"}
    )

    assert normal.metadata["report_index_class"] == "normal"
    assert normal.metadata["label_annotation_available"] is True
    assert unindexed.metadata["report_index_class"] == "indeterminate"
    assert unindexed.metadata["label_annotation_available"] is False
    with pytest.raises(ValueError, match="unavailable label annotations"):
        report_relevance_gain(normal, unindexed)


def test_v9_protocol_keeps_confirmation_uninstantiated() -> None:
    config = json.loads(
        (ROOT / "config/v9_similar_case_rag_development.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["confirmation_ids_instantiated"] is False
    assert config["hidden_reference_fields"] == [
        "findings",
        "impression",
        "report_labels",
        "radgraph_entities_relations",
    ]
    assert config["candidate_exclusions"] == ["same_study", "same_patient"]


def test_v9_full_source_split_matches_frozen_protocol() -> None:
    protocol = json.loads(
        (ROOT / "config/v9_full_source_split_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (ROOT / "data/splits/v9/v9_full_source_split.json").read_text(
            encoding="utf-8"
        )
    )
    partitions = {
        name: set(block["case_ids"])
        for name, block in manifest["partitions"].items()
    }

    for name, case_ids in partitions.items():
        block = manifest["partitions"][name]
        expected = protocol["partitions"][name]
        assert len(case_ids) == expected["total"] == block["case_count"]
        assert block["report_indexed_normal"] == expected["normal"]
        assert block["report_indexed_abnormal"] == expected["abnormal"]

    assert not (partitions["train"] & partitions["validation"])
    assert not (partitions["train"] & partitions["test"])
    assert not (partitions["validation"] & partitions["test"])
    assert len(set.union(*partitions.values())) == 3759

    strict = set(
        manifest["strict_project_history_untouched_test_subset"]["case_ids"]
    )
    assert len(strict) == 262
    assert strict <= partitions["test"]
    assert sum(
        block["complete_findings_and_impression_reference"]
        for block in manifest["partitions"].values()
    ) == protocol["qa_complete_reference_source_count"] == 3244


def test_v9_radgraph_preprocessing_preserves_section_boundary() -> None:
    text = normalized_report_text(
        {
            "findings": "  Mild   pulmonary edema. ",
            "impression": " Pulmonary edema.  ",
        }
    )
    assert text == "Mild pulmonary edema.\nPulmonary edema."


def test_v9_empty_report_is_not_treated_as_empty_radgraph_facts() -> None:
    record = annotation_record(
        case_id="CXR-empty",
        report_text="",
        model_type="modern-radgraph-xl",
        annotation=None,
    )
    assert record["status"] == "empty_report"
    assert record["facts"] == []
    assert record["annotation"] is None


def test_openi_formal_radgraph_records_replace_problem_proxy(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "CXR1",
                "indication": "Dyspnea",
                "findings": "Mild edema.",
                "impression": "Pulmonary edema.",
                "problems": "Edema",
                "images": [{"filename": "1.png"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    radgraph_path = tmp_path / "radgraph.jsonl"
    radgraph_path.write_text(
        json.dumps(
            {
                "case_id": "CXR1",
                "status": "ok",
                "facts": ["entity|edema|observation::definitely present"],
                "report_text_sha256": "abc123",
                "model_type": "modern-radgraph-xl",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = read_radgraph_case_records(radgraph_path)
    assert records["CXR1"].status == "ok"
    cases = read_openi_paired_cases(
        cases_path,
        source_unique_patient=True,
        radgraph_path=radgraph_path,
    )
    assert cases[0].radgraph_facts == frozenset(
        {"entity|edema|observation::definitely present"}
    )
    assert cases[0].metadata["radgraph_annotation_available"] is True
    assert cases[0].metadata["radgraph_annotation_source"] == "modern-radgraph-xl"


def test_openi_empty_radgraph_record_is_unavailable(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "CXR2",
                "problems": "normal",
                "images": [{"filename": "2.png"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    radgraph_path = tmp_path / "radgraph.jsonl"
    radgraph_path.write_text(
        json.dumps(
            {
                "case_id": "CXR2",
                "status": "empty_report",
                "facts": [],
                "report_text_sha256": "empty-hash",
                "model_type": "modern-radgraph-xl",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    case = read_openi_paired_cases(cases_path, radgraph_path=radgraph_path)[0]
    assert case.radgraph_facts == frozenset()
    assert case.metadata["radgraph_annotation_available"] is False


def test_v9_report_policy_requires_material_maximum_gain() -> None:
    assert select_report_policy(0.20, 0.204, 0.005) == "normalized_mean_chunk_embedding"
    assert select_report_policy(0.20, 0.205, 0.005) == "maximum_image_chunk_cosine"


def test_v9_fusion_tie_rule_prefers_conservative_text_weight() -> None:
    sweep = [
        {
            "weights": {"bm25": 0.5, "image_image": 0.25, "image_report": 0.25},
            "metrics": {"ndcg@10": 0.201},
        },
        {
            "weights": {"bm25": 0.75, "image_image": 0.25, "image_report": 0.0},
            "metrics": {"ndcg@10": 0.20},
        },
        {
            "weights": {"bm25": 1.0, "image_image": 0.0, "image_report": 0.0},
            "metrics": {"ndcg@10": 0.25},
        },
    ]
    selected = select_fusion_weights(sweep, tolerance=0.005)
    assert selected["weights"]["bm25"] == 0.75


def test_v9_leave_one_out_bm25_masks_target_and_changes_corpus_statistics() -> None:
    from medical_rag.retrieval.bm25_retriever import BM25Retriever

    retriever = BM25Retriever().fit(
        [
            {"case_id": "a", "report_text": "edema edema"},
            {"case_id": "b", "report_text": "edema"},
            {"case_id": "c", "report_text": "clear lungs"},
        ]
    )
    scores = exact_leave_one_out_bm25_scores(
        retriever, "edema", excluded_index=0
    )
    assert scores[0] == float("-inf")
    assert scores[1] > scores[2]


def test_v9_reranker_feature_matrix_has_frozen_nine_features() -> None:
    features = feature_matrix(
        bm25=np.asarray([2.0, 1.0]),
        image_image=np.asarray([0.1, 0.2]),
        image_report=np.asarray([0.3, 0.4]),
        question_type="findings",
        excluded_index=None,
    )
    assert features.shape == (2, 9)
    assert features[0, 6:].tolist() == [1.0, 0.0, 0.0]
