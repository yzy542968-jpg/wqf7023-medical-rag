from __future__ import annotations

import json
from pathlib import Path

import pytest

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
from medical_rag.similar_case.openi_adapter import openi_row_to_paired_case
from medical_rag.similar_case.prompt import build_evidence_constrained_prompt
from medical_rag.similar_case.radgraph_adapter import (
    match_radgraph_facts,
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
