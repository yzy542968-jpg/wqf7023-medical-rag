from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from medical_rag.multimodal.fusion import (
    aggregate_view_embeddings,
    reciprocal_rank_fusion,
    select_text_weight,
)
from medical_rag.multimodal.evaluation import (
    aggregate_case_images,
    build_report_embedding_text,
    build_text_query,
    evaluate_rankings_and_answers,
)
from medical_rag.multimodal.openi_images import official_filename_candidates, resolve_official_image


ROOT = Path(__file__).resolve().parents[1]


def test_aggregate_view_embeddings_normalizes_mean() -> None:
    result = aggregate_view_embeddings([np.array([3.0, 0.0]), np.array([0.0, 4.0])])
    assert np.linalg.norm(result) == pytest.approx(1.0)
    assert result.tolist() == pytest.approx([2**-0.5, 2**-0.5])


def test_rrf_endpoints_preserve_single_modality_rankings() -> None:
    text = ["CXR1", "CXR2", "CXR3"]
    image = ["CXR3", "CXR2", "CXR1"]
    assert reciprocal_rank_fusion(text, image, text_weight=1.0) == text
    assert reciprocal_rank_fusion(text, image, text_weight=0.0) == image


def test_rrf_rejects_mismatched_candidate_pools() -> None:
    with pytest.raises(ValueError, match="same case IDs"):
        reciprocal_rank_fusion(["CXR1"], ["CXR2"], text_weight=0.5)


def test_weight_selection_uses_registered_tie_break() -> None:
    text = {"q1": ["A", "B"], "q2": ["A", "B"]}
    image = {"q1": ["B", "A"], "q2": ["B", "A"]}
    relevant = {"q1": "A", "q2": "B"}
    selected = select_text_weight(text, image, relevant, [0.0, 0.4, 0.6, 1.0])
    assert selected["selected_text_weight"] == 0.4


def test_preregistered_cohorts_match_frozen_benchmarks() -> None:
    config = json.loads((ROOT / "config" / "multimodal_v4.json").read_text(encoding="utf-8"))
    for split in ("development", "confirmation"):
        registered = config["cohorts"][split]
        benchmark = json.loads((ROOT / registered["benchmark_path"]).read_text(encoding="utf-8"))
        assert benchmark["case_count"] == registered["case_count"]
        assert benchmark["question_count"] == registered["question_count"]
        assert benchmark["case_id_fingerprint_sha256"] == registered["case_id_fingerprint_sha256"]
        assert benchmark["content_fingerprint_sha256"] == registered["content_fingerprint_sha256"]


def test_multimodal_query_and_report_text_keep_modal_roles_separate() -> None:
    case = {"indication": "Cough", "findings": "Clear lungs.", "impression": "No acute disease."}
    question = {"question": "What are the findings?"}
    assert build_text_query(case, question) == "Clinical indication: Cough\nQuestion: What are the findings?"
    assert build_report_embedding_text(case) == "Findings: Clear lungs.\nImpression: No acute disease."


def test_case_image_aggregation_fails_when_a_case_has_no_view() -> None:
    with pytest.raises(ValueError, match="without image embeddings"):
        aggregate_case_images(np.array([[1.0, 0.0]]), ["CXR1"], ["CXR1", "CXR2"])


def test_retrieval_evaluation_scores_downstream_selected_report() -> None:
    questions = [
        {
            "qid": "q1",
            "case_id": "CXR1",
            "answer_source": "impression",
            "reference_answer": "No acute disease.",
        }
    ]
    cases = {
        "CXR1": {"impression": "No acute disease."},
        "CXR2": {"impression": "Pleural effusion."},
    }
    correct, _ = evaluate_rankings_and_answers(questions, {"q1": ["CXR1", "CXR2"]}, cases)
    wrong, _ = evaluate_rankings_and_answers(questions, {"q1": ["CXR2", "CXR1"]}, cases)
    assert correct["hit@1"] == 1.0
    assert correct["token_f1"] == 1.0
    assert wrong["hit@1"] == 0.0
    assert wrong["token_f1"] < 1.0


def test_official_openi_filename_mapping_handles_general_and_cxr1_names() -> None:
    general = official_filename_candidates("CXR1000", "1000_IM-0003-1001.dcm.png")
    special = official_filename_candidates("CXR1", "1_IM-0001-4001.dcm.png")
    assert "CXR1000_IM-0003-1001.png" in general
    assert "CXR1_1_IM-0001-4001.png" in special

    lookup = {"CXR1_1_IM-0001-4001.png": Path("official.png")}
    assert resolve_official_image("CXR1", "1_IM-0001-4001.dcm.png", lookup) == Path("official.png")
