from __future__ import annotations

import numpy as np

from medical_rag.dashboard.v6_runtime import (
    build_v6_query,
    extractive_v6_answer,
    fuse_v6_shortlist,
)


def test_v6_query_keeps_indication_and_question() -> None:
    assert build_v6_query("  Chest pain ", " What is the impression? ") == (
        "Clinical indication: Chest pain\nQuestion: What is the impression?"
    )


def test_v6_fusion_is_deterministic_and_tie_breaks_by_case_id() -> None:
    ranking = ["CXR2", "CXR1", "CXR3"]
    fused, fused_scores, text_scores, image_scores = fuse_v6_shortlist(
        ranking,
        [1.0, 1.0, 0.0],
        {"CXR1": 0.0, "CXR2": 1.0, "CXR3": 0.5},
        shortlist_size=3,
        text_weight=0.5,
    )
    assert fused == ["CXR2", "CXR1", "CXR3"]
    assert set(fused_scores) == set(ranking)
    assert text_scores["CXR2"] == text_scores["CXR1"] == 1.0
    assert image_scores["CXR1"] == 0.0
    assert np.isclose(fused_scores["CXR2"], 1.0)


def test_v6_fusion_preserves_tail_after_shortlist() -> None:
    fused, *_ = fuse_v6_shortlist(
        ["CXR1", "CXR2", "CXR3"],
        [1.0, 0.5, 0.0],
        {"CXR1": 0.0, "CXR2": 0.5, "CXR3": 1.0},
        shortlist_size=2,
        text_weight=0.5,
    )
    assert fused[-1] == "CXR3"


def test_v6_extractive_answer_routes_to_impression_for_summary_questions() -> None:
    case = {"findings": "The lungs are clear.", "impression": "No acute disease."}
    assert extractive_v6_answer("Summarize the conclusion.", case) == "No acute disease."
    assert extractive_v6_answer("What findings are documented?", case) == "The lungs are clear."
