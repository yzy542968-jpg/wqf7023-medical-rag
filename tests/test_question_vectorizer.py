from __future__ import annotations

import json

from medical_rag.qa.question_vectorizer import RadReStructQuestionVectorizer
from medical_rag.qa.radrestruct import RadReStructQuestion
from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy


def test_vectorizer_maps_ordered_rows_and_fills_suppressed_children(tmp_path) -> None:
    root = tmp_path / "rad"
    root.mkdir()
    (root / "report_keys.json").write_text(
        json.dumps(
            [
                "lung_signs_yes",
                "lung_signs_no",
                "lung_signs_opacity_yes_0",
                "lung_signs_opacity_no_0",
                "lung_signs_opacity_location_left_0",
                "lung_signs_opacity_location_unspecified_0",
            ]
        ),
        encoding="utf-8",
    )
    (root / "vectorized_question_ids.json").write_text(
        json.dumps([0, 0, 1, 1, 2, 2]), encoding="utf-8"
    )
    (root / "vectorized_choice_options.json").write_text(
        json.dumps(
            {"0": "single_choice", "1": "single_choice", "2": "multi_choice"}
        ),
        encoding="utf-8",
    )
    (root / "max_instances.json").write_text("{}", encoding="utf-8")
    (root / "template_final_clean.json").write_text(
        json.dumps(
            [
                {
                    "area": "lung",
                    "signs": {
                        "opacity": {"infos": {"location": {}}},
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    question = RadReStructQuestion(
        question="Are there signs?",
        answers=("no",),
        history=(),
        answer_type="single_choice",
        options=("yes", "no"),
        path="lung_signs",
    )
    vectorizer = RadReStructQuestionVectorizer(RadReStructHierarchy(root))
    assert vectorizer.question_ids((question,)) == (0,)
    assert vectorizer.vectorize_answers((question,)).tolist() == [0, 1, 0, 1, 0, 0]
