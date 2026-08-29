from __future__ import annotations

import json

import pytest

from medical_rag.qa.radrestruct import (
    RadReStructQuestion,
    canonical_openi_case_id,
    iter_radrestruct_cases,
)


def test_canonical_openi_case_id() -> None:
    assert canonical_openi_case_id(1) == "CXR1"
    assert canonical_openi_case_id("001") == "CXR1"
    assert canonical_openi_case_id("CXR001") == "CXR1"
    with pytest.raises(ValueError):
        canonical_openi_case_id("report-one")


def test_radrestruct_question_validation() -> None:
    question = RadReStructQuestion.from_json(
        [
            "Is there pleural fluid?",
            ["yes"],
            [],
            {
                "answer_type": "single_choice",
                "options": ["yes", "no"],
                "path": "pleura_signs",
            },
        ]
    )
    assert question.answers == ("yes",)
    assert question.options == ("yes", "no")


def test_iter_radrestruct_cases(tmp_path) -> None:
    root = tmp_path / "radrestruct"
    root.mkdir()
    (root / "id_to_img_mapping_frontal_reports.json").write_text(
        json.dumps({"1": ["CXR1_1_IM-0001-4001"]}), encoding="utf-8"
    )
    for split, ids in (("train", [1]), ("val", []), ("test", [])):
        (root / f"{split}_ids.json").write_text(json.dumps(ids), encoding="utf-8")
        qa_dir = root / f"{split}_qa_pairs"
        qa_dir.mkdir()
    (root / "train_qa_pairs" / "1.json").write_text(
        json.dumps(
            [
                [
                    "Is there an abnormality?",
                    ["no"],
                    [],
                    {
                        "answer_type": "single_choice",
                        "options": ["yes", "no"],
                        "path": "lung_signs",
                    },
                ]
            ]
        ),
        encoding="utf-8",
    )

    cases = list(iter_radrestruct_cases(root))
    assert len(cases) == 1
    assert cases[0].case_id == "CXR1"
    assert cases[0].questions[0].answers == ("no",)
