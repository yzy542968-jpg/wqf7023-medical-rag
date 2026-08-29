from __future__ import annotations

import json

import numpy as np

from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy


def test_hierarchy_requires_expected_shape(tmp_path) -> None:
    root = tmp_path / "rad"
    root.mkdir()
    (root / "report_keys.json").write_text(
        json.dumps(["lung_signs_yes", "lung_signs_no"]), encoding="utf-8"
    )
    (root / "vectorized_question_ids.json").write_text(
        json.dumps([0, 0]), encoding="utf-8"
    )
    (root / "vectorized_choice_options.json").write_text(
        json.dumps({"0": "single_choice"}), encoding="utf-8"
    )
    (root / "max_instances.json").write_text("{}", encoding="utf-8")
    (root / "template_final_clean.json").write_text(
        json.dumps([{"area": "lung", "signs": {}}]), encoding="utf-8"
    )
    hierarchy = RadReStructHierarchy(root)
    prediction = np.asarray([[1, 0], [0, 1]], dtype=np.uint8)
    assert np.array_equal(hierarchy.clean(prediction), prediction)


def test_hierarchy_rejects_multiple_single_choice_answers(tmp_path) -> None:
    root = tmp_path / "rad"
    root.mkdir()
    (root / "report_keys.json").write_text(
        json.dumps(["lung_signs_yes", "lung_signs_no"]), encoding="utf-8"
    )
    (root / "vectorized_question_ids.json").write_text(
        json.dumps([0, 0]), encoding="utf-8"
    )
    (root / "vectorized_choice_options.json").write_text(
        json.dumps({"0": "single_choice"}), encoding="utf-8"
    )
    (root / "max_instances.json").write_text("{}", encoding="utf-8")
    (root / "template_final_clean.json").write_text(
        json.dumps([{"area": "lung", "signs": {}}]), encoding="utf-8"
    )
    hierarchy = RadReStructHierarchy(root)
    with np.testing.assert_raises(ValueError):
        hierarchy.clean(np.asarray([[1, 1]], dtype=np.uint8))


def test_hierarchy_clears_descendants_after_area_no(tmp_path) -> None:
    root = tmp_path / "rad"
    root.mkdir()
    keys = [
        "lung_signs_yes",
        "lung_signs_no",
        "lung_signs_opacity_yes",
        "lung_signs_opacity_no",
        "lung_signs_opacity_location_left",
        "lung_signs_opacity_location_right",
    ]
    (root / "report_keys.json").write_text(json.dumps(keys), encoding="utf-8")
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
    hierarchy = RadReStructHierarchy(root)
    prediction = np.asarray([[0, 1, 1, 0, 1, 0]], dtype=np.uint8)
    cleaned = hierarchy.clean(prediction)
    assert cleaned.tolist() == [[0, 1, 0, 1, 0, 0]]
    assert prediction.tolist() == [[0, 1, 1, 0, 1, 0]]
