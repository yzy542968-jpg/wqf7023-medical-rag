from __future__ import annotations

import json

import numpy as np

from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy
from medical_rag.qa.structured_decoding import (
    decode_answer_probabilities,
    knn_answer_probabilities,
)


def _hierarchy(tmp_path) -> RadReStructHierarchy:
    root = tmp_path / "rad"
    root.mkdir()
    (root / "report_keys.json").write_text(
        json.dumps(
            [
                "lung_signs_yes",
                "lung_signs_no",
                "lung_signs_opacity_yes",
                "lung_signs_opacity_no",
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
    return RadReStructHierarchy(root)


def test_decoder_respects_choice_types_and_hierarchy(tmp_path) -> None:
    hierarchy = _hierarchy(tmp_path)
    probabilities = np.asarray(
        [
            [0.2, 0.8, 0.9, 0.1, 0.8, 0.7],
            [0.8, 0.2, 0.7, 0.3, 0.6, 0.8],
        ],
        dtype=np.float32,
    )
    decoded = decode_answer_probabilities(
        probabilities,
        hierarchy,
        multi_choice_threshold=0.5,
        fixed_choice_threshold=0.5,
    )
    assert decoded.tolist() == [
        [0, 1, 0, 1, 0, 0],
        [1, 0, 1, 0, 1, 0],
    ]


def test_knn_probabilities_support_uniform_and_softmax() -> None:
    similarities = np.asarray([[0.9, 0.8, 0.1]], dtype=np.float32)
    targets = np.asarray([[1, 0], [0, 1], [0, 0]], dtype=np.float32)
    uniform = knn_answer_probabilities(
        similarities,
        targets,
        top_k=2,
        weighting="uniform",
        softmax_temperature=0.05,
    )
    assert np.allclose(uniform, [[0.5, 0.5]])
    weighted = knn_answer_probabilities(
        similarities,
        targets,
        top_k=2,
        weighting="cosine_softmax",
        softmax_temperature=0.05,
    )
    assert weighted[0, 0] > weighted[0, 1]


def test_decoder_clips_only_roundoff_sized_probability_error(tmp_path) -> None:
    hierarchy = _hierarchy(tmp_path)
    probabilities = np.asarray(
        [[1.0000001, -0.0000001, 0.9, 0.1, 0.4, 0.6]], dtype=np.float32
    )
    decoded = decode_answer_probabilities(
        probabilities,
        hierarchy,
        multi_choice_threshold=0.5,
        fixed_choice_threshold=0.5,
    )
    assert decoded.tolist() == [[1, 0, 1, 0, 0, 1]]
