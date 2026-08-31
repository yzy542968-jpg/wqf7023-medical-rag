from __future__ import annotations

import numpy as np

from medical_rag.similar_case.v17_question_conditioned import (
    answer_stratum,
    deterministic_top_ids,
    fixed_point_free_permutation,
    minmax,
    set_f1,
    summarize_proxy_rows,
    weighted_ranking,
)


def test_minmax_constant_feature_is_zero() -> None:
    assert np.array_equal(minmax([2.0, 2.0]), np.zeros(2))


def test_weighted_ranking_is_stable_on_ties() -> None:
    ranked, _ = weighted_ranking(
        ["B", "A"], np.asarray([[0.0, 1.0], [1.0, 0.0]]), [0.5, 0.5]
    )
    assert ranked == ["A", "B"]


def test_answer_strata_and_set_f1() -> None:
    assert answer_stratum(["Yes"]) == "positive"
    assert answer_stratum(["no"]) == "negative"
    assert answer_stratum(["left", "upper"]) == "non_binary"
    assert set_f1(["left", "upper"], ["left"]) == 2.0 / 3.0


def test_deterministic_controls_are_reproducible_and_fixed_point_free() -> None:
    candidates = ["c", "a", "b", "d"]
    assert deterministic_top_ids(candidates, domain="x", seed=7, key="q", count=3) == deterministic_top_ids(
        list(reversed(candidates)), domain="x", seed=7, key="q", count=3
    )
    mapping = fixed_point_free_permutation(["q1", "q2", "q3"], domain="m", seed=2)
    assert set(mapping) == {"q1", "q2", "q3"}
    assert set(mapping.values()) == set(mapping)
    assert all(left != right for left, right in mapping.items())


def test_proxy_summary_balances_strata() -> None:
    rows = [
        {"stratum": "positive", "top1_exact": 1, "top3_any_exact": 1, "top1_option_f1": 1, "top1_covered": 1, "top3_any_covered": 1, "top1_qrel_v2": 0.4},
        {"stratum": "negative", "top1_exact": 0, "top3_any_exact": 1, "top1_option_f1": 0, "top1_covered": 1, "top3_any_covered": 1, "top1_qrel_v2": 0.2},
        {"stratum": "non_binary", "top1_exact": 0, "top3_any_exact": 0, "top1_option_f1": 0, "top1_covered": 0, "top3_any_covered": 0, "top1_qrel_v2": 0.0},
    ]
    summary = summarize_proxy_rows(rows)
    assert summary["balanced_top1_qid_answer_agreement"] == 1.0 / 3.0
    assert summary["top3_any_exact"] == 2.0 / 3.0

