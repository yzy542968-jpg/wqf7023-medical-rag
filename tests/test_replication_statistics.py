from __future__ import annotations

from scripts.finalize_locked_replication import grouped_bootstrap, wilson_interval


def test_wilson_interval_contains_observed_proportion() -> None:
    lower, upper = wilson_interval(30, 100)
    assert lower < 0.30 < upper


def test_grouped_bootstrap_uses_case_as_sampling_unit() -> None:
    rows = [
        {
            "case_id": case_id,
            "draft_token_f1": value,
            "final_token_f1": value,
            "support_rate": value,
        }
        for case_id, value in (("c1", 0.0), ("c2", 1.0))
    ]
    result = grouped_bootstrap(rows, iterations=50, seed=3)
    assert result["case_count"] == 2
    assert result["grouping_unit"] == "case_id"
