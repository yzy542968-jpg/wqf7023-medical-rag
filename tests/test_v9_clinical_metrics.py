from scripts.evaluate_v9_clinical_metrics import (
    grouped_bootstrap_ci,
    paired_grouped_bootstrap,
)


def test_grouped_bootstrap_uses_case_means() -> None:
    result = grouped_bootstrap_ci(
        {"a": [0.0, 1.0], "b": [1.0, 1.0]}, iterations=200, seed=7
    )
    assert result["mean"] == 0.75
    assert result["ci_95_low"] <= result["mean"] <= result["ci_95_high"]


def test_paired_grouped_bootstrap_preserves_pairs() -> None:
    result = paired_grouped_bootstrap(
        {"a": [0.8], "b": [0.5]},
        {"a": [0.3], "b": [0.4]},
        iterations=200,
        seed=8,
    )
    assert abs(result["difference"] - 0.3) < 1e-12
    assert result["case_count"] == 2
