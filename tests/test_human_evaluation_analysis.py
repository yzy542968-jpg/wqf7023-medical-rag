from __future__ import annotations

import numpy as np

from scripts.analyze_blinded_human_evaluation import paired_bootstrap


def test_paired_bootstrap_reports_positive_difference() -> None:
    result = paired_bootstrap(
        np.array([5.0, 4.0, 5.0, 4.0]),
        np.array([3.0, 3.0, 2.0, 3.0]),
        iterations=1000,
        seed=7023,
    )

    assert result["mean_difference"] == 1.75
    assert result["ci_low_95"] > 0
    assert result["two_sided_bootstrap_p"] == 0.0
    assert result["paired_randomization_p"] < 0.2
