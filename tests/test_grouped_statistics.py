from __future__ import annotations

import unittest


from scripts.run_grouped_statistical_analysis import (
    grouped_bootstrap_ci,
    holm_adjust,
    paired_grouped_bootstrap,
)


class GroupedStatisticsTests(unittest.TestCase):
    def test_grouped_bootstrap_preserves_observed_mean(self) -> None:
        observed, low, high = grouped_bootstrap_ci(
            {"a": [1.0, 1.0], "b": [0.0, 0.0]}, iterations=1000, seed=1
        )
        self.assertEqual(observed, 0.5)
        self.assertLessEqual(low, observed)
        self.assertGreaterEqual(high, observed)

    def test_paired_bootstrap_detects_consistent_positive_difference(self) -> None:
        result = paired_grouped_bootstrap(
            {"a": [1.0], "b": [1.0], "c": [1.0]},
            {"a": [0.0], "b": [0.0], "c": [0.0]},
            iterations=1000,
            seed=1,
        )
        self.assertEqual(result["mean_difference"], 1.0)
        self.assertEqual(result["two_sided_bootstrap_p"], 0.0)
        # With only three paired units, an exact two-sided sign-flip test cannot
        # attain a p-value below 0.25 even when every difference is positive.
        self.assertLess(result["paired_randomization_p"], 0.35)

    def test_holm_adjustment_preserves_order_and_monotonicity(self) -> None:
        adjusted = holm_adjust([0.04, 0.001, 0.02])
        self.assertEqual(adjusted, [0.04, 0.003, 0.04])


if __name__ == "__main__":
    unittest.main()
