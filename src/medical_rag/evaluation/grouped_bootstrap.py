"""Case-grouped bootstrap utilities for development audits."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def grouped_bootstrap_ci(
    values_by_group: Mapping[str, float],
    *,
    repetitions: int = 10_000,
    seed: int = 2026,
) -> dict[str, float | int]:
    """Return a percentile CI for the mean of one value per case/group."""

    if not values_by_group:
        raise ValueError("values_by_group cannot be empty")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    values = np.asarray([float(value) for _, value in sorted(values_by_group.items())], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci_95_low": float(np.quantile(means, 0.025)),
        "ci_95_high": float(np.quantile(means, 0.975)),
        "probability_greater_than_zero": float(np.mean(means > 0.0)),
        "group_count": int(len(values)),
        "repetitions": int(repetitions),
        "seed": int(seed),
    }


__all__ = ["grouped_bootstrap_ci"]
