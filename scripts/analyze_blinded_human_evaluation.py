from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from scripts.run_grouped_statistical_analysis import holm_adjust

DEFAULT_REFERENCE_SYSTEM = "final_adaptive_direct_semantic_agent"


def paired_bootstrap(
    first: np.ndarray, second: np.ndarray, *, iterations: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    differences = first - second
    sampled = np.empty(iterations, dtype="float64")
    for index in range(iterations):
        positions = rng.integers(0, len(differences), size=len(differences))
        sampled[index] = float(differences[positions].mean())
    randomization_rng = np.random.default_rng(seed + 1_000_000)
    signs = randomization_rng.choice([-1.0, 1.0], size=(iterations, len(differences)))
    randomized = (signs * differences).mean(axis=1)
    randomization_p = (float(np.sum(np.abs(randomized) >= abs(differences.mean()))) + 1) / (
        iterations + 1
    )
    return {
        "mean_difference": float(differences.mean()),
        "ci_low_95": float(np.quantile(sampled, 0.025)),
        "ci_high_95": float(np.quantile(sampled, 0.975)),
        "two_sided_bootstrap_p": float(
            min(1.0, 2 * min(np.mean(sampled <= 0), np.mean(sampled >= 0)))
        ),
        "paired_randomization_p": randomization_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the completed blinded review sheet.")
    parser.add_argument(
        "--ratings",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "human_evaluation"
        / "held_out_blinded_human_evaluation_36.csv",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "human_evaluation"
        / "held_out_blinded_human_evaluation_key.csv",
    )
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7023)
    parser.add_argument("--reference-system", default=DEFAULT_REFERENCE_SYSTEM)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "human_evaluation"
        / "held_out_human_evaluation_results.json",
    )
    args = parser.parse_args()

    ratings = pd.read_csv(args.ratings, dtype=str, keep_default_na=False)
    key = pd.read_csv(args.key, dtype=str)
    long_rows = []
    missing = []
    for _, row in ratings.iterrows():
        best = row["best_response_A_B_C_D_or_tie"].strip().upper()
        if best not in {"A", "B", "C", "D", "TIE"}:
            missing.append(f"{row['sample_id']}:best_response")
        for label in "ABCD":
            lower = label.lower()
            values = {
                "correctness": row[f"{lower}_correctness_1_5"].strip(),
                "grounding": row[f"{lower}_evidence_grounding_1_5"].strip(),
                "harmful": row[f"{lower}_potentially_harmful_0_1"].strip(),
            }
            for field, value in values.items():
                if not value:
                    missing.append(f"{row['sample_id']}:{label}:{field}")
            long_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "question_type": row["question_type"],
                    "response_label": label,
                    "correctness": values["correctness"],
                    "grounding": values["grounding"],
                    "harmful": values["harmful"],
                    "preference_win": float(best == label),
                    "tie": float(best == "TIE"),
                }
            )
    if missing:
        raise SystemExit(
            f"Human evaluation is incomplete: {len(missing)} required cells missing or invalid. "
            f"First entries: {', '.join(missing[:8])}"
        )

    long = pd.DataFrame(long_rows).merge(
        key[["sample_id", "response_label", "system"]],
        on=["sample_id", "response_label"],
        how="left",
        validate="one_to_one",
    )
    for field in ["correctness", "grounding", "harmful"]:
        long[field] = pd.to_numeric(long[field], errors="raise")
    if not long["correctness"].between(1, 5).all():
        raise ValueError("correctness scores must be between 1 and 5")
    if not long["grounding"].between(1, 5).all():
        raise ValueError("grounding scores must be between 1 and 5")
    if not long["harmful"].isin([0, 1]).all():
        raise ValueError("harmful scores must be 0 or 1")

    summary = (
        long.groupby("system", sort=True)
        .agg(
            n=("sample_id", "size"),
            mean_correctness=("correctness", "mean"),
            median_correctness=("correctness", "median"),
            mean_grounding=("grounding", "mean"),
            median_grounding=("grounding", "median"),
            harmful_rate=("harmful", "mean"),
            preference_win_rate=("preference_win", "mean"),
        )
        .reset_index()
    )
    if args.reference_system not in set(summary["system"]):
        raise ValueError(f"Reference system not found in blinded key: {args.reference_system}")
    comparisons = []
    indexed = long.set_index(["system", "sample_id"])
    systems = [value for value in summary["system"] if value != args.reference_system]
    for comparison_index, system in enumerate(systems):
        for metric_index, metric in enumerate(["correctness", "grounding", "harmful"]):
            final_values = indexed.loc[args.reference_system][metric].sort_index()
            baseline_values = indexed.loc[system][metric].sort_index()
            if not final_values.index.equals(baseline_values.index):
                raise ValueError("paired systems do not contain identical sample IDs")
            result = paired_bootstrap(
                final_values.to_numpy(dtype=float),
                baseline_values.to_numpy(dtype=float),
                iterations=args.iterations,
                seed=args.seed + comparison_index * 10 + metric_index,
            )
            comparisons.append(
                {
                    "system_a": args.reference_system,
                    "system_b": system,
                    "metric": metric,
                    **result,
                }
            )
    adjusted = holm_adjust(
        [float(value["paired_randomization_p"]) for value in comparisons]
    )
    for comparison, adjusted_p in zip(comparisons, adjusted, strict=True):
        comparison["holm_adjusted_randomization_p"] = adjusted_p

    by_question_type = (
        long.groupby(["system", "question_type"], sort=True)
        .agg(
            n=("sample_id", "size"),
            mean_correctness=("correctness", "mean"),
            mean_grounding=("grounding", "mean"),
            harmful_rate=("harmful", "mean"),
            preference_win_rate=("preference_win", "mean"),
        )
        .reset_index()
    )

    output = {
        "sample_count": int(ratings.shape[0]),
        "response_count": int(long.shape[0]),
        "summary": summary.to_dict(orient="records"),
        "summary_by_question_type": by_question_type.to_dict(orient="records"),
        "paired_bootstrap": comparisons,
        "overall_tie_rate": float(long.groupby("sample_id")["tie"].first().mean()),
        "iterations": args.iterations,
        "seed": args.seed,
        "reference_system": args.reference_system,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    summary.to_csv(args.output.with_suffix(".csv"), index=False)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
