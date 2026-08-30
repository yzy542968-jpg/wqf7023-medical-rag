"""Evaluate the frozen Final-QA confirmation without changing its policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_final_qa_full_role import (  # noqa: E402
    B3,
    B4,
    B6,
    aggregate_binary_metrics,
    cases_for_role,
    keyed_full,
    official_compatible_metrics,
    row_summary,
)
from evaluate_final_qa_history_policy import negative_transfer  # noqa: E402
from evaluate_final_qa_qlora_pilot import metrics, read_jsonl  # noqa: E402
from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy  # noqa: E402
from medical_rag.qa.question_vectorizer import (  # noqa: E402
    RadReStructQuestionVectorizer,
)
from medical_rag.qa.structured_metrics import (  # noqa: E402
    bootstrap_supported_macro_f1_difference,
    structured_qa_metrics,
)


CONDITIONS = (B3, B4, B6)
FINAL = "final_question_conditional_gate"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _question_id_by_key(
    cases: list[Any], vectorizer: RadReStructQuestionVectorizer
) -> dict[tuple[str, int], int]:
    result: dict[tuple[str, int], int] = {}
    for case in cases:
        question_ids = vectorizer.question_ids(case.questions)
        for question_index, question_id in enumerate(question_ids):
            result[(case.case_id, question_index)] = int(question_id)
    return result


def apply_frozen_policy(
    rows_by_condition: dict[str, dict[tuple[str, int], dict[str, Any]]],
    question_ids: dict[tuple[str, int], int],
    policy: dict[str, Any],
) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, int]]:
    source_by_question = {
        int(record["question_id"]): str(record["source"])
        for record in policy["question_policy"]
    }
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    source_counts = {B3: 0, B6: 0}
    for key in sorted(rows_by_condition[B3]):
        source = source_by_question.get(question_ids[key], B3)
        if source not in source_counts:
            raise RuntimeError(f"Unsupported frozen gate source: {source}")
        selected[key] = rows_by_condition[source][key]
        source_counts[source] += 1
    return selected, source_counts


def build_matrices(
    *,
    cases: list[Any],
    keyed_systems: dict[str, dict[tuple[str, int], dict[str, Any]]],
    vectorizer: RadReStructQuestionVectorizer,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    target_rows: list[np.ndarray] = []
    prediction_rows: dict[str, list[np.ndarray]] = {
        system: [] for system in keyed_systems
    }
    for case in cases:
        target_rows.append(vectorizer.vectorize_answers(case.questions))
        for system, keyed_rows in keyed_systems.items():
            answers: list[list[str]] = []
            for question_index, question in enumerate(case.questions):
                row = keyed_rows[(case.case_id, question_index)]
                indices = [int(value) for value in row["predicted_indices"]]
                answers.append(
                    [
                        question.options[index]
                        for index in indices
                        if 0 <= index < len(question.options)
                    ]
                )
            prediction_rows[system].append(
                vectorizer.vectorize_answers(case.questions, answers)
            )
    return np.stack(target_rows), {
        system: np.stack(values) for system, values in prediction_rows.items()
    }


def case_grouped_exact_bootstrap(
    left: dict[tuple[str, int], dict[str, Any]],
    right: dict[tuple[str, int], dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    if set(left) != set(right):
        raise RuntimeError("Paired exact-bootstrap keys differ")
    differences: dict[str, list[float]] = defaultdict(list)
    for key in sorted(left):
        left_exact = set(left[key]["predicted_indices"]) == set(
            left[key]["gold_indices"]
        )
        right_exact = set(right[key]["predicted_indices"]) == set(
            right[key]["gold_indices"]
        )
        differences[key[0]].append(float(left_exact) - float(right_exact))
    case_values = np.asarray(
        [np.mean(differences[case_id]) for case_id in sorted(differences)],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=float)
    for offset in range(0, samples, 1000):
        current = min(1000, samples - offset)
        indices = rng.integers(
            0, len(case_values), size=(current, len(case_values))
        )
        estimates[offset : offset + current] = case_values[indices].mean(axis=1)
    return {
        "samples": samples,
        "seed": seed,
        "case_count": int(len(case_values)),
        "observed_difference": float(case_values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "probability_difference_greater_than_zero": float((estimates > 0).mean()),
    }


def _system_summary(
    rows: dict[tuple[str, int], dict[str, Any]],
    targets: np.ndarray,
    predictions: np.ndarray,
    report_keys: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "question_level": row_summary(list(rows.values())),
        "structured": structured_qa_metrics(targets, predictions).as_dict(),
        "aggregate_binary": aggregate_binary_metrics(targets, predictions),
        "official_compatible": official_compatible_metrics(
            targets, predictions, report_keys
        ),
    }


def _row_hash(rows: Iterable[dict[str, Any]]) -> str:
    canonical = "\n".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"))
        for row in sorted(
            rows,
            key=lambda row: (
                str(row["condition"]),
                str(row["case_id"]),
                int(row["question_index"]),
            ),
        )
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    generation_summary = json.loads(
        args.generation_summary.read_text(encoding="utf-8")
    )
    rows = read_jsonl(args.rows)
    expected = int(config["expected_question_count"])
    rows_by_condition = {
        condition: keyed_full(rows, condition) for condition in CONDITIONS
    }
    counts = {key: len(value) for key, value in rows_by_condition.items()}
    if any(count != expected for count in counts.values()):
        raise RuntimeError(f"Confirmation rows are incomplete: {counts}")
    if len({frozenset(rows) for rows in rows_by_condition.values()}) != 1:
        raise RuntimeError("Confirmation conditions do not use identical questions")
    if len(rows) != expected * len(CONDITIONS):
        raise RuntimeError("Confirmation rows contain unexpected conditions or duplicates")

    hierarchy = RadReStructHierarchy(args.radrestruct_root)
    vectorizer = RadReStructQuestionVectorizer(hierarchy)
    cases = cases_for_role(args.radrestruct_root, manifest, "test")
    if len(cases) != int(config["expected_case_count"]):
        raise RuntimeError("Confirmation case count differs from protocol")
    question_ids = _question_id_by_key(cases, vectorizer)
    selected, source_counts = apply_frozen_policy(
        rows_by_condition, question_ids, policy
    )
    keyed_systems = {**rows_by_condition, FINAL: selected}
    targets, predictions = build_matrices(
        cases=cases, keyed_systems=keyed_systems, vectorizer=vectorizer
    )
    systems = {
        system: _system_summary(
            keyed_rows, targets, predictions[system], hierarchy.report_keys
        )
        for system, keyed_rows in keyed_systems.items()
    }
    bootstrap_samples = int(config["statistics"]["bootstrap_replicates"])
    bootstrap_seed = int(config["statistics"]["seed"])
    exact_vs_b3 = case_grouped_exact_bootstrap(
        selected,
        rows_by_condition[B3],
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    macro_vs_b3 = bootstrap_supported_macro_f1_difference(
        targets,
        predictions[FINAL],
        predictions[B3],
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    macro_vs_b4 = bootstrap_supported_macro_f1_difference(
        targets,
        predictions[FINAL],
        predictions[B4],
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    noninferiority_margin = float(
        config["primary_hypotheses"]["h2_macro_noninferiority"]["margin"]
    )
    h1_passed = float(exact_vs_b3["ci95_low"]) > 0.0
    h2_passed = float(macro_vs_b3["ci95_low"]) >= noninferiority_margin
    result = {
        "study": config["study"],
        "status": "final_qa_confirmation_complete_frozen_policy",
        "test_accessed": True,
        "test_accessed_only_after_protocol_and_manifest_freeze": True,
        "case_count": len(cases),
        "question_count": expected,
        "label_count": int(targets.shape[1]),
        "systems": systems,
        "final_policy_source_counts": source_counts,
        "primary_hypotheses": {
            "h1_question_exact_superiority": {
                "criterion": "paired case-grouped 95% CI lower bound > 0",
                "bootstrap": exact_vs_b3,
                "passed": h1_passed,
            },
            "h2_macro_noninferiority": {
                "margin": noninferiority_margin,
                "criterion": "paired case-grouped 95% CI lower bound >= margin",
                "bootstrap": macro_vs_b3,
                "passed": h2_passed,
            },
            "combined_positive_claim_passed": h1_passed and h2_passed,
        },
        "secondary": {
            "final_macro_vs_random_b4": macro_vs_b4,
            "b6_negative_transfer_from_b3": negative_transfer(
                rows_by_condition[B3], rows_by_condition[B6]
            ),
            "final_negative_transfer_from_b3": negative_transfer(
                rows_by_condition[B3], selected
            ),
        },
        "runtime": {
            "generation_summary_status": generation_summary.get("status"),
            "elapsed_seconds_this_generation_invocation": generation_summary.get(
                "elapsed_seconds_this_invocation"
            ),
            "peak_vram_mb_this_generation_invocation": generation_summary.get(
                "peak_vram_mb_this_invocation"
            ),
        },
        "audit": {
            "protocol_commit": config["protocol_commit"],
            "config_sha256": _sha256_file(args.config),
            "manifest_sha256": _sha256_file(args.manifest),
            "policy_sha256": _sha256_file(args.policy),
            "rows_file_sha256": _sha256_file(args.rows),
            "canonical_rows_sha256": _row_hash(rows),
            "manifest_overlap_checks": manifest["overlap_checks"],
        },
        "boundary": config["boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/final_qa_confirmation.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "data/splits/final_qa/final_qa_confirmation_manifest.json",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT / "data/splits/final_qa/final_qa_final_gate_policy.json",
    )
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--generation-summary", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data/splits/final_qa/final_qa_confirmation_result.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
