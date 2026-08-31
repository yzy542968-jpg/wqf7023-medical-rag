"""Cross-fit an inference-time evidence-relevance gate on frozen V17 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.qa.radrestruct import iter_radrestruct_cases  # noqa: E402
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v11_evidence import select_hierarchical_evidence  # noqa: E402
from medical_rag.similar_case.v11_question_planner import plan_question  # noqa: E402
from run_v17_generation_pilot import _case_bootstrap, _metrics  # noqa: E402


ARMS = ("related", "random", "mismatched")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _fold(case_id: str, seed: int, folds: int) -> int:
    digest = hashlib.sha256(f"v17-relevance-gate|{seed}|{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _exact(row: dict[str, Any]) -> float:
    return float(set(row["gold_indices"]) == set(row["predicted_indices"]))


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = _load_json(args.config)
    manifest = _load_json(args.pilot_manifest)
    case_ids = set(str(value) for value in manifest["case_ids"])
    retrieval = {
        (str(row["case_id"]), int(row["question_index"])): row
        for row in _read_jsonl(args.retrieval_rows)
        if str(row["case_id"]) in case_ids
    }
    generation_rows = _read_jsonl(args.generation_rows)
    generated: dict[str, dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    for row in generation_rows:
        generated[str(row["condition"])][(str(row["case_id"]), int(row["question_index"]))] = row
    keys = sorted(generated["no_history"])
    if any(set(generated[arm]) != set(keys) for arm in ("no_history", *ARMS)):
        raise RuntimeError("V17 generation arms are not paired")

    raw_cases = {str(row["case_id"]): row for row in _read_jsonl(args.cases)}
    radgraph = read_radgraph_case_records(args.radgraph)
    facts = {case_id: tuple(record.facts) for case_id, record in radgraph.items() if record.status == "ok"}
    questions = {
        (case.case_id, index): question
        for case in iter_radrestruct_cases(args.radrestruct_root)
        if case.case_id in case_ids
        for index, question in enumerate(case.questions)
    }
    features: dict[str, dict[tuple[str, int], dict[str, float]]] = {arm: {} for arm in ARMS}
    evidence_cfg = config["evidence"]
    for key in keys:
        target_id, _index = key
        question = questions[key]
        plan = plan_question(question.question, raw_cases[target_id].get("indication", ""))
        for arm in ARMS:
            history_ids = [str(value) for value in retrieval[key]["control_rankings"][arm]]
            selected = select_hierarchical_evidence(
                [raw_cases[case_id] for case_id in history_ids],
                query=question.question,
                facts_by_case={case_id: facts.get(case_id, ()) for case_id in history_ids},
                plan=plan,
                maximum_cases=int(evidence_cfg["maximum_cases"]),
                maximum_units_per_case=int(evidence_cfg["maximum_units_per_case"]),
                maximum_total_units=int(evidence_cfg["maximum_total_units"]),
                maximum_characters=int(evidence_cfg["maximum_characters"]),
            )
            scores = np.asarray([unit.score for unit in selected.units], dtype=np.float64)
            features[arm][key] = {
                "max_score": float(scores.max()) if len(scores) else 0.0,
                "mean_score": float(scores.mean()) if len(scores) else 0.0,
                "top2_mean_score": float(np.sort(scores)[-2:].mean()) if len(scores) else 0.0,
            }

    folds = 5
    fold_by_case = {case_id: _fold(case_id, int(config["seed"]), folds) for case_id in case_ids}
    selected_rows: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        arm: {} for arm in ("no_history", *ARMS)
    }
    decisions: list[dict[str, Any]] = []
    for outer_fold in range(folds):
        training_keys = [key for key in keys if fold_by_case[key[0]] != outer_fold]
        evaluation_keys = [key for key in keys if fold_by_case[key[0]] == outer_fold]
        candidates: list[tuple[float, float, str]] = []
        for feature_name in ("max_score", "mean_score", "top2_mean_score"):
            training_scores = np.asarray([features["related"][key][feature_name] for key in training_keys])
            thresholds = sorted(set(float(value) for value in np.quantile(training_scores, np.linspace(0, 1, 21))))
            thresholds.append(float("inf"))
            for threshold in thresholds:
                exact = np.mean([
                    _exact(generated["related"][key])
                    if features["related"][key][feature_name] >= threshold
                    else _exact(generated["no_history"][key])
                    for key in training_keys
                ])
                candidates.append((float(exact), float(threshold), feature_name))
        best_exact = max(value[0] for value in candidates)
        eligible = [value for value in candidates if abs(value[0] - best_exact) <= 1e-12]
        best_exact, threshold, feature_name = max(eligible, key=lambda value: (value[1], value[2]))
        decisions.append(
            {
                "outer_fold": outer_fold,
                "training_case_count": len({key[0] for key in training_keys}),
                "evaluation_case_count": len({key[0] for key in evaluation_keys}),
                "feature": feature_name,
                "threshold": threshold if np.isfinite(threshold) else "infinity_no_history",
                "training_related_selective_exact": best_exact,
            }
        )
        for key in evaluation_keys:
            selected_rows["no_history"][key] = generated["no_history"][key]
            for arm in ARMS:
                use_history = features[arm][key][feature_name] >= threshold
                chosen = generated[arm][key] if use_history else generated["no_history"][key]
                selected_rows[arm][key] = chosen
                chosen = dict(chosen)
                chosen["v17_relevance_gate_used_history"] = use_history

    metrics = {arm: _metrics(rows.values()) for arm, rows in selected_rows.items()}
    coverage = {
        arm: float(np.mean([
            selected_rows[arm][key] is generated[arm][key] for key in keys
        ]))
        for arm in ARMS
    }
    comparisons = {
        f"selective_related_minus_{arm}": _case_bootstrap(
            selected_rows["related"],
            selected_rows[arm],
            seed=int(config["seed"]),
            replicates=int(config["evaluation"]["bootstrap_replicates"]),
        )
        for arm in ("no_history", "random", "mismatched")
    }
    relevance_specific = (
        metrics["related"]["exact_answer_set_accuracy"] > metrics["random"]["exact_answer_set_accuracy"]
        and metrics["related"]["exact_answer_set_accuracy"] > metrics["mismatched"]["exact_answer_set_accuracy"]
    )
    result = {
        "study": "V17 cross-fitted inference-time evidence-relevance gate",
        "status": "relevance_specific_signal" if relevance_specific else "mixed_or_no_relevance_specific_signal",
        "data_role": "final_qa_calibration_cross_fitted",
        "test_accessed": False,
        "case_count": len(case_ids),
        "question_count": len(keys),
        "outer_fold_decisions": decisions,
        "history_coverage": coverage,
        "condition_metrics": metrics,
        "comparisons": comparisons,
        "related_exceeds_both_matched_controls": relevance_specific,
        "boundary": (
            "Post-hoc case-level cross-fitted Calibration analysis using only inference-time "
            "evidence scores; not independent confirmation or clinical accuracy."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/v17_generation_pilot_qlora_amendment.json")
    parser.add_argument("--pilot-manifest", type=Path, default=ROOT / "data/splits/final_qa/v17_generation_pilot_manifest.json")
    parser.add_argument("--retrieval-rows", type=Path, default=ROOT / "experiments/v17_exploratory/v17_retrieval_calibration_rows.jsonl")
    parser.add_argument("--generation-rows", type=Path, default=ROOT / "experiments/v17_exploratory/v17_generation_pilot_qlora_rows.jsonl")
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--radgraph", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v17_exploratory/v17_relevance_gate_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
