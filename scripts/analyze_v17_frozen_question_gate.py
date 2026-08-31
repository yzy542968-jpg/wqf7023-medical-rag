"""Apply the pre-V17 frozen Final-QA question-ID gate to V17 whole-report rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_v17_generation_pilot import _case_bootstrap, _metrics  # noqa: E402


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(args: argparse.Namespace) -> dict[str, Any]:
    retrieval = {
        (str(row["case_id"]), int(row["question_index"])): int(row["question_id"])
        for row in _read_jsonl(args.retrieval_rows)
    }
    generated: dict[str, dict[tuple[str, int], dict[str, Any]]] = defaultdict(dict)
    for row in _read_jsonl(args.generation_rows):
        generated[str(row["condition"])][(str(row["case_id"]), int(row["question_index"]))] = row
    keys = sorted(generated["no_history"])
    if any(set(generated[arm]) != set(keys) for arm in ("related", "random", "mismatched")):
        raise RuntimeError("Whole-report V17 arms are not paired")
    policy = _load_json(args.policy)
    use_history = {
        int(row["question_id"]): str(row["source"]).startswith("b6_")
        for row in policy["question_policy"]
    }
    selected: dict[str, dict[tuple[str, int], dict[str, Any]]] = {
        arm: {} for arm in ("no_history", "related", "random", "mismatched")
    }
    selected_question_count = 0
    for key in keys:
        decision = use_history.get(retrieval[key], False)
        selected_question_count += int(decision)
        selected["no_history"][key] = generated["no_history"][key]
        for arm in ("related", "random", "mismatched"):
            selected[arm][key] = generated[arm][key] if decision else generated["no_history"][key]
    metrics = {arm: _metrics(rows.values()) for arm, rows in selected.items()}
    comparisons = {
        f"selective_related_minus_{arm}": _case_bootstrap(
            selected["related"], selected[arm], seed=17017, replicates=10000
        )
        for arm in ("no_history", "random", "mismatched")
    }
    result = {
        "study": "V17 pre-existing frozen question-ID gate sensitivity",
        "status": "exploratory_sensitivity_complete",
        "data_role": "final_qa_calibration",
        "test_accessed": False,
        "policy_history_question_id_count": int(policy["history_question_id_count"]),
        "selected_question_count": selected_question_count,
        "selected_question_fraction": selected_question_count / len(keys),
        "condition_metrics": metrics,
        "comparisons": comparisons,
        "boundary": (
            "The gate was frozen before V17 using a case-disjoint Final-QA role, but this "
            "remains same-source exploratory sensitivity analysis, not independent confirmation."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-rows", type=Path, default=ROOT / "experiments/v17_exploratory/v17_retrieval_calibration_rows.jsonl")
    parser.add_argument("--generation-rows", type=Path, default=ROOT / "experiments/v17_exploratory/v17_whole_report_rows.jsonl")
    parser.add_argument("--policy", type=Path, default=ROOT / "data/splits/final_qa/final_qa_final_gate_policy.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/v17_exploratory/v17_frozen_question_gate_summary.json")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))

