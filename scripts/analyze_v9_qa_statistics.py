from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROWS = ROOT / "experiments" / "post_submission_v9" / "v9_qa_raw_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "experiments" / "post_submission_v9" / "v9_qa_raw_summary.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v9" / "v9_full_source_split.json"
DEFAULT_OUTPUT = ROOT / "data" / "splits" / "v9" / "v9_qa_statistical_analysis.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_scores(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["case_id"])][str(row["system"])].append(float(row["token_f1"]))
    return {
        case_id: {system: statistics.fmean(values) for system, values in systems.items()}
        for case_id, systems in grouped.items()
    }


def paired_bootstrap(
    scores: Mapping[str, Mapping[str, float]],
    left: str,
    right: str,
    *,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    differences = np.asarray(
        [float(values[left]) - float(values[right]) for values in scores.values()],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        samples[index] = float(rng.choice(differences, len(differences), replace=True).mean())
    return {
        "case_count": len(differences),
        "difference": float(differences.mean()),
        "ci_95_low": float(np.quantile(samples, 0.025)),
        "ci_95_high": float(np.quantile(samples, 0.975)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze frozen V9 QA outcomes by case.")
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if summary["status"] != "formal_test_qa_outcomes_frozen_no_retuning":
        raise RuntimeError("V9 QA outcomes are not frozen.")
    if sha256(args.rows) != summary["outputs"]["rows_sha256"]:
        raise RuntimeError("V9 QA rows changed after the raw summary was frozen.")
    rows = read_jsonl(args.rows)
    if len(rows) != 5480:
        raise RuntimeError("V9 QA matrix is incomplete.")
    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    strict_ids = set(map(str, split["strict_project_history_untouched_test_subset"]["case_ids"]))

    systems = [
        "g0_no_retrieval",
        "g1_bm25_rag",
        "g2_fixed_multimodal_rag",
        "g3_learned_multimodal_rag",
    ]
    comparisons = [
        ("g1_minus_g0", systems[1], systems[0]),
        ("g2_minus_g0", systems[2], systems[0]),
        ("g3_minus_g0", systems[3], systems[0]),
        ("g3_minus_g2", systems[3], systems[2]),
        ("g3_minus_g1", systems[3], systems[1]),
    ]
    frames: dict[str, list[dict[str, Any]]] = {"all": rows}
    frames.update(
        {
            f"question_{question_type}": [row for row in rows if row["question_type"] == question_type]
            for question_type in ("findings", "impression")
        }
    )
    frames["report_indexed_normal"] = [
        row for row in rows if str(cases[str(row["case_id"])].get("problems", "")).strip().lower() == "normal"
    ]
    frames["report_indexed_abnormal"] = [
        row for row in rows if str(cases[str(row["case_id"])].get("problems", "")).strip().lower() not in {"", "normal", "no indexing"}
    ]
    frames["strict_project_history_untouched"] = [row for row in rows if str(row["case_id"]) in strict_ids]

    frame_results = {}
    for frame_index, (frame_name, frame_rows) in enumerate(frames.items()):
        scores = case_scores(frame_rows)
        frame_results[frame_name] = {
            "case_count": len(scores),
            "system_token_f1": {
                system: statistics.fmean(values[system] for values in scores.values())
                for system in systems
            },
            "comparisons": {
                name: paired_bootstrap(
                    scores,
                    left,
                    right,
                    iterations=10000,
                    seed=7032 + frame_index * 10 + comparison_index,
                )
                for comparison_index, (name, left, right) in enumerate(comparisons)
            },
        }
    output = {
        "analysis": "V9 frozen QA case-grouped statistical analysis",
        "status": "complete_no_retuning",
        "rows_sha256": sha256(args.rows),
        "raw_summary_sha256": sha256(args.summary),
        "script_sha256": sha256(Path(__file__)),
        "bootstrap_iterations": 10000,
        "confidence_level": 0.95,
        "primary_comparison": "all.g3_minus_g0",
        "secondary_comparisons_are_descriptive": True,
        "frames": frame_results,
        "human_clinical_adjudication": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
