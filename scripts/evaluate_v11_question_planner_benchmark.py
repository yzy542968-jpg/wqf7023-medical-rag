"""Evaluate the deterministic V11 question planner on a frozen author set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.similar_case.v11_question_planner import plan_question  # noqa: E402


DEFAULT_BENCHMARK = ROOT / "data/splits/v11/v11_question_planner_benchmark.json"
DEFAULT_OUTPUT = ROOT / "data/splits/v11/v11_question_planner_benchmark_summary.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload: dict[str, Any] = json.loads(args.benchmark.read_text(encoding="utf-8"))
    examples = payload["examples"]
    predictions = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        predicted = plan_question(example["question"]).intent
        expected = str(example["intent"])
        predictions.append({"id": example["id"], "expected": expected, "predicted": predicted})
        confusion[expected][predicted] += 1
    labels = sorted({str(example["intent"]) for example in examples})
    per_intent = {}
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values())
        predicted_count = sum(confusion[other][label] for other in labels)
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_intent[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    summary = {
        "study": "V11 deterministic question planner benchmark",
        "status": "development_only_author_defined_intent_labels",
        "benchmark_sha256": sha256(args.benchmark),
        "planner_source_sha256": sha256(ROOT / "src/medical_rag/similar_case/v11_question_planner.py"),
        "example_count": len(examples),
        "accuracy": sum(row["expected"] == row["predicted"] for row in predictions) / len(predictions),
        "macro_f1": sum(float(row["f1"]) for row in per_intent.values()) / len(per_intent),
        "per_intent": per_intent,
        "confusion": {label: dict(sorted(confusion[label].items())) for label in labels},
        "claim_boundary": "This is an author-defined intent robustness benchmark, not physician annotation or clinical validation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
