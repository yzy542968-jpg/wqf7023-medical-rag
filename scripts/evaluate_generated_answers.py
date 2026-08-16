from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated answers with lightweight automatic metrics.")
    parser.add_argument("--generations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = _read_jsonl(args.generations)
    if not rows:
        raise ValueError(f"No generation rows found: {args.generations}")

    f1_scores = []
    top1_hits = []
    retrieved_hits = []
    answer_lengths = []
    insufficient_count = 0

    for row in rows:
        answer = extract_final_answer(row.get("answer", ""))
        reference = row.get("reference_answer", "")
        relevant = set(row.get("relevant_case_ids", []))
        retrieved = row.get("retrieved_case_ids", [])
        f1_scores.append(token_f1(answer, reference))
        top1_hits.append(float(bool(retrieved) and retrieved[0] in relevant))
        retrieved_hits.append(float(bool(relevant.intersection(retrieved))))
        answer_lengths.append(len(answer.split()))
        if "insufficient" in answer.lower():
            insufficient_count += 1

    metrics = {
        "answer_token_f1": sum(f1_scores) / len(f1_scores),
        "top1_case_accuracy": sum(top1_hits) / len(top1_hits),
        "retrieved_case_hit_rate": sum(retrieved_hits) / len(retrieved_hits),
        "average_answer_words": sum(answer_lengths) / len(answer_lengths),
        "insufficient_answer_rate": insufficient_count / len(rows),
    }
    output = {
        "generations": str(args.generations),
        "record_count": len(rows),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
