from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.case_scoped_benchmark import expected_section
from medical_rag.retrieval.scoped_chunk_retriever import ScopedBM25ChunkRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock v2 top-k using only the calibration split.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_scoped_benchmark_v2.json",
    )
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--target-recall", type=float, default=0.95)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "benchmark_v2" / "calibration" / "locked_top_k.json",
    )
    args = parser.parse_args()

    payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
    calibration_qids = set(payload["split"]["calibration"]["qids"])
    questions = [row for row in payload["questions"] if row["qid"] in calibration_qids]
    retriever = ScopedBM25ChunkRetriever().fit(payload["chunks"])
    curve = []
    for top_k in range(1, args.max_k + 1):
        recalls = []
        context_sizes = []
        context_characters = []
        for question in questions:
            results = retriever.search(
                question["question"],
                top_k=top_k,
                case_id=question["scope_case_id"],
                allowed_sections={expected_section(question["question_type"])},
            )
            retrieved = {row["chunk_id"] for row in results}
            relevant = set(question["relevant_chunk_ids"])
            recalls.append(len(retrieved & relevant) / len(relevant))
            context_sizes.append(len(results))
            context_characters.append(sum(len(row["text"]) for row in results))
        curve.append(
            {
                "top_k": top_k,
                "mean_recall": mean(recalls),
                "complete_evidence_coverage_rate": mean(value == 1.0 for value in recalls),
                "mean_retrieved_chunks": mean(context_sizes),
                "mean_context_characters": mean(context_characters),
            }
        )

    eligible = [row for row in curve if row["mean_recall"] >= args.target_recall]
    selected = eligible[0]["top_k"] if eligible else curve[-1]["top_k"]
    result = {
        "selection_partition": "calibration",
        "selection_case_count": payload["split"]["calibration"]["case_count"],
        "selection_question_count": len(questions),
        "target_mean_recall": args.target_recall,
        "selected_top_k": selected,
        "selection_rule": "smallest k reaching target mean evidence recall",
        "curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({**result, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
