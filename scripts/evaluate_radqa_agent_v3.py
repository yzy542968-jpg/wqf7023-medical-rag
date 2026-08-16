from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.radqa_agent import (
    SYSTEMS,
    answerability_metrics,
    build_agent_prompt,
    evaluate_retrieval_system,
    select_answerability_threshold,
)
from medical_rag.retrieval.scoped_chunk_retriever import ScopedBM25ChunkRetriever


def rows_for_split(payload: dict[str, Any], split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    qids = set(payload["split"][split]["qids"])
    chunk_ids = set(payload["split"][split]["chunk_ids"])
    return (
        [row for row in payload["questions"] if row["qid"] in qids],
        [row for row in payload["chunks"] if row["chunk_id"] in chunk_ids],
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate and evaluate the V3 RadQA evidence-retrieval agent."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "processed" / "radqa_natural_qa_benchmark_v3.json",
    )
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "benchmark_v3_radqa",
    )
    args = parser.parse_args()
    payload = json.loads(args.benchmark.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "benchmark": payload["benchmark"],
        "content_fingerprint_sha256": payload["content_fingerprint_sha256"],
        "top_k": args.top_k,
        "systems": {},
    }
    system_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for split in ("dev", "test"):
        questions, chunks = rows_for_split(payload, split)
        retriever = ScopedBM25ChunkRetriever().fit(chunks)
        for system in SYSTEMS:
            metrics, rows = evaluate_retrieval_system(
                questions, chunks, retriever, system, args.top_k
            )
            summary["systems"].setdefault(system, {})[split] = metrics
            system_rows[(system, split)] = rows

    calibration = select_answerability_threshold(
        system_rows[("report_scoped_bm25", "dev")]
    )
    threshold = float(calibration["selected"]["threshold"])
    confirmation = answerability_metrics(
        system_rows[("report_scoped_bm25", "test")], threshold
    )
    summary["answerability_action"] = {
        "calibration_split": "dev",
        "final_split": "test",
        "selection": calibration,
        "test": confirmation,
    }

    test_questions, _ = rows_for_split(payload, "test")
    test_question_by_id = {row["qid"]: row for row in test_questions}
    test_rows = system_rows[("report_scoped_bm25", "test")]
    prompt_rows = [
        build_agent_prompt(test_question_by_id[row["qid"]], row, threshold)
        for row in test_rows
    ]
    summary_path = args.output_dir / "radqa_v3_agent_summary.json"
    rows_path = args.output_dir / "radqa_v3_test_retrieval_rows.jsonl"
    prompts_path = args.output_dir / "radqa_v3_test_agent_prompt_pack.jsonl"
    summary["test_rows_path"] = str(rows_path)
    summary["test_prompt_pack_path"] = str(prompts_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl(rows_path, test_rows)
    write_jsonl(prompts_path, prompt_rows)
    print(json.dumps({**summary, "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()

