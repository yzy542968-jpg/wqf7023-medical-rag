from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.radqa_answer_metrics import (
    evaluate_generation_records,
    summarize_generation_rows,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate V3 RadQA generations with multi-reference and unanswerable metrics."
    )
    parser.add_argument(
        "--generations",
        type=Path,
        default=ROOT
        / "experiments"
        / "benchmark_v3_radqa"
        / "radqa_v3_test_qwen15.jsonl",
    )
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=ROOT
        / "experiments"
        / "benchmark_v3_radqa"
        / "radqa_v3_test_agent_prompt_pack.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "benchmark_v3_radqa" / "generation_evaluation",
    )
    args = parser.parse_args()
    prompts = {row["qid"]: row for row in read_jsonl(args.prompt_pack)}
    rows = evaluate_generation_records(read_jsonl(args.generations), prompts)
    summary = {
        "benchmark": "RadQA natural-question evidence retrieval v3",
        "evaluation": "multi-reference exact match, Token-F1, and unanswerable accuracy",
        **summarize_generation_rows(rows),
        "generations_path": str(args.generations),
        "prompt_pack_path": str(args.prompt_pack),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "radqa_v3_generation_rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["rows_path"] = str(rows_path)
    summary_path = args.output_dir / "radqa_v3_generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "summary_path": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()

