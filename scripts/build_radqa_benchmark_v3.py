from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.radqa_benchmark import (
    benchmark_summary,
    build_radqa_benchmark,
    load_radqa_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the V3 natural-question benchmark from credentialed RadQA files."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=ROOT / "data" / "raw" / "radqa"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "radqa_natural_qa_benchmark_v3.json",
    )
    args = parser.parse_args()
    payload = build_radqa_benchmark(load_radqa_files(args.input_dir))
    payload["source_directory"] = str(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output), **benchmark_summary(payload)}, indent=2))


if __name__ == "__main__":
    main()

