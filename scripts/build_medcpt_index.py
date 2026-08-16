from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.retrieval.medcpt_retriever import build_medcpt_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a MedCPT dense retrieval index for OpenI cases.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    build_medcpt_index(
        cases_path=args.cases,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
    )
    print(f"Wrote MedCPT index to {args.output}")


if __name__ == "__main__":
    main()

