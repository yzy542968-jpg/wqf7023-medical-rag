from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.retrieval.tfidf_retriever import TfidfRetriever, load_cases_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TF-IDF retrieval over OpenI cases.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", default=5, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    retriever = TfidfRetriever().fit(cases)
    results = retriever.search(args.query, top_k=args.top_k)

    payload = {
        "query": args.query,
        "top_k": args.top_k,
        "results": results,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

