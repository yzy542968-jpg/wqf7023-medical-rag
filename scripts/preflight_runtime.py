from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.dashboard.runtime import resolve_dashboard_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Dashboard data readiness.")
    parser.add_argument(
        "--require-full",
        action="store_true",
        help="Fail unless the local OpenI case file and dense index are both available.",
    )
    args = parser.parse_args()
    runtime = resolve_dashboard_runtime(ROOT)
    payload = {
        "mode": runtime.mode,
        "case_count": runtime.full_case_count,
        "cases_path": str(runtime.cases_path),
        "dense_index_path": (
            str(runtime.dense_index_path) if runtime.dense_index_path else None
        ),
        "dense_retrieval_available": runtime.dense_retrieval_available,
        "reason": runtime.reason,
        "ready": not args.require_full
        or (not runtime.is_demo and runtime.dense_retrieval_available),
    }
    print(json.dumps(payload, indent=2))
    if not payload["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
