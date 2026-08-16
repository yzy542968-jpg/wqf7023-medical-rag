from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class DashboardRuntime:
    mode: str
    cases_path: Path
    dense_index_path: Path | None
    full_case_count: int
    reason: str

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"

    @property
    def dense_retrieval_available(self) -> bool:
        return self.dense_index_path is not None


def _jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(bool(line.strip()) for line in handle)


def resolve_dashboard_runtime(root: Path) -> DashboardRuntime:
    full_cases = root / "data" / "processed" / "openi_cases.jsonl"
    dense_index = root / "data" / "processed" / "openi_medcpt_full.npz"
    demo_cases = root / "data" / "processed" / "sample_cases.jsonl"

    force_demo = os.environ.get("MEDICAL_RAG_DEMO_MODE", "").strip() == "1"
    if full_cases.exists() and not force_demo:
        return DashboardRuntime(
            mode="full",
            cases_path=full_cases,
            dense_index_path=dense_index if dense_index.exists() else None,
            full_case_count=_jsonl_count(full_cases),
            reason=(
                "full OpenI case file and dense index available"
                if dense_index.exists()
                else "full OpenI case file available; dense index missing"
            ),
        )
    if not demo_cases.exists():
        raise FileNotFoundError(
            "Neither the full OpenI case file nor the tracked demo case file is available."
        )
    return DashboardRuntime(
        mode="demo",
        cases_path=demo_cases,
        dense_index_path=None,
        full_case_count=_jsonl_count(demo_cases),
        reason=(
            "Demo Mode was explicitly requested"
            if force_demo
            else "full OpenI artifacts are absent; using tracked software-demo cases"
        ),
    )
