from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.benchmark_v2_validity import audit_human_evaluation


REGISTRY_PATH = Path("experiments/final_submission/final_results_registry.json")
MANIFEST_PATH = Path("experiments/final_submission/submission_manifest.json")
DECISIONS_PATH = Path("config/submission_decisions.json")
GITHUB_LIMIT_BYTES = 100 * 1024 * 1024

REQUIRED_FILES = [
    Path("README.md"),
    Path("requirements.txt"),
    Path("app.py"),
    Path("human_evaluation_app.py"),
    Path("docs/P2_FINAL_MANUSCRIPT.md"),
    Path("docs/P2_PROJECT_CONTROL.md"),
    Path("docs/BENCHMARK_V3_RADQA_RUNBOOK.md"),
    DECISIONS_PATH,
    REGISTRY_PATH,
]

DELIVERABLE_VARIANTS = {
    "manuscript_docx": [
        Path("deliverables/22097191_ZHANG_YUE_P2_Research_Project.docx"),
        Path("deliverables/22097191_ZHANG_YUE_P2_Research_Project_DRAFT.docx"),
    ],
    "manuscript_pdf": [
        Path("deliverables/22097191_ZHANG_YUE_P2_Research_Project.pdf"),
        Path("deliverables/22097191_ZHANG_YUE_P2_Research_Project_DRAFT.pdf"),
    ],
    "defence_pptx": [
        Path("deliverables/22097191_ZHANG_YUE_P2_Defence.pptx"),
        Path("deliverables/22097191_ZHANG_YUE_P2_Defence_DRAFT.pptx"),
    ],
}

HUMAN_FILES = {
    "v1": Path(
        "experiments/final_optimized/human_evaluation/"
        "held_out_blinded_human_evaluation_36.csv"
    ),
    "v2": Path(
        "experiments/benchmark_v2/human_evaluation/"
        "v2_confirmation_blinded_human_evaluation_36.csv"
    ),
}

FORBIDDEN_TRACKED = re.compile(
    r"(^|/)(data/raw|\.venv|\.hf_cache|\.cache|__pycache__|\.pytest_cache)(/|$)"
    r"|(^|/)(secrets\.toml|\.env(?:\..*)?)$"
    r"|\.(?:pem|key|pt|pth|bin|safetensors)$",
    flags=re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def package_versions() -> dict[str, str | None]:
    names = [
        "accelerate",
        "numpy",
        "pandas",
        "Pillow",
        "pytest",
        "python-docx",
        "radgraph",
        "streamlit",
        "torch",
        "transformers",
    ]
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def cuda_status() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch_not_installed"}
    available = bool(torch.cuda.is_available())
    return {
        "available": available,
        "torch_cuda_build": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()) if available else 0,
        "device_name": torch.cuda.get_device_name(0) if available else None,
    }


def tracked_files() -> list[Path]:
    return [Path(value) for value in git("ls-files").splitlines() if value.strip()]


def resolve_human_evaluation(
    human: dict[str, dict[str, Any]], policy: dict[str, Any]
) -> tuple[bool, bool]:
    complete = len(human) == len(HUMAN_FILES) and all(
        value["completed_rows"] == value["rows"] for value in human.values()
    )
    explicitly_not_conducted = (
        len(human) == len(HUMAN_FILES)
        and policy.get("status") == "not_conducted"
        and policy.get("limitations_declared") is True
        and policy.get("scores_claimed") is False
        and all(value["completed_rows"] == 0 for value in human.values())
    )
    return complete or explicitly_not_conducted, complete


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and audit the WQF7023 P2 submission manifest."
    )
    parser.add_argument("--output", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when external submission gates are still pending.",
    )
    args = parser.parse_args()

    missing = [str(path) for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    decisions = (
        json.loads((ROOT / DECISIONS_PATH).read_text(encoding="utf-8"))
        if (ROOT / DECISIONS_PATH).is_file()
        else {}
    )
    human_policy = decisions.get("human_evaluation", {})
    selected_deliverables: dict[str, Path] = {}
    for artifact_type, variants in DELIVERABLE_VARIANTS.items():
        selected = next((path for path in variants if (ROOT / path).is_file()), None)
        if selected is None:
            missing.append(f"{artifact_type}: one of {', '.join(map(str, variants))}")
        else:
            selected_deliverables[artifact_type] = selected
    registry: dict[str, Any] = {}
    locked_hash_mismatches: list[dict[str, str]] = []
    if (ROOT / REGISTRY_PATH).is_file():
        registry = json.loads((ROOT / REGISTRY_PATH).read_text(encoding="utf-8"))
        for relative, expected in registry.get("locked_artifact_sha256", {}).items():
            artifact = ROOT / relative
            actual = sha256(artifact) if artifact.is_file() else "missing"
            if actual != expected:
                locked_hash_mismatches.append(
                    {"path": relative, "expected": expected, "actual": actual}
                )

    human = {
        name: audit_human_evaluation(ROOT / path)
        for name, path in HUMAN_FILES.items()
        if (ROOT / path).is_file()
    }
    human_resolved, human_complete = resolve_human_evaluation(human, human_policy)
    manuscript_text = (
        (ROOT / "docs/P2_FINAL_MANUSCRIPT.md").read_text(encoding="utf-8")
        if (ROOT / "docs/P2_FINAL_MANUSCRIPT.md").is_file()
        else ""
    )
    human_claim_consistent = human_complete or (
        human_policy.get("status") == "not_conducted"
        and "No independent human evaluation was conducted" in manuscript_text
        and registry.get("human_evaluation_policy") == human_policy
    )

    tracked = tracked_files()
    forbidden = [path.as_posix() for path in tracked if FORBIDDEN_TRACKED.search(path.as_posix())]
    oversized = [
        {"path": path.as_posix(), "bytes": (ROOT / path).stat().st_size}
        for path in tracked
        if (ROOT / path).is_file() and (ROOT / path).stat().st_size >= GITHUB_LIMIT_BYTES
    ]

    deliverables = {}
    for artifact_type, path in selected_deliverables.items():
        deliverables[artifact_type] = {
            "path": path.as_posix(),
            "bytes": (ROOT / path).stat().st_size,
            "sha256": sha256(ROOT / path),
            "draft": "DRAFT" in path.name.upper(),
        }

    official_radqa = [
        Path("data/raw/radqa/train.json"),
        Path("data/raw/radqa/dev.json"),
        Path("data/raw/radqa/test.json"),
    ]
    official_radqa_available = all((ROOT / path).is_file() for path in official_radqa)
    remote_url = git("remote", "get-url", "origin")
    head = git("rev-parse", "--verify", "HEAD")

    automated_gates = {
        "required_files_present": not missing,
        "locked_artifact_hashes_match": not locked_hash_mismatches,
        "tracked_release_exclusions_pass": not forbidden,
        "github_file_size_limit_pass": not oversized,
        "human_evaluation_claim_consistency_pass": human_claim_consistent,
    }
    external_gates = {
        "human_evaluation_disposition_resolved": human_resolved,
        "remote_repository_configured": bool(remote_url),
        "non_draft_submission_artifacts_present": len(deliverables)
        == len(DELIVERABLE_VARIANTS)
        and all(not value["draft"] for value in deliverables.values()),
    }

    manifest = {
        "project": "WQF7023 Medical RAG",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "automated_ready": all(automated_gates.values()),
        "submission_ready": all(automated_gates.values()) and all(external_gates.values()),
        "automated_gates": automated_gates,
        "external_gates": external_gates,
        "missing_required_files": missing,
        "locked_hash_mismatches": locked_hash_mismatches,
        "forbidden_tracked_files": forbidden,
        "oversized_tracked_files": oversized,
        "human_evaluation": {
            "policy": human_policy,
            "complete": human_complete,
            "disposition_resolved": human_resolved,
            "packages": human,
        },
        "deliverables": deliverables,
        "v3_radqa": {
            "framework_implemented": all(
                (ROOT / path).is_file()
                for path in [
                    Path("scripts/build_radqa_benchmark_v3.py"),
                    Path("scripts/evaluate_radqa_agent_v3.py"),
                    Path("scripts/evaluate_radqa_generation_v3.py"),
                ]
            ),
            "official_credentialed_files_available": official_radqa_available,
            "thesis_result_status": "eligible_to_run" if official_radqa_available else "conditional_only",
        },
        "git": {
            "branch": git("branch", "--show-current") or None,
            "head": head or None,
            "remote_origin": remote_url or None,
            "tracked_file_count": len(tracked),
            "working_tree_clean": not bool(git("status", "--porcelain")),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "cuda": cuda_status(),
        },
    }

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

    if not manifest["automated_ready"]:
        raise SystemExit(1)
    if args.strict and not manifest["submission_ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
