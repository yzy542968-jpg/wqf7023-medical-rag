from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.case_scoped_benchmark import content_fingerprint


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _case_ids(payload: dict[str, Any]) -> set[str]:
    ids = {
        str(row["case_id"])
        for row in payload.get("questions", [])
        if row.get("case_id") is not None
    }
    ids.update(str(value) for value in payload.get("case_ids", []))
    for part in payload.get("split", {}).values():
        ids.update(str(value) for value in part.get("case_ids", []))
    return ids


def main() -> None:
    failures: list[str] = []
    protected = _read(ROOT / "config" / "post_submission_protected_artifacts.json")
    protected_checks = {}
    for relative, expected in protected["artifacts"].items():
        path = ROOT / relative
        actual = _sha256(path) if path.exists() else None
        matched = actual == expected
        protected_checks[relative] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matched": matched,
        }
        if not matched:
            failures.append(f"protected artifact changed: {relative}")

    prior_paths = [
        ROOT / "data" / "processed" / "openi_case_qa_seed_clean.json",
        ROOT / "data" / "processed" / "openi_case_scoped_benchmark_v2.json",
        ROOT / "data" / "processed" / "openi_case_scoped_confirmation_v2.json",
    ]
    prior_ids: set[str] = set()
    for path in prior_paths:
        prior_ids.update(_case_ids(_read(path)))
    hard = _read(ROOT / "data" / "processed" / "openi_case_scoped_hard_v21.json")
    replication = _read(
        ROOT / "data" / "processed" / "openi_locked_replication_cohort.json"
    )
    hard_ids = _case_ids(hard)
    replication_ids = _case_ids(replication)
    overlaps = {
        "prior_vs_v21": sorted(prior_ids & hard_ids),
        "prior_vs_replication": sorted(prior_ids & replication_ids),
        "v21_vs_replication": sorted(hard_ids & replication_ids),
    }
    for name, values in overlaps.items():
        if values:
            failures.append(f"case overlap in {name}: {len(values)}")

    hard_fingerprint = content_fingerprint(hard["questions"], hard["chunks"])
    if hard_fingerprint != hard["content_fingerprint_sha256"]:
        failures.append("v2.1 content fingerprint mismatch")
    replication_canonical = json.dumps(
        replication["questions"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    replication_fingerprint = hashlib.sha256(
        replication_canonical.encode("utf-8")
    ).hexdigest()
    if replication_fingerprint != replication["content_fingerprint_sha256"]:
        failures.append("replication content fingerprint mismatch")

    hard_summary = _read(ROOT / "experiments" / "post_submission_v21" / "summary.json")
    transfer_summary = _read(
        ROOT
        / "experiments"
        / "post_submission_v21"
        / "template_transfer"
        / "summary.json"
    )
    planner_manifest = _read(
        ROOT
        / "experiments"
        / "post_submission_v22"
        / "planner_pack_manifest.json"
    )
    semantic_summary = _read(
        ROOT / "experiments" / "post_submission_v22" / "summary.json"
    )
    replication_summary = _read(
        ROOT / "experiments" / "locked_replication" / "summary.json"
    )
    if hard_summary["benchmark_content_fingerprint_sha256"] != hard_fingerprint:
        failures.append("v2.1 result summary points to different benchmark content")
    transfer_protocol = transfer_summary.get("protocol", {})
    if transfer_protocol.get("parameter_tuning_after_evaluation") is not False:
        failures.append("template-transfer no-retuning declaration is missing")
    expected_transfer_count = hard["split"]["test"]["question_count"]
    if transfer_protocol.get("question_count") != expected_transfer_count:
        failures.append("template-transfer question count does not match v2.1 test")
    expected_threshold = hard_summary["threshold_selection"][
        "closed_loop_agent_v2"
    ]["threshold"]
    if transfer_protocol.get("answerability_threshold") != expected_threshold:
        failures.append("template-transfer threshold differs from frozen development threshold")
    semantic_protocol = semantic_summary.get("protocol", {})
    if semantic_protocol.get("planner_prompt_sha256") != planner_manifest.get(
        "planner_prompt_sha256"
    ):
        failures.append("v2.2 semantic-planner prompt hash mismatch")
    if semantic_protocol.get("prompt_frozen_before_generation") is not True:
        failures.append("v2.2 prompt-freeze declaration is missing")
    if semantic_protocol.get("post_test_prompt_changes_permitted") is not False:
        failures.append("v2.2 post-test prompt-change policy is invalid")
    if semantic_protocol.get("test_or_transfer_tuning") is not False:
        failures.append("v2.2 no-test-tuning declaration is missing")
    if semantic_summary.get("planner", {}).get("n") != planner_manifest.get(
        "record_count"
    ):
        failures.append("v2.2 planner output count does not match manifest")
    if semantic_summary.get("planner", {}).get("parse_failure_count") != 0:
        failures.append("v2.2 planner contains parse failures")
    if replication_summary.get("status") != "complete":
        failures.append("locked replication is not complete")
    if replication_summary.get("generation", {}).get("unique_qid_count") != replication[
        "question_count"
    ]:
        failures.append("replication generation count does not match cohort")
    if not replication_summary.get("no_replication_tuning"):
        failures.append("replication no-tuning declaration is missing")

    output = {
        "audit": "post_submission_release",
        "passed": not failures,
        "source_tag": protected["source_tag"],
        "protected_artifacts": protected_checks,
        "case_counts": {
            "prior_union": len(prior_ids),
            "v21": len(hard_ids),
            "replication": len(replication_ids),
        },
        "case_overlaps": {name: len(values) for name, values in overlaps.items()},
        "fingerprints": {
            "v21": hard_fingerprint,
            "replication": replication_fingerprint,
        },
        "replication_complete": replication_summary.get("status") == "complete",
        "template_transfer": {
            "question_count": transfer_protocol.get("question_count"),
            "template_fingerprint_sha256": transfer_protocol.get(
                "template_fingerprint_sha256"
            ),
            "retuned_after_evaluation": transfer_protocol.get(
                "parameter_tuning_after_evaluation"
            ),
        },
        "semantic_planner": {
            "prompt_sha256": semantic_protocol.get("planner_prompt_sha256"),
            "record_count": semantic_summary.get("planner", {}).get("n"),
            "parse_failure_count": semantic_summary.get("planner", {}).get(
                "parse_failure_count"
            ),
            "test_or_transfer_tuning": semantic_protocol.get(
                "test_or_transfer_tuning"
            ),
            "original_macro_f1": semantic_summary.get("original_test", {})
            .get("raw", {})
            .get("macro_f1"),
            "transfer_macro_f1": semantic_summary.get("transfer_test", {})
            .get("raw", {})
            .get("macro_f1"),
        },
        "human_evaluation_disposition": "future_work_not_conducted",
        "failures": failures,
    }
    output_path = ROOT / "experiments" / "post_submission_release_audit.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
