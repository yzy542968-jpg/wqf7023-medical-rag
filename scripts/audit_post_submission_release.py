from __future__ import annotations

import hashlib
import json
import subprocess
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


def _canonical_json_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_git_json(commit: str, relative_path: str) -> dict[str, Any] | None:
    try:
        content = subprocess.check_output(
            ["git", "show", f"{commit}:{relative_path}"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return json.loads(content)


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
    hybrid_manifest = _read(
        ROOT
        / "experiments"
        / "post_submission_v23"
        / "preregistration_manifest.json"
    )
    hybrid_summary = _read(
        ROOT / "experiments" / "post_submission_v23" / "summary.json"
    )
    runtime_artifact = _read(
        ROOT
        / "experiments"
        / "post_submission_v23"
        / "generation_runtime_profile.json"
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
    hybrid_protocol = hybrid_summary.get("protocol", {})
    if hybrid_protocol.get("policy_id") != hybrid_manifest.get("policy_id"):
        failures.append("v2.3 policy ID differs from preregistration")
    if hybrid_protocol.get("test_or_transfer_tuning") is not False:
        failures.append("v2.3 no-test-tuning declaration is missing")
    if hybrid_protocol.get("policy_frozen_before_generation") is not True:
        failures.append("v2.3 policy-freeze declaration is missing")
    prereg_commit = str(hybrid_protocol.get("preregistration_git_commit", ""))
    committed_manifest = _read_git_json(
        prereg_commit,
        "experiments/post_submission_v23/preregistration_manifest.json",
    )
    if committed_manifest != hybrid_manifest:
        failures.append("v2.3 manifest does not match the preregistration commit")
    if hybrid_protocol.get(
        "preregistration_manifest_canonical_sha256"
    ) != _canonical_json_sha256(hybrid_manifest):
        failures.append("v2.3 preregistration manifest hash mismatch")
    hybrid_sets = hybrid_summary.get("evaluation_sets", {})
    original_hybrid = (
        hybrid_sets.get("original_test", {})
        .get("systems", {})
        .get("hybrid", {})
        .get("raw", {})
    )
    original_frozen = hard_summary["systems"]["closed_loop_agent_v2"]["test"]
    if original_hybrid != original_frozen:
        failures.append("v2.3 does not reproduce frozen v2.1 original-test metrics")
    transfer2 = hybrid_sets.get("reserved_wording_set_2", {})
    if transfer2.get("semantic_planner", {}).get("parse_failure_count") != 0:
        failures.append("v2.3 second transfer set contains planner parse failures")
    macro_ci = (
        transfer2.get("paired_case_bootstrap", {})
        .get("macro_f1_delta_hybrid_minus_lexical", {})
        .get("ci95", [None, None])
    )
    if macro_ci[0] is None or macro_ci[0] <= 0:
        failures.append("v2.3 transfer robustness gain is not positive at CI lower bound")
    runtime_profile = hybrid_summary.get("runtime_profile", {})
    if runtime_profile != runtime_artifact:
        failures.append("v2.3 embedded runtime profile differs from tracked artifact")
    cuda_profile = runtime_profile.get("cuda", {}) or {}
    if runtime_profile.get("records_generated") != hybrid_manifest.get("record_count"):
        failures.append("v2.3 runtime profile does not cover the frozen planner pack")
    if runtime_profile.get("device") != "cuda":
        failures.append("v2.3 runtime profile is not a CUDA measurement")
    peak_allocated = cuda_profile.get("peak_allocated_bytes")
    peak_reserved = cuda_profile.get("peak_reserved_bytes")
    total_memory = cuda_profile.get("total_memory_bytes")
    if not all(
        isinstance(value, (int, float)) and value > 0
        for value in (peak_allocated, peak_reserved, total_memory)
    ):
        failures.append("v2.3 CUDA peak-memory measurements are missing")
    elif not peak_allocated <= peak_reserved <= total_memory:
        failures.append("v2.3 CUDA memory measurements are inconsistent")
    if replication_summary.get("status") != "complete":
        failures.append("locked replication is not complete")
    if replication_summary.get("generation", {}).get("unique_qid_count") != replication[
        "question_count"
    ]:
        failures.append("replication generation count does not match cohort")
    if not replication_summary.get("no_replication_tuning"):
        failures.append("replication no-tuning declaration is missing")

    multimodal_config_path = ROOT / "config" / "multimodal_v42.json"
    multimodal_config = _read(multimodal_config_path)
    multimodal_manifest = _read(
        ROOT / "experiments" / "post_submission_v42" / "preregistration_manifest.json"
    )
    multimodal_development = _read(
        ROOT / "experiments" / "post_submission_v42" / "development_retrieval_summary.json"
    )
    multimodal_confirmation = _read(
        ROOT / "experiments" / "post_submission_v42" / "confirmation_retrieval_summary.json"
    )
    multimodal_statistics = _read(
        ROOT / "experiments" / "post_submission_v42" / "confirmation_statistics.json"
    )
    multimodal_runtime = _read(
        ROOT / "experiments" / "post_submission_v42" / "runtime_profile.json"
    )
    multimodal_source = _read(
        ROOT / "data" / "processed" / "openi_multimodal_source_manifest.json"
    )
    v42_prereg_commit = "5846649"
    v42_development_commit = "a0358c4"
    v42_confirmation_commit = "9bb6bf7"
    committed_v42_manifest = _read_git_json(
        v42_prereg_commit,
        "experiments/post_submission_v42/preregistration_manifest.json",
    )
    committed_v42_development = _read_git_json(
        v42_development_commit,
        "experiments/post_submission_v42/development_retrieval_summary.json",
    )
    committed_v42_confirmation = _read_git_json(
        v42_confirmation_commit,
        "experiments/post_submission_v42/confirmation_retrieval_summary.json",
    )
    if committed_v42_manifest != multimodal_manifest:
        failures.append("v4.2 manifest does not match preregistration commit")
    if committed_v42_development != multimodal_development:
        failures.append("v4.2 development policy does not match locked commit")
    if committed_v42_confirmation != multimodal_confirmation:
        failures.append("v4.2 confirmation result does not match one-shot commit")
    if multimodal_manifest.get("config_sha256") != _sha256(multimodal_config_path):
        failures.append("v4.2 config hash differs from preregistration manifest")
    if multimodal_manifest.get("confirmation_tuning") is not False:
        failures.append("v4.2 no-confirmation-tuning declaration is missing")
    if multimodal_manifest.get("confirmation_run_limit") != 1:
        failures.append("v4.2 confirmation run limit is not one")
    if not multimodal_development.get("development_gate", {}).get("passed"):
        failures.append("v4.2 locked development gate did not pass")
    if multimodal_confirmation.get("development_commit") != v42_development_commit:
        failures.append("v4.2 confirmation does not reference locked development commit")
    if multimodal_confirmation.get("split_case_count") != 120:
        failures.append("v4.2 confirmation case count is not 120")
    if multimodal_source.get("pairing", {}).get("matched_image_count") != 7466:
        failures.append("v4.2 official image manifest is incomplete")
    if multimodal_source.get("pairing", {}).get("missing_image_count") != 0:
        failures.append("v4.2 official image manifest has missing references")
    v42_report_mrr = multimodal_confirmation.get("metrics", {}).get(
        "report_only_bm25", {}
    ).get("mrr")
    v42_paired_mrr = multimodal_confirmation.get("metrics", {}).get(
        "paired_biovil_t_shortlist_reranker", {}
    ).get("mrr")
    v42_mrr_stats = multimodal_statistics.get("comparisons", {}).get("mrr", {})
    if not isinstance(v42_report_mrr, (int, float)) or not isinstance(
        v42_paired_mrr, (int, float)
    ):
        failures.append("v4.2 confirmation MRR values are missing")
    elif abs(
        (v42_paired_mrr - v42_report_mrr)
        - float(v42_mrr_stats.get("mean_difference", float("nan")))
    ) > 1e-12:
        failures.append("v4.2 MRR difference does not match paired statistics")
    if v42_mrr_stats.get("ci_low", 0.0) <= 0.0:
        failures.append("v4.2 primary MRR interval does not exclude zero")
    if multimodal_statistics.get("confirmation_result_commit") != v42_confirmation_commit:
        failures.append("v4.2 statistics do not reference locked confirmation result")
    if multimodal_statistics.get("resamples") != 5000:
        failures.append("v4.2 paired bootstrap does not use 5000 resamples")
    if multimodal_runtime.get("confirmation_result_commit") != v42_confirmation_commit:
        failures.append("v4.2 runtime profile does not reference locked confirmation result")
    if multimodal_runtime.get("latency", {}).get(
        "warm_paired_request_estimated_mean_ms", 0.0
    ) <= 0.0:
        failures.append("v4.2 warm request latency is missing")
    if multimodal_runtime.get("peak_cuda_memory_mib", 0.0) <= 0.0:
        failures.append("v4.2 CUDA memory profile is missing")

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
        "hybrid_planner": {
            "policy_id": hybrid_protocol.get("policy_id"),
            "preregistration_git_commit": prereg_commit,
            "manifest_matches_preregistration_commit": committed_manifest
            == hybrid_manifest,
            "original_macro_f1": original_hybrid.get("macro_f1"),
            "transfer2_lexical_macro_f1": transfer2.get("systems", {})
            .get("lexical", {})
            .get("raw", {})
            .get("macro_f1"),
            "transfer2_hybrid_macro_f1": transfer2.get("systems", {})
            .get("hybrid", {})
            .get("raw", {})
            .get("macro_f1"),
            "transfer2_macro_f1_delta_ci95": macro_ci,
            "transfer2_false_answer_rate_delta_ci95": transfer2.get(
                "paired_case_bootstrap", {}
            )
            .get("false_answer_rate_delta_hybrid_minus_lexical", {})
            .get("ci95"),
            "runtime": {
                "model_load_seconds": runtime_profile.get("timing_seconds", {}).get(
                    "model_and_tokenizer_load"
                ),
                "generation_seconds": runtime_profile.get("timing_seconds", {}).get(
                    "generation"
                ),
                "peak_allocated_mib": cuda_profile.get("peak_allocated_mib"),
                "peak_reserved_mib": cuda_profile.get("peak_reserved_mib"),
            },
        },
        "multimodal_v42": {
            "preregistration_commit": v42_prereg_commit,
            "development_commit": v42_development_commit,
            "confirmation_commit": v42_confirmation_commit,
            "manifest_matches_preregistration_commit": committed_v42_manifest
            == multimodal_manifest,
            "development_matches_locked_commit": committed_v42_development
            == multimodal_development,
            "confirmation_matches_one_shot_commit": committed_v42_confirmation
            == multimodal_confirmation,
            "matched_image_count": multimodal_source.get("pairing", {}).get(
                "matched_image_count"
            ),
            "confirmation_case_count": multimodal_confirmation.get(
                "split_case_count"
            ),
            "report_only_mrr": v42_report_mrr,
            "paired_mrr": v42_paired_mrr,
            "mrr_difference_ci95": [
                v42_mrr_stats.get("ci_low"),
                v42_mrr_stats.get("ci_high"),
            ],
            "warm_request_mean_ms": multimodal_runtime.get("latency", {}).get(
                "warm_paired_request_estimated_mean_ms"
            ),
            "peak_cuda_memory_mib": multimodal_runtime.get(
                "peak_cuda_memory_mib"
            ),
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
