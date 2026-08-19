from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_v6_confirmation_retrieval import (  # noqa: E402
    DEFAULT_MEDSIGLIP_CACHE,
    embedding_cache_signature,
    image_score_maps_max,
    load_cache,
)
from run_v6_development_multimodal_retrieval import (  # noqa: E402
    build_bm25_inputs,
    fused_rankings,
    image_file_lookup,
    resolve_case_images,
)

from medical_rag.multimodal.medsiglip import (  # noqa: E402
    DEFAULT_MODEL as MEDSIGLIP_MODEL,
    DEFAULT_REVISION as MEDSIGLIP_REVISION,
)
from medical_rag.multimodal.v6_chunking import build_report_chunks  # noqa: E402
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl  # noqa: E402


PROTOCOL_COMMIT = "eee7405"
RETRIEVAL_OUTCOME_COMMIT = "c6442c9"
VERIFIED_QA_OUTCOME_COMMIT = "3ae127f"

DEFAULT_CONFIG = ROOT / "config" / "v6_confirmation.json"
DEFAULT_COHORT = ROOT / "data" / "splits" / "v6" / "v6_confirmation_cohort.json"
DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_IMAGE_ROOT = ROOT / "data" / "raw" / "openi_official_images"
DEFAULT_RETRIEVAL_ROWS = (
    ROOT / "experiments" / "post_submission_v6" / "confirmation_retrieval_rows.jsonl"
)
DEFAULT_RETRIEVAL_SUMMARY = (
    ROOT / "experiments" / "post_submission_v6" / "confirmation_retrieval_summary.json"
)
DEFAULT_VERIFIED_ROWS = (
    ROOT / "experiments" / "post_submission_v6" / "confirmation_qa_factorial_verified_rows.jsonl"
)
DEFAULT_VERIFIED_SUMMARY = (
    ROOT
    / "experiments"
    / "post_submission_v6"
    / "confirmation_qa_factorial_verified_summary.json"
)
DEFAULT_OUTPUT = (
    ROOT / "experiments" / "post_submission_v6" / "confirmation_statistical_analysis.json"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def commit_exists(commit: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def reciprocal_rank(ranking: Sequence[str], target_case_id: str) -> float:
    try:
        return 1.0 / (list(ranking).index(target_case_id) + 1)
    except ValueError:
        return 0.0


def paired_case_bootstrap(
    differences_by_case: Mapping[str, float],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    case_ids = sorted(differences_by_case)
    if not case_ids:
        raise ValueError("At least one case is required for paired bootstrap.")
    values = np.asarray([differences_by_case[case_id] for case_id in case_ids])
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(values), size=(resamples, len(values)))
    distribution = values[sample_indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(distribution, [alpha / 2, 1 - alpha / 2])
    return {
        "case_count": len(case_ids),
        "point_difference": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": confidence_level,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "ci_method": "percentile_case_grouped_paired_bootstrap",
    }


def per_case_system_metric(
    rows: Iterable[Mapping[str, Any]], metric: str
) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["system"])][str(row["case_id"])].append(float(row[metric]))
    return {
        system: {case_id: mean(values) for case_id, values in cases.items()}
        for system, cases in grouped.items()
    }


def paired_differences(
    per_case: Mapping[str, Mapping[str, float]],
    treatment: str,
    baseline: str,
    case_ids: Iterable[str],
) -> dict[str, float]:
    return {
        case_id: float(per_case[treatment][case_id] - per_case[baseline][case_id])
        for case_id in sorted(set(case_ids))
    }


def reconstruct_primary_rankings(
    *,
    config_path: Path,
    cohort_path: Path,
    cases_path: Path,
    image_root: Path,
    medsiglip_cache_path: Path,
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, Any]]:
    from transformers import AutoProcessor

    config = read_json(config_path)
    cohort = read_json(cohort_path)
    cases = {str(case["case_id"]): case for case in load_cases_jsonl(cases_path)}
    candidate_ids = [str(value) for value in cohort["case_ids"]]
    target_ids = [str(value) for value in cohort["target_case_ids"]]
    questions = list(cohort["questions"])

    processor = AutoProcessor.from_pretrained(
        MEDSIGLIP_MODEL,
        revision=MEDSIGLIP_REVISION,
        cache_dir=str(ROOT / ".hf_cache"),
        local_files_only=True,
        use_fast=False,
    )
    chunks = [
        chunk
        for case_id in candidate_ids
        for chunk in build_report_chunks(
            cases[case_id],
            processor.tokenizer,
            max_tokens=int(
                config["multimodal_retrieval"]["primary_encoder"]["max_text_tokens"]
            ),
        )
    ]
    chunk_ids = [str(row["chunk_id"]) for row in chunks]
    chunk_case_ids = [str(row["case_id"]) for row in chunks]
    chunk_texts = [str(row["text"]) for row in chunks]
    case_images = resolve_case_images(
        candidate_ids, cases, image_file_lookup(image_root)
    )
    view_paths = [path for case_id in candidate_ids for path in case_images[case_id]]
    signature = embedding_cache_signature(
        encoder=MEDSIGLIP_MODEL,
        revision=MEDSIGLIP_REVISION,
        config=config_path,
        cohort=cohort_path,
        cases=cases_path,
        candidate_ids=candidate_ids,
        chunk_ids=chunk_ids,
        chunk_texts=chunk_texts,
        view_paths=view_paths,
    )
    cached = load_cache(
        medsiglip_cache_path,
        signature,
        ["chunks", "images"],
    )
    if cached is None:
        raise RuntimeError("The frozen MedSigLIP embedding cache signature did not match.")
    arrays, _ = cached
    text_rankings, text_scores = build_bm25_inputs(questions, candidate_ids, cases)
    image_maps = image_score_maps_max(
        arrays["images"],
        arrays["chunks"],
        chunk_case_ids,
        candidate_ids,
        target_ids,
    )
    medsiglip_rankings = fused_rankings(
        questions,
        text_rankings,
        text_scores,
        image_maps,
        shortlist_size=int(config["multimodal_retrieval"]["shortlist_size"]),
        text_weight=float(config["multimodal_retrieval"]["text_weight"]),
    )
    return text_rankings, medsiglip_rankings, {
        "embedding_cache": str(medsiglip_cache_path.relative_to(ROOT)).replace("\\", "/"),
        "embedding_cache_sha256": file_sha256(medsiglip_cache_path),
        "embedding_cache_signature": signature,
        "chunk_count": len(chunks),
        "view_count": len(view_paths),
    }


def differences_for_rankings(
    questions: Sequence[Mapping[str, Any]],
    baseline_rankings: Mapping[str, Sequence[str]],
    treatment_rankings: Mapping[str, Sequence[str]],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    baseline_by_case: dict[str, list[float]] = defaultdict(list)
    treatment_by_case: dict[str, list[float]] = defaultdict(list)
    for question in questions:
        qid = str(question["qid"])
        case_id = str(question["case_id"])
        baseline_by_case[case_id].append(
            reciprocal_rank(baseline_rankings[qid], case_id)
        )
        treatment_by_case[case_id].append(
            reciprocal_rank(treatment_rankings[qid], case_id)
        )
    baseline = {case_id: mean(values) for case_id, values in baseline_by_case.items()}
    treatment = {
        case_id: mean(values) for case_id, values in treatment_by_case.items()
    }
    differences = {
        case_id: treatment[case_id] - baseline[case_id] for case_id in baseline
    }
    return baseline, treatment, differences


def analyze_pair(
    per_case: Mapping[str, Mapping[str, float]],
    *,
    treatment: str,
    baseline: str,
    case_ids: Iterable[str],
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    selected = sorted(set(case_ids))
    differences = paired_differences(per_case, treatment, baseline, selected)
    result = paired_case_bootstrap(
        differences,
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
    )
    result["baseline_mean"] = mean(per_case[baseline][case_id] for case_id in selected)
    result["treatment_mean"] = mean(
        per_case[treatment][case_id] for case_id in selected
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen V6 confirmation case-grouped statistical analysis."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--retrieval-rows", type=Path, default=DEFAULT_RETRIEVAL_ROWS)
    parser.add_argument("--retrieval-summary", type=Path, default=DEFAULT_RETRIEVAL_SUMMARY)
    parser.add_argument("--verified-rows", type=Path, default=DEFAULT_VERIFIED_ROWS)
    parser.add_argument("--verified-summary", type=Path, default=DEFAULT_VERIFIED_SUMMARY)
    parser.add_argument("--medsiglip-cache", type=Path, default=DEFAULT_MEDSIGLIP_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))
    if args.output.exists():
        raise RuntimeError("Formal V6 statistical analysis already exists; refusing rerun.")
    for commit in (
        PROTOCOL_COMMIT,
        RETRIEVAL_OUTCOME_COMMIT,
        VERIFIED_QA_OUTCOME_COMMIT,
    ):
        if not commit_exists(commit):
            raise RuntimeError(f"Required frozen commit is unavailable: {commit}")

    config = read_json(args.config)
    cohort = read_json(args.cohort)
    retrieval_summary = read_json(args.retrieval_summary)
    verified_summary = read_json(args.verified_summary)
    retrieval_rows = read_jsonl(args.retrieval_rows)
    verified_rows = read_jsonl(args.verified_rows)
    if retrieval_summary["outputs"]["rows_sha256"] != file_sha256(args.retrieval_rows):
        raise RuntimeError("Retrieval rows no longer match their frozen summary.")
    if verified_summary["outputs"]["verified_rows_sha256"] != file_sha256(
        args.verified_rows
    ):
        raise RuntimeError("Verified QA rows no longer match their frozen summary.")
    if len(retrieval_rows) != 1440 or len(verified_rows) != 1440:
        raise RuntimeError("A frozen V6 confirmation row matrix is incomplete.")

    statistics = config["statistics"]
    resamples = int(statistics["bootstrap_resamples"])
    seed = int(statistics["bootstrap_seed"])
    confidence_level = float(statistics["confidence_level"])
    questions = list(cohort["questions"])
    target_ids = [str(value) for value in cohort["target_case_ids"]]
    target_classes = {
        str(row["case_id"]): str(row["report_index_class"])
        for row in cohort["cases"]
        if row["role"] == "target"
    }
    normal_ids = [
        case_id
        for case_id in target_ids
        if target_classes[case_id] == "report_indexed_normal"
    ]
    abnormal_ids = [
        case_id
        for case_id in target_ids
        if target_classes[case_id] == "report_indexed_abnormal"
    ]
    if len(normal_ids) != 86 or len(abnormal_ids) != 34:
        raise RuntimeError("The frozen V6 target spectrum composition changed.")

    bm25_rankings, medsiglip_rankings, ranking_audit = reconstruct_primary_rankings(
        config_path=args.config,
        cohort_path=args.cohort,
        cases_path=args.cases,
        image_root=args.image_root,
        medsiglip_cache_path=args.medsiglip_cache,
    )
    bm25_rr, medsiglip_rr, retrieval_differences = differences_for_rankings(
        questions, bm25_rankings, medsiglip_rankings
    )
    reconstructed_bm25_mrr = mean(bm25_rr.values())
    reconstructed_medsiglip_mrr = mean(medsiglip_rr.values())
    frozen_bm25_mrr = float(retrieval_summary["metrics"]["bm25"]["mrr"])
    frozen_medsiglip_mrr = float(
        retrieval_summary["metrics"]["medsiglip_max_chunk_reranker"]["mrr"]
    )
    if not np.isclose(reconstructed_bm25_mrr, frozen_bm25_mrr, atol=1e-12):
        raise RuntimeError("Reconstructed BM25 MRR does not match the frozen outcome.")
    if not np.isclose(
        reconstructed_medsiglip_mrr, frozen_medsiglip_mrr, atol=1e-12
    ):
        raise RuntimeError("Reconstructed MedSigLIP MRR does not match the frozen outcome.")

    primary_retrieval = paired_case_bootstrap(
        retrieval_differences,
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
    )
    primary_retrieval.update(
        {
            "baseline": "bm25",
            "treatment": "medsiglip_max_chunk_reranker",
            "baseline_mrr": reconstructed_bm25_mrr,
            "treatment_mrr": reconstructed_medsiglip_mrr,
            "criterion": "ci_lower_gt_zero",
            "criterion_passed": primary_retrieval["ci_lower"] > 0,
        }
    )

    per_case_verified = per_case_system_metric(verified_rows, "final_token_f1")
    qa_pairs = {
        "qwen2_5": ("medsiglip_qwen2_5", "bm25_qwen2_5"),
        "medgemma_1_5": ("medsiglip_medgemma_1_5", "bm25_medgemma_1_5"),
    }
    primary_qa = {}
    for generator, (treatment, baseline) in qa_pairs.items():
        result = analyze_pair(
            per_case_verified,
            treatment=treatment,
            baseline=baseline,
            case_ids=target_ids,
            resamples=resamples,
            seed=seed,
            confidence_level=confidence_level,
        )
        result.update(
            {
                "metric": "verified_token_f1",
                "baseline_system": baseline,
                "treatment_system": treatment,
                "criterion": "point_difference_gt_zero",
                "criterion_passed": result["point_difference"] > 0,
                "ci_excludes_zero": result["ci_lower"] > 0
                or result["ci_upper"] < 0,
            }
        )
        primary_qa[generator] = result

    qwen_differences = paired_differences(
        per_case_verified, "medsiglip_qwen2_5", "bm25_qwen2_5", target_ids
    )
    medgemma_differences = paired_differences(
        per_case_verified,
        "medsiglip_medgemma_1_5",
        "bm25_medgemma_1_5",
        target_ids,
    )
    did = paired_case_bootstrap(
        {
            case_id: medgemma_differences[case_id] - qwen_differences[case_id]
            for case_id in target_ids
        },
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
    )
    did["definition"] = "delta_medgemma_minus_delta_qwen"
    did["confirmatory_threshold"] = None

    secondary_metrics = {}
    for metric in (
        "draft_token_f1",
        "support_rate",
        "agent_abstained",
        "revised",
    ):
        per_case = per_case_system_metric(verified_rows, metric)
        secondary_metrics[metric] = {
            generator: analyze_pair(
                per_case,
                treatment=treatment,
                baseline=baseline,
                case_ids=target_ids,
                resamples=resamples,
                seed=seed,
                confidence_level=confidence_level,
            )
            for generator, (treatment, baseline) in qa_pairs.items()
        }

    subgroup_results = {}
    for subgroup, subgroup_ids in {
        "report_indexed_normal": normal_ids,
        "report_indexed_abnormal": abnormal_ids,
    }.items():
        subgroup_retrieval = paired_case_bootstrap(
            {case_id: retrieval_differences[case_id] for case_id in subgroup_ids},
            resamples=resamples,
            seed=seed,
            confidence_level=confidence_level,
        )
        subgroup_results[subgroup] = {
            "case_count": len(subgroup_ids),
            "retrieval_mrr": subgroup_retrieval,
            "verified_token_f1": {
                generator: analyze_pair(
                    per_case_verified,
                    treatment=treatment,
                    baseline=baseline,
                    case_ids=subgroup_ids,
                    resamples=resamples,
                    seed=seed,
                    confidence_level=confidence_level,
                )
                for generator, (treatment, baseline) in qa_pairs.items()
            },
        }

    shuffled = retrieval_summary["random_image_control"]
    result = {
        "experiment": "V6 model-modernized confirmation statistical analysis",
        "status": "formal_confirmation_statistics_frozen",
        "protocol_commit": PROTOCOL_COMMIT,
        "retrieval_outcome_commit": RETRIEVAL_OUTCOME_COMMIT,
        "verified_qa_outcome_commit": VERIFIED_QA_OUTCOME_COMMIT,
        "analysis_implementation_sha256": file_sha256(Path(__file__)),
        "method": {
            "unit": "case_id",
            "paired": True,
            "bootstrap_resamples": resamples,
            "bootstrap_seed": seed,
            "confidence_level": confidence_level,
            "percentile_quantile_method": "NumPy default linear interpolation",
            "patient_level_independence_verified": False,
        },
        "input_artifacts": {
            "config_sha256": file_sha256(args.config),
            "cohort_sha256": file_sha256(args.cohort),
            "retrieval_rows_sha256": file_sha256(args.retrieval_rows),
            "retrieval_summary_sha256": file_sha256(args.retrieval_summary),
            "verified_rows_sha256": file_sha256(args.verified_rows),
            "verified_summary_sha256": file_sha256(args.verified_summary),
            "ranking_reconstruction": ranking_audit,
        },
        "integrity": {
            "target_case_count": len(target_ids),
            "question_count": len(questions),
            "normal_target_count": len(normal_ids),
            "abnormal_target_count": len(abnormal_ids),
            "reconstructed_bm25_mrr_matches_frozen": True,
            "reconstructed_medsiglip_mrr_matches_frozen": True,
        },
        "primary_retrieval": primary_retrieval,
        "alignment_specificity": {
            "correctly_aligned_medsiglip_mrr": frozen_medsiglip_mrr,
            "shuffled_control_count": int(shuffled["count"]),
            "shuffled_mrr_exceedance_count": int(shuffled["mrr_exceedance_count"]),
            "plus_one_monte_carlo_p_mrr": float(
                shuffled["plus_one_monte_carlo_p_mrr"]
            ),
            "criterion": "plus_one_p_le_0_05",
            "criterion_passed": float(shuffled["plus_one_monte_carlo_p_mrr"])
            <= 0.05,
        },
        "primary_qa": primary_qa,
        "difference_in_differences": did,
        "secondary_qa_metrics": secondary_metrics,
        "predefined_subgroup_sensitivity": subgroup_results,
        "claim_boundary": (
            "Case-ID-grouped same-source automated evaluation; not patient-level "
            "independence, external validation, clinical correctness, or safety."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
