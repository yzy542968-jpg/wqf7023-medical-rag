"""Run the protocol-frozen V17 retrieval-only Calibration experiment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.qa.question_vectorizer import RadReStructQuestionVectorizer  # noqa: E402
from medical_rag.qa.radrestruct import iter_radrestruct_cases  # noqa: E402
from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy  # noqa: E402
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v10_reranker import FactAwareFeatureIndex  # noqa: E402
from medical_rag.similar_case.v11_qrel import (  # noqa: E402
    prepare_qrel_case,
    qrel_v2_profile_prepared,
)
from medical_rag.similar_case.v11_question_planner import plan_question  # noqa: E402
from medical_rag.similar_case.v17_question_conditioned import (  # noqa: E402
    answer_stratum,
    deterministic_top_ids,
    fixed_point_free_permutation,
    set_f1,
    summarize_proxy_rows,
    weighted_ranking,
)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_embeddings(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as payload:
        case_ids = [str(value) for value in payload["case_ids"]]
        matrix = np.asarray(payload["case_image_embeddings"], dtype=np.float32)
        signature = str(payload["signature"].item())
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.isfinite(matrix).all():
        raise ValueError("Invalid image embedding cache")
    matrix /= norms
    return dict(zip(case_ids, matrix, strict=True)), signature


def _intent_coverage(case: dict[str, Any], facts: tuple[str, ...], preferences: tuple[str, ...]) -> float:
    availability = {
        "findings": bool(str(case.get("findings") or "").strip()),
        "impression": bool(str(case.get("impression") or "").strip()),
        "radgraph": bool(facts),
        "comparison": bool(str(case.get("comparison") or "").strip()),
    }
    weights = (1.0, 0.5, 0.25)
    numerator = sum(weight * float(availability.get(name, False)) for name, weight in zip(preferences, weights))
    denominator = sum(weights[: len(preferences)])
    return numerator / denominator if denominator else 0.0


def _proxy_row(
    *,
    target_answers: frozenset[str],
    candidate_ids: list[str],
    question_id: int,
    answers_by_case: dict[str, dict[int, frozenset[str]]],
    top1_qrel_v2: float,
) -> dict[str, Any]:
    candidate_answers = [answers_by_case.get(case_id, {}).get(question_id) for case_id in candidate_ids]
    first = candidate_answers[0] if candidate_answers else None
    covered = [value is not None for value in candidate_answers]
    return {
        "stratum": answer_stratum(target_answers),
        "top1_exact": int(first == target_answers) if first is not None else 0,
        "top3_any_exact": int(any(value == target_answers for value in candidate_answers if value is not None)),
        "top1_option_f1": set_f1(target_answers, first or ()),
        "top1_covered": int(bool(first is not None)),
        "top3_any_covered": int(any(covered)),
        "top1_qrel_v2": float(top1_qrel_v2),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = _load_json(args.config)
    manifest = _load_json(args.manifest)
    if config["data_roles"]["sealed"] != "final_qa_test":
        raise ValueError("V17 configuration must explicitly seal Final-QA Test")

    raw_cases = {str(row["case_id"]): row for row in _read_jsonl(args.cases)}
    embeddings, embedding_signature = _load_embeddings(args.embeddings)
    radgraph_records = read_radgraph_case_records(args.radgraph)
    facts_by_case = {
        case_id: tuple(record.facts)
        for case_id, record in radgraph_records.items()
        if record.status == "ok"
    }

    hierarchy = RadReStructHierarchy(args.radrestruct_root)
    vectorizer = RadReStructQuestionVectorizer(hierarchy)
    rad_cases = {case.case_id: case for case in iter_radrestruct_cases(args.radrestruct_root)}
    answers_by_case: dict[str, dict[int, frozenset[str]]] = {}
    qids_by_case: dict[str, tuple[int, ...]] = {}
    for case_id, case in rad_cases.items():
        qids = vectorizer.question_ids(case.questions)
        qids_by_case[case_id] = qids
        answers_by_case[case_id] = {
            qid: frozenset(" ".join(value.lower().split()) for value in question.answers)
            for qid, question in zip(qids, case.questions, strict=True)
        }

    bank_cases = manifest["roles"]["train"]["cases"]
    calibration_cases = manifest["roles"]["calibration"]["cases"]
    bank_clusters = {str(row["case_id"]): str(row["cluster_id"]) for row in bank_cases}
    calibration_clusters = {str(row["case_id"]): str(row["cluster_id"]) for row in calibration_cases}
    bank_ids = sorted(
        case_id
        for case_id in bank_clusters
        if case_id in raw_cases and case_id in embeddings and case_id in facts_by_case and case_id in answers_by_case
    )
    calibration_ids = sorted(
        case_id
        for case_id in calibration_clusters
        if case_id in raw_cases and case_id in embeddings and case_id in rad_cases
    )
    if args.limit_cases is not None:
        calibration_ids = calibration_ids[: int(args.limit_cases)]
    if len(bank_ids) < 100 or not calibration_ids:
        raise RuntimeError("V17 lacks a valid historical bank or Calibration cases")
    if set(bank_clusters[case_id] for case_id in bank_ids) & set(calibration_clusters[case_id] for case_id in calibration_ids):
        raise RuntimeError("Historical bank overlaps Calibration duplicate clusters")

    fact_index = FactAwareFeatureIndex.build(bank_ids, raw_cases, facts_by_case)
    bank_matrix = np.stack([embeddings[case_id] for case_id in bank_ids])
    bank_index = {case_id: index for index, case_id in enumerate(bank_ids)}
    prepared_cases = {
        case_id: prepare_qrel_case(raw_cases[case_id], facts_by_case)
        for case_id in set(bank_ids) | set(calibration_ids)
    }
    qrel_cache: dict[tuple[str, str], float] = {}

    question_feature_cache: dict[str, np.ndarray] = {}
    indication_feature_cache: dict[str, np.ndarray] = {}
    intent_cache: dict[tuple[str, tuple[str, ...]], float] = {}
    rows: list[dict[str, Any]] = []
    recipe_names = [str(recipe["name"]) for recipe in config["retrieval"]["recipes"]]

    for target_case_id in calibration_ids:
        target_case = raw_cases[target_case_id]
        image_scores = embeddings[target_case_id] @ bank_matrix.T
        eligible = np.asarray(
            [bank_clusters[case_id] != calibration_clusters[target_case_id] for case_id in bank_ids],
            dtype=bool,
        )
        image_scores[~eligible] = -np.inf
        shortlist_indices = np.asarray(
            sorted(
                np.flatnonzero(eligible).tolist(),
                key=lambda index: (-float(image_scores[index]), bank_ids[index]),
            )[: int(config["retrieval"]["shortlist_k"])],
            dtype=int,
        )
        shortlist_ids = [bank_ids[index] for index in shortlist_indices]
        indication = " ".join(str(target_case.get("indication") or "").split())
        if indication not in indication_feature_cache:
            indication_feature_cache[indication] = fact_index.query_features(indication)[:, 0]

        rad_case = rad_cases[target_case_id]
        for question_index, (question_id, question) in enumerate(
            zip(qids_by_case[target_case_id], rad_case.questions, strict=True)
        ):
            question_text = question.question
            if question_text not in question_feature_cache:
                question_feature_cache[question_text] = fact_index.query_features(question_text)
            question_features = question_feature_cache[question_text]
            plan = plan_question(question_text, indication)
            coverage = []
            for case_id in shortlist_ids:
                cache_key = (case_id, plan.evidence_preferences)
                if cache_key not in intent_cache:
                    intent_cache[cache_key] = _intent_coverage(
                        raw_cases[case_id], facts_by_case.get(case_id, ()), plan.evidence_preferences
                    )
                coverage.append(intent_cache[cache_key])
            features = np.column_stack(
                [
                    image_scores[shortlist_indices],
                    question_features[shortlist_indices, 2],
                    question_features[shortlist_indices, 0],
                    indication_feature_cache[indication][shortlist_indices],
                    np.asarray(coverage, dtype=np.float32),
                ]
            )
            rankings: dict[str, list[str]] = {}
            for recipe in config["retrieval"]["recipes"]:
                ranked, _scores = weighted_ranking(shortlist_ids, features, recipe["weights"])
                rankings[str(recipe["name"])] = ranked[: int(config["retrieval"]["output_k"])]
            rows.append(
                {
                    "query_key": f"{target_case_id}|{question_index}",
                    "case_id": target_case_id,
                    "question_index": question_index,
                    "question_id": int(question_id),
                    "question": question_text,
                    "target_answers": sorted(answers_by_case[target_case_id][question_id]),
                    "stratum": answer_stratum(answers_by_case[target_case_id][question_id]),
                    "rankings": rankings,
                }
            )

    def qrel(target_id: str, candidate_id: str) -> float:
        key = (target_id, candidate_id)
        if key not in qrel_cache:
            qrel_cache[key] = float(
                qrel_v2_profile_prepared(prepared_cases[target_id], prepared_cases[candidate_id])["qrel_v2"]
            )
        return qrel_cache[key]

    recipe_proxy_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in recipe_names}
    for row in rows:
        target_answers = frozenset(row["target_answers"])
        for name in recipe_names:
            candidate_ids = row["rankings"][name]
            recipe_proxy_rows[name].append(
                _proxy_row(
                    target_answers=target_answers,
                    candidate_ids=candidate_ids,
                    question_id=int(row["question_id"]),
                    answers_by_case=answers_by_case,
                    top1_qrel_v2=qrel(row["case_id"], candidate_ids[0]),
                )
            )
    recipe_summaries = {name: summarize_proxy_rows(values) for name, values in recipe_proxy_rows.items()}
    tolerance = float(config["retrieval"]["tie_tolerance"])
    best_value = max(float(value["balanced_top1_qid_answer_agreement"]) for value in recipe_summaries.values())
    selected_recipe = next(
        name
        for name in recipe_names
        if best_value - float(recipe_summaries[name]["balanced_top1_qid_answer_agreement"]) <= tolerance
    )

    random_by_qid = {
        question_id: deterministic_top_ids(
            bank_ids,
            domain=config["controls"]["random_domain"],
            seed=int(config["seed"]),
            key=str(question_id),
            count=int(config["retrieval"]["output_k"]),
        )
        for question_id in sorted({int(row["question_id"]) for row in rows})
    }
    mismatch = fixed_point_free_permutation(
        [row["query_key"] for row in rows],
        domain=config["controls"]["mismatched_domain"],
        seed=int(config["seed"]),
    )
    by_key = {row["query_key"]: row for row in rows}
    control_proxy_rows: dict[str, list[dict[str, Any]]] = {"related": [], "random": [], "mismatched": []}
    for row in rows:
        arm_ids = {
            "related": row["rankings"][selected_recipe],
            "random": random_by_qid[int(row["question_id"])],
            "mismatched": by_key[mismatch[row["query_key"]]]["rankings"][selected_recipe],
        }
        row["selected_recipe"] = selected_recipe
        row["control_rankings"] = arm_ids
        target_answers = frozenset(row["target_answers"])
        for arm, candidate_ids in arm_ids.items():
            control_proxy_rows[arm].append(
                _proxy_row(
                    target_answers=target_answers,
                    candidate_ids=candidate_ids,
                    question_id=int(row["question_id"]),
                    answers_by_case=answers_by_case,
                    top1_qrel_v2=qrel(row["case_id"], candidate_ids[0]),
                )
            )
    control_summaries = {name: summarize_proxy_rows(values) for name, values in control_proxy_rows.items()}

    baseline = recipe_summaries["image_only"]
    selected = recipe_summaries[selected_recipe]
    positive_selected = selected["strata"].get("positive", {}).get("top1_exact", 0.0)
    positive_baseline = baseline["strata"].get("positive", {}).get("top1_exact", 0.0)
    go_checks = {
        "selected_recipe_is_nonbaseline": selected_recipe != "image_only",
        "balanced_top1_above_image_only": float(selected["balanced_top1_qid_answer_agreement"])
        > float(baseline["balanced_top1_qid_answer_agreement"]),
        "related_top3_above_random": float(control_summaries["related"]["top3_any_exact"])
        > float(control_summaries["random"]["top3_any_exact"]),
        "related_top3_above_mismatched": float(control_summaries["related"]["top3_any_exact"])
        > float(control_summaries["mismatched"]["top3_any_exact"]),
        "positive_top1_not_below_image_only": float(positive_selected) >= float(positive_baseline),
    }
    go = all(go_checks.values())
    summary = {
        "study": "V17 question-conditioned retrieval Calibration",
        "status": "retrieval_go" if go else "retrieval_no_go",
        "protocol_commit": "106f07d",
        "data_role": "final_qa_calibration",
        "test_accessed": False,
        "calibration_case_count": len(calibration_ids),
        "question_count": len(rows),
        "historical_bank_case_count": len(bank_ids),
        "historical_bank_manifest_case_count": len(bank_cases),
        "historical_bank_asset_exclusion_count": len(bank_cases) - len(bank_ids),
        "historical_bank_asset_exclusion_case_ids": sorted(
            set(str(row["case_id"]) for row in bank_cases) - set(bank_ids)
        ),
        "embedding_signature": embedding_signature,
        "selected_recipe": selected_recipe,
        "recipe_summaries": recipe_summaries,
        "matched_control_summaries": control_summaries,
        "go_checks": go_checks,
        "generation_authorized": go,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_boundary": (
            "Question-ID answer agreement and qrel-v2 are report-derived development proxies, "
            "not physician-adjudicated clinical similarity or diagnostic accuracy."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v17_retrieval_calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "v17_retrieval_calibration_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "config/v17_question_conditioned_retrieval.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/final_qa/final_qa_development_manifest.json")
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/processed/v10_medsiglip_embeddings.npz")
    parser.add_argument("--radgraph", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/v17_exploratory")
    parser.add_argument("--limit-cases", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
