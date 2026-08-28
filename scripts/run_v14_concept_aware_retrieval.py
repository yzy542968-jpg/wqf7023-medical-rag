"""Run the leakage-controlled V14 concept-aware retrieval development study."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from lightgbm import LGBMRanker, early_stopping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from develop_v13_target_concepts import fit_linear_heads, label_cases  # noqa: E402
from evaluate_v10_pathology_utility import resolve_checkpoint  # noqa: E402
from evaluate_v13_target_concepts import predict  # noqa: E402
from medical_rag.evaluation.chexbert_pathology import CHEXBERT_LABELS  # noqa: E402
from medical_rag.evaluation.target_concepts import (  # noqa: E402
    case_id_fingerprint,
    logistic_probabilities,
)
from medical_rag.retrieval.concept_reranking import (  # noqa: E402
    FEATURE_NAMES,
    append_concept_features,
    cluster_fold_assignments,
)
from medical_rag.retrieval.medcpt_retriever import encode_queries  # noqa: E402
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v10_runtime import FrozenR5Runtime, QUESTIONS  # noqa: E402
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from medical_rag.similar_case.v11_qrel import (  # noqa: E402
    prepare_qrel_case,
    qrel_v2_profile_prepared,
)
from run_v12_retrieval_pilot import build_retrieval_state, ndcg, spectrum  # noqa: E402

warnings.filterwarnings("ignore", message="Found 'eval_at' in params")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="lightgbm")

PROTOCOL_COMMIT = "e80531f"
FACT_WEIGHTS = {
    "lesion_type": 0.25,
    "anatomy": 0.20,
    "severity": 0.10,
    "polarity": 0.15,
    "uncertainty": 0.10,
}
RANKER_CONFIGS = {
    "default": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 40,
        "reg_lambda": 1.0,
    },
    "deeper": {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 40,
        "reg_lambda": 1.0,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def case_to_cluster(split: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for row in split["clusters"]:
        cluster_id = str(row["cluster_id"])
        for case_id in row["case_ids"]:
            canonical = str(case_id)
            if canonical in mapping:
                raise RuntimeError(f"Case occurs in multiple clusters: {canonical}")
            mapping[canonical] = cluster_id
    return mapping


def fact_only_score(profile: dict[str, Any]) -> float:
    available = profile["component_available"]
    denominator = sum(weight for name, weight in FACT_WEIGHTS.items() if available[name])
    if denominator <= 0.0:
        return 0.0
    return float(
        sum(FACT_WEIGHTS[name] * float(profile[name]) for name in FACT_WEIGHTS if available[name])
        / denominator
    )


def qrel_profiles(
    query_id: str,
    candidate_ids: Sequence[str],
    prepared_by_case: dict[str, Any],
) -> dict[str, dict[str, float]]:
    output = {"combined": {}, "label_only": {}, "fact_only": {}}
    for candidate_id in candidate_ids:
        if candidate_id == query_id:
            continue
        profile = qrel_v2_profile_prepared(
            prepared_by_case[query_id], prepared_by_case[candidate_id]
        )
        output["combined"][candidate_id] = float(profile["qrel_v2"])
        output["label_only"][candidate_id] = float(profile["report_label"])
        output["fact_only"][candidate_id] = fact_only_score(profile)
    return output


def oof_concept_probabilities(
    train_ids: Sequence[str],
    embeddings_by_id: dict[str, np.ndarray],
    labels_by_id: dict[str, np.ndarray],
    assignments: dict[str, int],
    *,
    cache_path: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    ordered = list(map(str, train_ids))
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as encoded:
            cached_ids = list(map(str, encoded["case_ids"].tolist()))
            cached_folds = np.asarray(encoded["folds"], dtype=np.int8)
            probabilities = np.asarray(encoded["probabilities"], dtype=np.float64)
        expected_folds = np.asarray([assignments[case_id] for case_id in ordered], dtype=np.int8)
        if cached_ids != ordered or not np.array_equal(cached_folds, expected_folds):
            raise RuntimeError("V14 OOF cache differs from current cases or fold assignments")
        if probabilities.shape != (len(ordered), len(CHEXBERT_LABELS)):
            raise RuntimeError("Invalid V14 OOF cache shape")
        return dict(zip(ordered, probabilities, strict=True)), [
            {"fold": fold, "held_out": int(np.count_nonzero(expected_folds == fold)), "cached": True}
            for fold in range(5)
        ]

    x = np.stack([embeddings_by_id[case_id] for case_id in ordered])
    y = np.stack([labels_by_id[case_id] for case_id in ordered])
    folds = np.asarray([assignments[case_id] for case_id in ordered], dtype=np.int8)
    probabilities = np.full((len(ordered), len(CHEXBERT_LABELS)), np.nan, dtype=np.float64)
    records = []
    for fold in range(5):
        held_out = folds == fold
        fit = ~held_out
        if not held_out.any() or not fit.any():
            raise RuntimeError(f"Invalid V14 OOF fold {fold}")
        coefficients, intercepts = fit_linear_heads(x[fit], y[fit], c_value=1.0)
        probabilities[held_out] = logistic_probabilities(
            x[held_out], coefficients, intercepts
        )
        records.append(
            {
                "fold": fold,
                "fit": int(fit.sum()),
                "held_out": int(held_out.sum()),
                "held_out_case_ids_sha256": case_id_fingerprint(
                    [ordered[index] for index in np.flatnonzero(held_out)]
                ),
                "cached": False,
            }
        )
        print(f"oof_fold={fold} held_out={int(held_out.sum())}", flush=True)
    if not np.isfinite(probabilities).all():
        raise RuntimeError("OOF concept probabilities are incomplete")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        case_ids=np.asarray(ordered),
        folds=folds,
        probabilities=probabilities.astype(np.float32),
    )
    return dict(zip(ordered, probabilities, strict=True)), records


def fit_ranker(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    groups_fit: Sequence[int],
    x_internal: np.ndarray,
    y_internal: np.ndarray,
    groups_internal: Sequence[int],
    *,
    config_name: str,
) -> LGBMRanker:
    config = RANKER_CONFIGS[config_name]
    ranker = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=[10],
        **config,
        random_state=2026,
        verbosity=-1,
    )
    ranker.fit(
        x_fit,
        np.clip(np.rint(y_fit * 10.0), 0, 10).astype(np.int32),
        group=list(groups_fit),
        eval_set=[
            (
                x_internal,
                np.clip(np.rint(y_internal * 10.0), 0, 10).astype(np.int32),
            )
        ],
        eval_group=[list(groups_internal)],
        callbacks=[early_stopping(25, verbose=False)],
    )
    return ranker


def case_grouped_bootstrap(
    rows: Sequence[dict[str, Any]],
    metric: str,
    *,
    iterations: int = 10000,
    seed: int = 7146,
) -> dict[str, float]:
    by_case: dict[str, list[float]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(
            float(row["concept_23"][metric]) - float(row["base_17"][metric])
        )
    values = np.asarray([mean(by_case[case_id]) for case_id in sorted(by_case)])
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        draws[index] = float(rng.choice(values, len(values), replace=True).mean())
    return {
        "difference": float(values.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "case_count": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--radgraph", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--roles", type=Path, default=ROOT / "data/splits/v10/v10_reranker_roles.json")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/processed/v10_medsiglip_embeddings.npz")
    parser.add_argument("--medcpt", type=Path, default=ROOT / "data/processed/openi_medcpt_full.npz")
    parser.add_argument("--checkpoints", type=Path, default=ROOT / "experiments/v10_publication/reranker_checkpoints")
    parser.add_argument("--v13-decision", type=Path, default=ROOT / "data/splits/v13/v13_target_concept_decision.json")
    parser.add_argument("--train-label-cache", type=Path, default=ROOT / "experiments/v13_target_concept/v13_train_calibration_chexbert_cache.json")
    parser.add_argument("--validation-label-cache", type=Path, default=ROOT / "experiments/v13_target_concept/v13_validation_chexbert_cache.json")
    parser.add_argument("--oof-cache", type=Path, default=ROOT / "experiments/v14_concept_retrieval/v14_oof_concepts.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "data/splits/v14/v14_concept_aware_retrieval_summary.json")
    parser.add_argument("--rows", type=Path, default=ROOT / "experiments/v14_concept_retrieval/v14_retrieval_rows.jsonl")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "experiments/v14_concept_retrieval/models")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--ranker-config", choices=tuple(RANKER_CONFIGS), default="default")
    args = parser.parse_args()

    started = time.perf_counter()
    raw_rows = [json.loads(line) for line in args.cases.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_cases = {str(row["case_id"]): row for row in raw_rows}
    formal = {
        case.study_id: case
        for case in read_openi_paired_cases(
            args.cases, source_unique_patient=True, radgraph_path=args.radgraph
        )
    }
    radgraph = read_radgraph_case_records(args.radgraph)
    split = read_json(args.split)
    source_partition_ids = {
        name: [str(case_id) for case_id in split["partitions"][name]["case_ids"]]
        for name in ("train", "calibration", "validation")
    }
    if any(set(source_partition_ids[left]) & set(source_partition_ids[right]) for left, right in (("train", "calibration"), ("train", "validation"), ("calibration", "validation"))):
        raise RuntimeError("V14 source partitions overlap")
    exclusions = {
        name: [
            case_id
            for case_id in source_partition_ids[name]
            if case_id not in formal
            or case_id not in radgraph
            or radgraph[case_id].status != "ok"
        ]
        for name in source_partition_ids
    }
    partition_ids = {
        name: [case_id for case_id in source_partition_ids[name] if case_id not in set(exclusions[name])]
        for name in source_partition_ids
    }
    train_source_ids = source_partition_ids["train"]
    train_ids = partition_ids["train"]
    calibration_ids = partition_ids["calibration"]
    validation_ids = partition_ids["validation"]
    all_ids = train_ids + calibration_ids + validation_ids
    facts_by_case = {case_id: tuple(radgraph[case_id].facts) for case_id in all_ids}
    prepared_by_case = {
        case_id: prepare_qrel_case(raw_cases[case_id], facts_by_case)
        for case_id in all_ids
    }

    with np.load(args.embeddings, allow_pickle=False) as encoded:
        image_ids = list(map(str, encoded["case_ids"].tolist()))
        report_ids = list(map(str, encoded["report_ids"].tolist()))
        image_matrix = np.asarray(encoded["case_image_embeddings"], dtype=np.float32)
        report_matrix = np.asarray(encoded["report_embeddings"], dtype=np.float32)
    image_by_id = dict(zip(image_ids, image_matrix, strict=True))
    report_by_id = dict(zip(report_ids, report_matrix, strict=True))
    with np.load(args.medcpt, allow_pickle=False) as encoded:
        medcpt_ids = list(map(str, encoded["case_ids"].tolist()))
        medcpt_matrix = np.asarray(encoded["embeddings"], dtype=np.float32)
    medcpt_by_id = dict(zip(medcpt_ids, medcpt_matrix, strict=True))
    train_medcpt = np.stack([medcpt_by_id[case_id] for case_id in train_ids])

    chexbert_checkpoint = resolve_checkpoint()
    train_and_calibration = [
        raw_cases[case_id]
        for case_id in train_source_ids + source_partition_ids["calibration"]
    ]
    train_calibration_labels = label_cases(
        train_and_calibration,
        cache_path=args.train_label_cache,
        checkpoint_hash=file_sha256(chexbert_checkpoint),
        device=args.device,
        batch_size=128,
    )
    train_label_matrix = train_calibration_labels[: len(train_source_ids)]
    train_labels_by_id = dict(zip(train_source_ids, train_label_matrix, strict=True))
    candidate_labels_by_id = {case_id: train_labels_by_id[case_id] for case_id in train_ids}

    cluster_mapping = case_to_cluster(split)
    fold_assignments = cluster_fold_assignments(train_source_ids, cluster_mapping)
    oof_by_id, oof_records = oof_concept_probabilities(
        train_source_ids,
        image_by_id,
        train_labels_by_id,
        fold_assignments,
        cache_path=args.oof_cache,
    )

    decision = read_json(args.v13_decision)
    checkpoint_path = ROOT / decision["selected_checkpoint"]["path"]
    if file_sha256(checkpoint_path) != decision["selected_checkpoint"]["sha256"]:
        raise RuntimeError("Frozen V13 checkpoint hash differs")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    development_x = np.stack([image_by_id[case_id] for case_id in calibration_ids + validation_ids])
    development_probabilities, _, model_type = predict(
        checkpoint_path, development_x, device=device
    )
    query_probabilities = {
        **oof_by_id,
        **dict(
            zip(
                calibration_ids + validation_ids,
                development_probabilities,
                strict=True,
            )
        ),
    }

    roles = read_json(args.roles)
    fit_ids = [case_id for case_id in roles["roles"]["pairwise_fit"]["case_ids"] if case_id in train_ids]
    internal_ids = [case_id for case_id in roles["roles"]["internal_early_stop"]["case_ids"] if case_id in train_ids]
    query_ids = fit_ids + internal_ids + calibration_ids + validation_ids
    query_texts = [
        "\n".join(part for part in (formal[case_id].indication, QUESTIONS[question_type]) if part)
        for case_id in query_ids
        for question_type in QUESTIONS
    ]
    encoded_queries = encode_queries(
        query_texts, batch_size=32, device=args.device, local_files_only=True
    )
    query_embedding = {
        (case_id, question_type): encoded_queries[index * len(QUESTIONS) + question_index]
        for index, case_id in enumerate(query_ids)
        for question_index, question_type in enumerate(QUESTIONS)
    }
    del encoded_queries
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    checkpoint_states = [
        torch.load(args.checkpoints / f"r5_seed_{seed}.pt", map_location="cpu", weights_only=True)
        for seed in (7041, 7042, 7043, 7044, 7045)
    ]
    r4_state = torch.load(args.checkpoints / "r4.pt", map_location="cpu", weights_only=True)
    runtime = FrozenR5Runtime.build(
        candidate_ids=train_ids,
        cases=formal,
        raw_cases=raw_cases,
        facts_by_case=facts_by_case,
        image_by_id=image_by_id,
        report_by_id=report_by_id,
        checkpoint_states=checkpoint_states,
        r4_checkpoint_state=r4_state,
    )
    candidate_label_matrix = np.stack([candidate_labels_by_id[case_id] for case_id in runtime.candidate_ids])
    term_cache: dict[str, tuple[np.ndarray, int]] = {}

    def build_states(ids: Sequence[str], *, leave_one_out: bool, label: str) -> list[dict[str, Any]]:
        states = []
        for position, case_id in enumerate(ids, start=1):
            profiles = qrel_profiles(case_id, runtime.candidate_ids, prepared_by_case)
            for question_type in QUESTIONS:
                state = build_retrieval_state(
                    runtime,
                    formal[case_id],
                    image_by_id[case_id],
                    case_id,
                    question_type,
                    query_embedding[(case_id, question_type)],
                    train_medcpt,
                    raw_cases,
                    facts_by_case,
                    prepared_by_case,
                    leave_one_out=leave_one_out,
                    term_cache=term_cache,
                    precomputed_qrels=profiles["combined"],
                )
                state["qrels_by_metric"] = profiles
                states.append(state)
            if position % 100 == 0 or position == len(ids):
                print(f"{label}_states={position}/{len(ids)}", flush=True)
        return states

    fit_states = build_states(fit_ids, leave_one_out=True, label="fit")
    internal_states = build_states(internal_ids, leave_one_out=True, label="internal")
    calibration_states = build_states(calibration_ids, leave_one_out=False, label="calibration")

    def matrix_and_labels(
        states: Sequence[dict[str, Any]], *, concept: bool
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        matrices = []
        labels = []
        groups = []
        for state in states:
            candidate_ids = state["rrf_rank"][:200]
            indices = [runtime.candidate_ids.index(case_id) for case_id in candidate_ids]
            values = state["features_by_index"][indices]
            if concept:
                values = append_concept_features(
                    values,
                    query_probabilities[state["query_id"]],
                    candidate_label_matrix[indices],
                )
            matrices.append(values)
            labels.extend(state["qrels_by_metric"]["combined"].get(case_id, 0.0) for case_id in candidate_ids)
            groups.append(len(candidate_ids))
        return np.concatenate(matrices), np.asarray(labels, dtype=np.float32), groups

    x_fit_base, y_fit, groups_fit = matrix_and_labels(fit_states, concept=False)
    x_internal_base, y_internal, groups_internal = matrix_and_labels(internal_states, concept=False)
    x_fit_concept, y_fit_concept, groups_fit_concept = matrix_and_labels(fit_states, concept=True)
    x_internal_concept, y_internal_concept, groups_internal_concept = matrix_and_labels(internal_states, concept=True)
    if not np.array_equal(y_fit, y_fit_concept) or groups_fit != groups_fit_concept:
        raise RuntimeError("Base and concept fit groups differ")
    if not np.array_equal(y_internal, y_internal_concept) or groups_internal != groups_internal_concept:
        raise RuntimeError("Base and concept internal groups differ")
    base_ranker = fit_ranker(
        x_fit_base,
        y_fit,
        groups_fit,
        x_internal_base,
        y_internal,
        groups_internal,
        config_name=args.ranker_config,
    )
    concept_ranker = fit_ranker(
        x_fit_concept,
        y_fit_concept,
        groups_fit_concept,
        x_internal_concept,
        y_internal_concept,
        groups_internal_concept,
        config_name=args.ranker_config,
    )

    def rank(state: dict[str, Any], ranker: LGBMRanker, *, concept: bool) -> list[str]:
        candidate_ids = state["rrf_rank"][:200]
        indices = [runtime.candidate_ids.index(case_id) for case_id in candidate_ids]
        values = state["features_by_index"][indices]
        if concept:
            values = append_concept_features(
                values,
                query_probabilities[state["query_id"]],
                candidate_label_matrix[indices],
            )
        scores = ranker.predict(values)
        return [case_id for _, case_id in sorted(zip(scores, candidate_ids), key=lambda item: (-float(item[0]), item[1]))]

    def evaluate(states: Sequence[dict[str, Any]], partition: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows = []
        for state in states:
            rankings = {
                "base_17": rank(state, base_ranker, concept=False),
                "concept_23": rank(state, concept_ranker, concept=True),
            }
            row: dict[str, Any] = {
                "partition": partition,
                "case_id": state["query_id"],
                "question_type": state["question_type"],
                "spectrum": state["spectrum"],
                "systems": {},
            }
            relevant = {
                case_id for case_id, gain in state["qrels_by_metric"]["combined"].items() if gain >= 0.5
            }
            pool = set(state["rrf_rank"][:200])
            for name, ranking in rankings.items():
                row[name] = {}
                for metric_name, qrels in state["qrels_by_metric"].items():
                    row[name][f"{metric_name}_ndcg10"] = ndcg(ranking, qrels, 10)
                row[name]["hit1"] = float(bool(relevant) and ranking[0] in relevant)
                row[name]["hit5"] = float(bool(relevant) and any(case_id in relevant for case_id in ranking[:5]))
                row[name]["top200"] = ranking[:200]
            row["relevant_count"] = len(relevant)
            row["rrf_relevant_presence"] = float(bool(relevant & pool))
            row["rrf_relevant_recall"] = float(len(relevant & pool) / len(relevant)) if relevant else 0.0
            rows.append(row)

        summary: dict[str, Any] = {"query_count": len(rows), "systems": {}, "bootstrap": {}}
        for system in ("base_17", "concept_23"):
            metrics = {}
            for metric in ("combined_ndcg10", "label_only_ndcg10", "fact_only_ndcg10", "hit1", "hit5"):
                metrics[metric] = mean([float(row[system][metric]) for row in rows])
            for group in ("normal", "abnormal", "indeterminate"):
                selected = [row for row in rows if row["spectrum"] == group]
                metrics[f"{group}_combined_ndcg10"] = mean(
                    [float(row[system]["combined_ndcg10"]) for row in selected]
                )
            summary["systems"][system] = metrics
        summary["candidate_frame"] = {
            "relevant_presence": mean([float(row["rrf_relevant_presence"]) for row in rows]),
            "relevant_recall": mean([float(row["rrf_relevant_recall"]) for row in rows]),
        }
        for metric in ("combined_ndcg10", "label_only_ndcg10", "fact_only_ndcg10", "hit1", "hit5"):
            summary["bootstrap"][metric] = case_grouped_bootstrap(rows, metric)
        return rows, summary

    calibration_rows, calibration_summary = evaluate(calibration_states, "calibration")
    calibration_delta = calibration_summary["bootstrap"]["combined_ndcg10"]["difference"]
    calibration_fact_delta = calibration_summary["bootstrap"]["fact_only_ndcg10"]["difference"]
    promoted = calibration_delta >= 0.005 and calibration_fact_delta >= -0.005
    validation_rows: list[dict[str, Any]] = []
    validation_summary: dict[str, Any] | None = None
    if promoted:
        validation_states = build_states(validation_ids, leave_one_out=False, label="validation")
        validation_rows, validation_summary = evaluate(validation_states, "validation")

    args.model_dir.mkdir(parents=True, exist_ok=True)
    base_model_path = args.model_dir / f"v14_{args.ranker_config}_base_17.txt"
    concept_model_path = args.model_dir / f"v14_{args.ranker_config}_concept_23.txt"
    base_ranker.booster_.save_model(str(base_model_path))
    concept_ranker.booster_.save_model(str(concept_model_path))
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    args.rows.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
            for row in calibration_rows + validation_rows
        ),
        encoding="utf-8",
    )
    output = {
        "study": "V14 concept-aware retrieval development",
        "status": "validation_evaluated" if promoted else "stopped_after_calibration",
        "protocol_commit": PROTOCOL_COMMIT,
        "test_loaded_or_evaluated": False,
        "promotion": {
            "calibration_combined_minimum": 0.005,
            "calibration_fact_only_floor": -0.005,
            "observed_combined_difference": calibration_delta,
            "observed_fact_only_difference": calibration_fact_delta,
            "promoted_to_validation": promoted,
        },
        "features": {"base": 17, "concept": 23, "appended": list(FEATURE_NAMES)},
        "ranker_configuration": {
            "name": args.ranker_config,
            **RANKER_CONFIGS[args.ranker_config],
            "random_state": 2026,
        },
        "oof": {
            "folds": 5,
            "seed": 7145,
            "case_count": len(oof_by_id),
            "case_ids_sha256": case_id_fingerprint(list(oof_by_id)),
            "records": oof_records,
            "cache_sha256": file_sha256(args.oof_cache),
        },
        "rankers": {
            "base_17": {
                "best_iteration": int(base_ranker.best_iteration_ or 0),
                "model_sha256": file_sha256(base_model_path),
            },
            "concept_23": {
                "best_iteration": int(concept_ranker.best_iteration_ or 0),
                "model_sha256": file_sha256(concept_model_path),
            },
        },
        "calibration": calibration_summary,
        "validation": validation_summary,
        "counts": {
            "source_train": len(train_source_ids),
            "historical_bank": len(train_ids),
            "fit_cases": len(fit_ids),
            "internal_cases": len(internal_ids),
            "source_calibration": len(source_partition_ids["calibration"]),
            "calibration_cases": len(calibration_ids),
            "source_validation": len(source_partition_ids["validation"]),
            "validation_cases": len(validation_ids) if promoted else 0,
        },
        "eligibility_exclusions": {
            name: {
                "count": len(case_ids),
                "case_ids_sha256": case_id_fingerprint(case_ids) if case_ids else None,
                "reason": "missing successful RadGraph annotation or formal paired case",
            }
            for name, case_ids in exclusions.items()
        },
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "radgraph_sha256": file_sha256(args.radgraph),
            "split_sha256": file_sha256(args.split),
            "roles_sha256": file_sha256(args.roles),
            "embeddings_sha256": file_sha256(args.embeddings),
            "medcpt_sha256": file_sha256(args.medcpt),
            "v13_checkpoint_sha256": file_sha256(checkpoint_path),
            "chexbert_checkpoint_sha256": file_sha256(chexbert_checkpoint),
            "model_type": model_type,
        },
        "artifacts": {
            "rows": str(args.rows.relative_to(ROOT).as_posix()),
            "rows_sha256": file_sha256(args.rows),
            "base_model": str(base_model_path.relative_to(ROOT).as_posix()),
            "concept_model": str(concept_model_path.relative_to(ROOT).as_posix()),
        },
        "runtime_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Train/Calibration/Validation development using automated report-derived "
            "relevance and concept proxies; not clinical diagnosis, safety, external "
            "validation, physician utility, or patient-level independence."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"promotion": output["promotion"], "calibration": calibration_summary, "validation": validation_summary}, indent=2))


if __name__ == "__main__":
    main()
