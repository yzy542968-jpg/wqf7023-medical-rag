from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.radgraph_adapter import read_radgraph_case_records  # noqa: E402
from medical_rag.similar_case.v10_reranker import (  # noqa: E402
    FactAwareFeatureIndex,
    R4Scorer,
    R5Scorer,
    augment_r4_features,
    sample_fact_aware_pairs,
    set_determinism,
    train_epoch,
)
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from train_v9_learned_reranker import (  # noqa: E402
    exact_leave_one_out_bm25_scores,
    feature_matrix,
    numeric_ndcg10,
    relevance_array,
    sample_pairs,
)


DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v10" / "v10_cluster_disjoint_split.json"
DEFAULT_ROLES = ROOT / "data" / "splits" / "v10" / "v10_reranker_roles.json"
DEFAULT_CONFIG = ROOT / "config" / "v10_reranker_development.json"
DEFAULT_EMBEDDINGS = ROOT / "data" / "processed" / "v10_medsiglip_embeddings.npz"
DEFAULT_CHECKPOINTS = ROOT / "experiments" / "v10_publication" / "reranker_checkpoints"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_reranker_development_summary.json"


QUESTIONS = {
    "findings": "What are the main radiographic findings?",
    "impression": "What is the most likely radiographic impression?",
    "acute": "Is there an acute cardiopulmonary abnormality? Explain briefly.",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def checkpoint_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate_states(model: torch.nn.Module, states: Sequence[dict[str, Any]], feature_key: str) -> float:
    values = []
    model.eval()
    with torch.inference_mode():
        for state in states:
            for features in state[feature_key]:
                scores = model(torch.from_numpy(features)).numpy()
                if state["excluded_index"] is not None:
                    scores[state["excluded_index"]] = -np.inf
                ranking = np.lexsort((np.arange(len(scores)), -scores))
                values.append(numeric_ndcg10(state["gains"], ranking))
    return statistics.fmean(values)


def fit_model(
    model: torch.nn.Module,
    *,
    high: np.ndarray,
    low: np.ndarray,
    weights: np.ndarray,
    internal_states: Sequence[dict[str, Any]],
    feature_key: str,
    seed: int,
    training: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], int, float]:
    set_determinism(seed)
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_value = -np.inf
    best_epoch = 0
    patience = 0
    history = []
    for epoch in range(1, int(training["maximum_epochs"]) + 1):
        loss = train_epoch(
            model,
            optimizer,
            high,
            low,
            weights,
            batch_size=int(training["batch_size"]),
            seed=seed + epoch,
        )
        value = evaluate_states(model, internal_states, feature_key)
        history.append({"epoch": epoch, "loss": loss, "internal_ndcg@10": value})
        if value >= best_value + float(training["early_stopping_minimum_improvement"]):
            best_value = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        if patience >= int(training["early_stopping_patience"]):
            break
    model.load_state_dict(best_state)
    return best_state, history, best_epoch, float(best_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V10 R4 and five-seed fact-aware R5 rerankers.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    config = read_json(args.config)
    split = read_json(args.split)
    roles = read_json(args.roles)
    if config["validation_outcomes_inspected"] or config["test_outcomes_inspected"]:
        raise RuntimeError("V10 reranker config records inspected outcomes")
    raw_rows = read_jsonl(args.cases)
    raw_cases = {str(row["case_id"]): row for row in raw_rows}
    formal = read_openi_paired_cases(
        args.cases,
        source_unique_patient=True,
        radgraph_path=args.radgraph,
    )
    formal_cases = {case.study_id: case for case in formal}
    radgraph = read_radgraph_case_records(args.radgraph)

    with np.load(args.embeddings, allow_pickle=False) as encoded:
        all_ids = [str(value) for value in encoded["case_ids"]]
        all_images = np.asarray(encoded["case_image_embeddings"], dtype=np.float32)
        report_ids = [str(value) for value in encoded["report_ids"]]
        report_embeddings = np.asarray(encoded["report_embeddings"], dtype=np.float32)
        embedding_signature = str(encoded["signature"].item())
    image_by_id = {case_id: all_images[index] for index, case_id in enumerate(all_ids)}
    report_by_id = {case_id: report_embeddings[index] for index, case_id in enumerate(report_ids)}

    train_ids = set(split["partitions"]["train"]["case_ids"])
    validation_ids = set(split["partitions"]["validation"]["case_ids"])
    eligible = {
        case_id
        for case_id in raw_cases
        if case_id in report_by_id
        and case_id in image_by_id
        and radgraph[case_id].status == "ok"
    }
    candidate_ids = sorted(train_ids & eligible)
    validation_query_ids = sorted(validation_ids & eligible)
    candidate_index = {case_id: index for index, case_id in enumerate(candidate_ids)}
    bank = [formal_cases[case_id] for case_id in candidate_ids]
    bank_images = np.stack([image_by_id[case_id] for case_id in candidate_ids])
    bank_reports = np.stack([report_by_id[case_id] for case_id in candidate_ids])
    bm25 = BM25Retriever().fit(
        [{"case_id": case.study_id, "report_text": case.report_text} for case in bank]
    )
    term_cache: dict[str, tuple[np.ndarray, int]] = {}
    facts_by_case = {case_id: tuple(radgraph[case_id].facts) for case_id in candidate_ids}
    fact_index = FactAwareFeatureIndex.build(candidate_ids, raw_cases, facts_by_case)
    prepared_labels = [dict(case.labels) for case in bank]
    prepared_facts = [case.radgraph_facts for case in bank]

    def query_state(case_id: str) -> dict[str, Any]:
        query_case = formal_cases[case_id]
        excluded_index = candidate_index.get(case_id)
        gains = relevance_array(
            query_case,
            bank,
            excluded_index,
            prepared_labels=prepared_labels,
            prepared_facts=prepared_facts,
        )
        image_image = bank_images @ image_by_id[case_id]
        image_report = bank_reports @ image_by_id[case_id]
        if excluded_index is not None:
            image_image[excluded_index] = -np.inf
            image_report[excluded_index] = -np.inf
        r4_features = []
        r5_features = []
        bm25_scores = []
        for question_type, question in QUESTIONS.items():
            query_text = "\n".join(part for part in (query_case.indication, question) if part)
            text_scores = exact_leave_one_out_bm25_scores(
                bm25,
                query_text,
                excluded_index=excluded_index,
                term_cache=term_cache,
            )
            base = feature_matrix(
                text_scores,
                image_image,
                image_report,
                question_type=question_type,
                excluded_index=excluded_index,
            )
            r4_features.append(base)
            r5_features.append(augment_r4_features(base, fact_index.query_features(query_text)))
            bm25_scores.append(text_scores)
        return {
            "case_id": case_id,
            "excluded_index": excluded_index,
            "gains": gains,
            "image_image": image_image,
            "image_report": image_report,
            "bm25": bm25_scores,
            "r4": r4_features,
            "r5": r5_features,
        }

    fit_ids = sorted(set(roles["roles"]["pairwise_fit"]["case_ids"]) & eligible)
    internal_ids = sorted(set(roles["roles"]["internal_early_stop"]["case_ids"]) & eligible)
    r4_high: list[np.ndarray] = []
    r4_low: list[np.ndarray] = []
    r4_weights: list[float] = []
    r5_high: list[np.ndarray] = []
    r5_low: list[np.ndarray] = []
    r5_weights: list[np.ndarray] = []
    pair_config = config["pair_sampling"]
    v9_pair_config = {
        "top_per_bm25": pair_config["component_top_k"],
        "top_per_image_image": pair_config["component_top_k"],
        "top_per_image_report": pair_config["component_top_k"],
        "top_per_relevance": pair_config["relevance_top_k"],
        "bottom_per_relevance": pair_config["relevance_bottom_k"],
        "high_candidates": 8,
        "low_candidates": 8,
        "minimum_gain_difference": pair_config["minimum_gain_difference"],
    }
    for position, case_id in enumerate(fit_ids, start=1):
        state = query_state(case_id)
        for index in range(len(QUESTIONS)):
            components = [state["bm25"][index], state["image_image"], state["image_report"]]
            high, low, weights = sample_pairs(
                state["r4"][index], state["gains"], components, config=v9_pair_config
            )
            r4_high.extend(high)
            r4_low.extend(low)
            r4_weights.extend(weights)
            high5, low5, weights5 = sample_fact_aware_pairs(
                state["r5"][index], state["gains"], components, config=pair_config
            )
            if len(weights5):
                r5_high.append(high5)
                r5_low.append(low5)
                r5_weights.append(weights5)
        if position % 100 == 0 or position == len(fit_ids):
            print(f"fit_states={position}/{len(fit_ids)}", flush=True)
    r4_high_array = np.asarray(r4_high, dtype=np.float32).reshape(-1, 9)
    r4_low_array = np.asarray(r4_low, dtype=np.float32).reshape(-1, 9)
    r4_weight_array = np.asarray(r4_weights, dtype=np.float32)
    r5_high_array = np.concatenate(r5_high, axis=0)
    r5_low_array = np.concatenate(r5_low, axis=0)
    r5_weight_array = np.concatenate(r5_weights, axis=0)

    internal_states = []
    for position, case_id in enumerate(internal_ids, start=1):
        internal_states.append(query_state(case_id))
        if position % 50 == 0 or position == len(internal_ids):
            print(f"internal_states={position}/{len(internal_ids)}", flush=True)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    training = config["training"]
    r4_model = R4Scorer()
    r4_state, r4_history, r4_epoch, r4_internal = fit_model(
        r4_model,
        high=r4_high_array,
        low=r4_low_array,
        weights=r4_weight_array,
        internal_states=internal_states,
        feature_key="r4",
        seed=int(config["r5_seeds"][0]),
        training=training,
    )
    torch.save(r4_state, args.checkpoint_dir / "r4.pt")

    r5_models = []
    r5_records = []
    for seed in config["r5_seeds"]:
        model = R5Scorer()
        state, history, epoch, internal = fit_model(
            model,
            high=r5_high_array,
            low=r5_low_array,
            weights=r5_weight_array,
            internal_states=internal_states,
            feature_key="r5",
            seed=int(seed),
            training=training,
        )
        model.load_state_dict(state)
        torch.save(state, args.checkpoint_dir / f"r5_seed_{seed}.pt")
        r5_models.append(model)
        r5_records.append(
            {
                "seed": seed,
                "best_epoch": epoch,
                "internal_ndcg@10": internal,
                "checkpoint_sha256": checkpoint_sha256(state),
                "history": history,
            }
        )

    validation_rows = []
    for position, case_id in enumerate(validation_query_ids, start=1):
        state = query_state(case_id)
        for question_index, question_type in enumerate(QUESTIONS):
            gains = state["gains"]
            component_ranking = np.lexsort((np.arange(len(candidate_ids)), -state["image_image"]))
            r4_model.eval()
            with torch.inference_mode():
                r4_scores = r4_model(torch.from_numpy(state["r4"][question_index])).numpy()
                seed_scores = np.stack(
                    [model(torch.from_numpy(state["r5"][question_index])).numpy() for model in r5_models]
                )
            r4_ranking = np.lexsort((np.arange(len(candidate_ids)), -r4_scores))
            seed_ndcgs = []
            for seed, scores in zip(config["r5_seeds"], seed_scores, strict=True):
                ranking = np.lexsort((np.arange(len(candidate_ids)), -scores))
                seed_ndcgs.append((seed, numeric_ndcg10(gains, ranking)))
            ensemble_scores = seed_scores.mean(axis=0)
            ensemble_ranking = np.lexsort((np.arange(len(candidate_ids)), -ensemble_scores))
            validation_rows.append(
                {
                    "case_id": case_id,
                    "question_type": question_type,
                    "image_image": numeric_ndcg10(gains, component_ranking),
                    "r4": numeric_ndcg10(gains, r4_ranking),
                    "r5_ensemble": numeric_ndcg10(gains, ensemble_ranking),
                    "r5_seeds": {str(seed): value for seed, value in seed_ndcgs},
                }
            )
        if position % 25 == 0 or position == len(validation_query_ids):
            print(f"validation_states={position}/{len(validation_query_ids)}", flush=True)

    metrics = {
        "image_image": statistics.fmean(row["image_image"] for row in validation_rows),
        "r4": statistics.fmean(row["r4"] for row in validation_rows),
        "r5_ensemble": statistics.fmean(row["r5_ensemble"] for row in validation_rows),
        "r5_seeds": {
            str(seed): statistics.fmean(row["r5_seeds"][str(seed)] for row in validation_rows)
            for seed in config["r5_seeds"]
        },
    }
    best_seed, best_seed_value = max(metrics["r5_seeds"].items(), key=lambda row: (row[1], -int(row[0])))
    use_ensemble = best_seed_value - metrics["r5_ensemble"] < float(
        config["ensemble_degradation_tolerance_ndcg10"]
    )
    selected_r5 = "ensemble" if use_ensemble else f"seed_{best_seed}"
    selected_value = metrics["r5_ensemble"] if use_ensemble else best_seed_value
    promoted = selected_value - metrics["r4"] >= float(config["promotion_margin_ndcg10"])

    summary = {
        "study": "V10 fact-aware reranker development",
        "status": "development_complete_test_not_run",
        "inputs": {
            "cases_sha256": file_sha256(args.cases),
            "radgraph_sha256": file_sha256(args.radgraph),
            "split_sha256": file_sha256(args.split),
            "roles_sha256": file_sha256(args.roles),
            "config_sha256": file_sha256(args.config),
            "embedding_signature": embedding_signature,
        },
        "counts": {
            "candidate_bank": len(candidate_ids),
            "fit_queries": len(fit_ids),
            "internal_queries": len(internal_ids),
            "validation_queries": len(validation_query_ids),
            "r4_pairs": len(r4_weight_array),
            "r5_pairs": len(r5_weight_array),
        },
        "r4": {
            "best_epoch": r4_epoch,
            "internal_ndcg@10": r4_internal,
            "checkpoint_sha256": checkpoint_sha256(r4_state),
            "history": r4_history,
        },
        "r5": r5_records,
        "validation_ndcg@10": metrics,
        "selected_r5": selected_r5,
        "selected_r5_ndcg@10": selected_value,
        "selected_r5_minus_r4": selected_value - metrics["r4"],
        "promoted": promoted,
        "validation_rows_sha256": hashlib.sha256(
            json.dumps(validation_rows, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "test_outcomes_inspected": False,
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation_ndcg@10": metrics, "selected_r5": selected_r5, "promoted": promoted}, indent=2))


if __name__ == "__main__":
    main()
