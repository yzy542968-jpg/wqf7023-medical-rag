from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from medical_rag.similar_case.openi_adapter import read_openi_paired_cases  # noqa: E402
from medical_rag.similar_case.v10_multiview import (  # noqa: E402
    AttentionTrainingRecord,
    ViewAttention,
    attention_record_loss,
    attention_view_scores,
    l2_normalize,
    make_attention_record,
    max_view_scores,
    mean_view_scores,
)
from medical_rag.similar_case.v10_split import file_sha256  # noqa: E402
from train_v9_learned_reranker import numeric_ndcg10, relevance_array  # noqa: E402


DEFAULT_CASES = ROOT / "data" / "processed" / "openi_cases.jsonl"
DEFAULT_RADGRAPH = ROOT / "data" / "processed" / "v9_radgraph_modern_xl.jsonl"
DEFAULT_SPLIT = ROOT / "data" / "splits" / "v10" / "v10_cluster_disjoint_split.json"
DEFAULT_ROLES = ROOT / "data" / "splits" / "v10" / "v10_reranker_roles.json"
DEFAULT_CONFIG = ROOT / "config" / "v10_multiview_development.json"
DEFAULT_EMBEDDINGS = ROOT / "data" / "processed" / "v10_medsiglip_embeddings.npz"
DEFAULT_CHECKPOINTS = ROOT / "experiments" / "v10_publication" / "multiview_checkpoints"
DEFAULT_ROWS = ROOT / "experiments" / "v10_publication" / "v10_multiview_validation_rows.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "splits" / "v10" / "v10_multiview_development_summary.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def evaluate_model(
    model: ViewAttention,
    states: Sequence[tuple[np.ndarray, np.ndarray]],
    candidate_images: np.ndarray,
    candidate_reports: np.ndarray,
) -> float:
    values = []
    for views, gains in states:
        scores = attention_view_scores([model], views, candidate_images, candidate_reports)
        ranking = np.lexsort((np.arange(len(scores)), -scores))
        values.append(numeric_ndcg10(gains, ranking))
    return statistics.fmean(values)


def fit_attention(
    records: Sequence[AttentionTrainingRecord],
    internal_states: Sequence[tuple[np.ndarray, np.ndarray]],
    candidate_images: np.ndarray,
    candidate_reports: np.ndarray,
    *,
    seed: int,
    config: Mapping[str, Any],
) -> tuple[ViewAttention, list[dict[str, float]], int, float]:
    set_determinism(seed)
    model = ViewAttention(width=int(config["attention_architecture"][0]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_value = -np.inf
    best_epoch = 0
    patience = 0
    history = []
    for epoch in range(1, int(config["training"]["maximum_epochs"]) + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(records))
        optimizer.zero_grad(set_to_none=True)
        losses = []
        pending = []
        for position, index in enumerate(order, start=1):
            loss = attention_record_loss(model, records[int(index)])
            pending.append(loss)
            losses.append(float(loss.detach()))
            if len(pending) == 32 or position == len(order):
                torch.stack(pending).mean().backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending = []
        internal = evaluate_model(model, internal_states, candidate_images, candidate_reports)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "internal_ndcg@10": internal})
        if internal >= best_value + float(config["training"]["early_stopping_minimum_improvement"]):
            best_value = internal
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        print(f"seed={seed} epoch={epoch} internal_ndcg@10={internal:.6f}", flush=True)
        if patience >= int(config["training"]["early_stopping_patience"]):
            break
    model.load_state_dict(best_state)
    return model, history, best_epoch, float(best_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Develop the V10 multi-view aggregation policy.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--radgraph", type=Path, default=DEFAULT_RADGRAPH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--rows-output", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    config = read_json(args.config)
    split = read_json(args.split)
    roles = read_json(args.roles)
    if config["validation_outcomes_inspected"] or config["test_outcomes_inspected"]:
        raise RuntimeError("multi-view config records inspected outcomes")

    cases = read_openi_paired_cases(
        args.cases,
        source_unique_patient=True,
        radgraph_path=args.radgraph,
    )
    case_by_id = {case.study_id: case for case in cases}
    with np.load(args.embeddings, allow_pickle=False) as encoded:
        case_ids = [str(value) for value in encoded["case_ids"]]
        case_images = l2_normalize(np.asarray(encoded["case_image_embeddings"], dtype=np.float32))
        report_ids = [str(value) for value in encoded["report_ids"]]
        reports = l2_normalize(np.asarray(encoded["report_embeddings"], dtype=np.float32))
        view_ids = [str(value) for value in encoded["view_case_ids"]]
        views = l2_normalize(np.asarray(encoded["view_embeddings"], dtype=np.float32))
        embedding_signature = str(encoded["signature"].item())
    image_by_id = {case_id: case_images[index] for index, case_id in enumerate(case_ids)}
    report_by_id = {case_id: reports[index] for index, case_id in enumerate(report_ids)}
    views_by_id: dict[str, list[np.ndarray]] = {}
    for case_id, embedding in zip(view_ids, views, strict=True):
        views_by_id.setdefault(case_id, []).append(embedding)

    train_ids = set(split["partitions"]["train"]["case_ids"])
    validation_ids = set(split["partitions"]["validation"]["case_ids"])
    eligible = set(case_by_id) & set(image_by_id) & set(report_by_id) & set(views_by_id)
    candidate_ids = sorted(train_ids & eligible)
    candidate_index = {case_id: index for index, case_id in enumerate(candidate_ids)}
    candidate_cases = [case_by_id[case_id] for case_id in candidate_ids]
    candidate_images = np.stack([image_by_id[case_id] for case_id in candidate_ids])
    candidate_reports = np.stack([report_by_id[case_id] for case_id in candidate_ids])
    candidate_vectors = l2_normalize(0.5 * candidate_images + 0.5 * candidate_reports)
    prepared_labels = [dict(case.labels) for case in candidate_cases]
    prepared_facts = [case.radgraph_facts for case in candidate_cases]

    def query_state(case_id: str) -> tuple[np.ndarray, np.ndarray]:
        excluded = candidate_index.get(case_id)
        gains = relevance_array(
            case_by_id[case_id],
            candidate_cases,
            excluded,
            prepared_labels=prepared_labels,
            prepared_facts=prepared_facts,
        )
        return np.stack(views_by_id[case_id]), gains

    fit_ids = sorted(set(roles["roles"]["pairwise_fit"]["case_ids"]) & eligible)
    internal_ids = sorted(set(roles["roles"]["internal_early_stop"]["case_ids"]) & eligible)
    validation_ids = sorted(validation_ids & eligible)
    pair_config = config["pair_sampling"]
    fit_records = []
    for position, case_id in enumerate(fit_ids, start=1):
        query_views, gains = query_state(case_id)
        record = make_attention_record(
            query_views,
            candidate_vectors,
            gains,
            high_candidates=int(pair_config["high_candidates"]),
            low_candidates=int(pair_config["low_candidates"]),
            minimum_gain_difference=float(pair_config["minimum_gain_difference"]),
        )
        if record is not None:
            fit_records.append(record)
        if position % 250 == 0 or position == len(fit_ids):
            print(f"attention_fit_records={position}/{len(fit_ids)}", flush=True)
    internal_states = [query_state(case_id) for case_id in internal_ids]

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    models = []
    seed_records = []
    for seed in config["seeds"]:
        model, history, best_epoch, internal = fit_attention(
            fit_records,
            internal_states,
            candidate_images,
            candidate_reports,
            seed=int(seed),
            config=config,
        )
        state = model.state_dict()
        torch.save(state, args.checkpoint_dir / f"attention_seed_{seed}.pt")
        models.append(model)
        seed_records.append(
            {
                "seed": seed,
                "best_epoch": best_epoch,
                "internal_ndcg@10": internal,
                "checkpoint_sha256": state_sha256(state),
                "history": history,
            }
        )

    rows = []
    for position, case_id in enumerate(validation_ids, start=1):
        query_views, gains = query_state(case_id)
        score_sets = {
            "mean": mean_view_scores(query_views, candidate_images, candidate_reports),
            "per_view_max": max_view_scores(query_views, candidate_images, candidate_reports),
            "learned_attention": attention_view_scores(
                models, query_views, candidate_images, candidate_reports
            ),
        }
        metrics = {}
        top3 = {}
        for name, scores in score_sets.items():
            ranking = np.lexsort((np.arange(len(scores)), -scores))
            metrics[name] = numeric_ndcg10(gains, ranking)
            top3[name] = [candidate_ids[index] for index in ranking[:3]]
        rows.append(
            {
                "case_id": case_id,
                "view_count": len(query_views),
                "ndcg@10": metrics,
                "top3": top3,
            }
        )
        if position % 50 == 0 or position == len(validation_ids):
            print(f"multiview_validation={position}/{len(validation_ids)}", flush=True)

    metrics = {
        name: statistics.fmean(row["ndcg@10"][name] for row in rows)
        for name in ("mean", "per_view_max", "learned_attention")
    }
    margin = float(config["promotion_margin_ndcg10"])
    if metrics["learned_attention"] - metrics["mean"] >= margin:
        selected = "learned_attention"
    elif metrics["per_view_max"] - metrics["mean"] >= margin:
        selected = "per_view_max"
    else:
        selected = "mean"

    args.rows_output.parent.mkdir(parents=True, exist_ok=True)
    args.rows_output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary = {
        "study": "V10 multi-view development",
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
            "fit_records": len(fit_records),
            "internal_queries": len(internal_states),
            "validation_queries": len(rows),
        },
        "attention_seeds": seed_records,
        "validation_ndcg@10": metrics,
        "selected_policy": selected,
        "promotion_margin_ndcg10": margin,
        "validation_rows_sha256": file_sha256(args.rows_output),
        "test_outcomes_inspected": False,
    }
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation_ndcg@10": metrics, "selected_policy": selected}, indent=2))


if __name__ == "__main__":
    main()
