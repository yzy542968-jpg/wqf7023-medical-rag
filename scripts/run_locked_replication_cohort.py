from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.metrics import evaluate_retrieval
from medical_rag.evaluation.replication_cohort import build_replication_cohort
from medical_rag.retrieval.adaptive_retrieval import select_adaptive_top1
from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.retrieval.hybrid_retriever import minmax
from medical_rag.retrieval.medcpt_reranker import MedCPTReranker, case_document
from medical_rag.retrieval.medcpt_retriever import MedCPTRetriever, encode_queries
from medical_rag.retrieval.tfidf_retriever import _tokens, load_cases_jsonl


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    payload = _read_json(path)
    ids = {
        str(row["case_id"])
        for row in payload.get("questions", [])
        if row.get("case_id") is not None
    }
    ids.update(str(value) for value in payload.get("case_ids", []))
    for part in payload.get("split", {}).values():
        ids.update(str(value) for value in part.get("case_ids", []))
    return ids


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prompt(question: dict[str, Any], selected_case: dict[str, Any] | None) -> str:
    if selected_case is None:
        context = "No sufficiently confident case was retrieved."
    else:
        context = "\n".join(
            [
                f"Case ID: {selected_case['case_id']}",
                f"Findings: {selected_case.get('findings', '')}",
                f"Impression: {selected_case.get('impression', '')}",
            ]
        )
    return "\n".join(
        [
            "Answer the medical question using the selected radiology case.",
            "Question:",
            str(question["question"]),
            "",
            "Selected radiology case:",
            context,
            "",
            "Answer clearly and concisely.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the previously locked retrieval system on an untouched OpenI cohort."
    )
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "data" / "processed" / "openi_cases.jsonl"
    )
    parser.add_argument(
        "--index", type=Path, default=ROOT / "data" / "processed" / "openi_medcpt_full.npz"
    )
    parser.add_argument(
        "--cohort-output",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_locked_replication_cohort.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "locked_replication",
    )
    parser.add_argument("--max-cases", type=int, default=300)
    parser.add_argument("--seed", type=int, default=47023)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    hybrid_config_path = (
        ROOT / "experiments" / "final_optimized" / "retrieval" / "hybrid_alpha_selection.json"
    )
    reranker_config_path = (
        ROOT / "experiments" / "final_optimized" / "reranking" / "medcpt_reranker_selection.json"
    )
    policy_config_path = (
        ROOT
        / "experiments"
        / "final_optimized"
        / "adaptive_retrieval"
        / "adaptive_policy_selection.json"
    )
    hybrid_config = _read_json(hybrid_config_path)
    reranker_config = _read_json(reranker_config_path)
    policy_config = _read_json(policy_config_path)
    alpha = float(hybrid_config["selected_alpha"])
    candidate_depth = int(reranker_config["selected_candidate_depth"])
    policy = {
        key: policy_config["selected_policy"][key]
        for key in (
            "reranker_margin_threshold",
            "base_margin_threshold",
            "minimum_base_score",
            "minimum_selected_margin",
        )
    }

    prior_paths = [
        ROOT / "data" / "processed" / "openi_case_qa_seed_clean.json",
        ROOT / "data" / "processed" / "openi_case_scoped_benchmark_v2.json",
        ROOT / "data" / "processed" / "openi_case_scoped_confirmation_v2.json",
        ROOT / "data" / "processed" / "openi_case_scoped_hard_v21.json",
    ]
    excluded: set[str] = set()
    for path in prior_paths:
        excluded.update(_case_ids(path))

    cases = load_cases_jsonl(args.cases)
    case_ids = [str(case["case_id"]) for case in cases]
    case_by_id = {str(case["case_id"]): case for case in cases}
    case_position = {case_id: index for index, case_id in enumerate(case_ids)}
    cohort = build_replication_cohort(
        cases, excluded, max_cases=args.max_cases, seed=args.seed
    )
    questions = cohort["questions"]
    args.cohort_output.parent.mkdir(parents=True, exist_ok=True)
    args.cohort_output.write_text(
        json.dumps(cohort, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    bm25 = BM25Retriever().fit(cases)
    medcpt = MedCPTRetriever.from_index(args.cases, args.index)
    dense_position = {case_id: index for index, case_id in enumerate(medcpt.case_ids)}
    query_embeddings = encode_queries(
        [str(row["question"]) for row in questions],
        batch_size=args.batch_size,
        device=args.device,
    )

    base_rows: list[dict[str, Any]] = []
    flat_pairs: list[tuple[str, str]] = []
    pair_keys: list[tuple[str, str]] = []
    for query_index, question in enumerate(questions):
        bm25_scores = np.array(
            [
                bm25._score_document(_tokens(str(question["question"])), index)
                for index in range(len(cases))
            ],
            dtype="float32",
        )
        dense_indexed = medcpt.embeddings @ query_embeddings[query_index]
        dense_scores = np.zeros(len(cases), dtype="float32")
        for case_id, dense_index in dense_position.items():
            case_index = case_position.get(case_id)
            if case_index is not None:
                dense_scores[case_index] = dense_indexed[dense_index]
        hybrid_scores = alpha * minmax(dense_scores) + (1.0 - alpha) * minmax(bm25_scores)
        ranked = hybrid_scores.argsort()[::-1][:20]
        candidates = [case_by_id[case_ids[int(index)]] for index in ranked]
        row = {
            "qid": question["qid"],
            "target_case_id": question["case_id"],
            "base_case_ids": [case_ids[int(index)] for index in ranked],
            "base_scores": [float(hybrid_scores[int(index)]) for index in ranked],
        }
        base_rows.append(row)
        for candidate in candidates[:candidate_depth]:
            flat_pairs.append((str(question["question"]), case_document(candidate)))
            pair_keys.append((str(question["qid"]), str(candidate["case_id"])))

    reranker = MedCPTReranker(
        str(reranker_config["model"]),
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=True,
    )
    reranker_scores = reranker.score(flat_pairs)
    score_by_pair = {
        key: float(score) for key, score in zip(pair_keys, reranker_scores, strict=True)
    }

    decisions: list[dict[str, Any]] = []
    hybrid_rankings: dict[str, list[str]] = {}
    reranked_rankings: dict[str, list[str]] = {}
    adaptive_rankings: dict[str, list[str]] = {}
    for row in base_rows:
        qid = str(row["qid"])
        base_ids = row["base_case_ids"]
        candidate_ids = base_ids[:candidate_depth]
        reranked_ids = sorted(
            candidate_ids, key=lambda case_id: score_by_pair[(qid, case_id)], reverse=True
        )
        scores = [score_by_pair[(qid, case_id)] for case_id in reranked_ids]
        decision = select_adaptive_top1(
            base_case_ids=base_ids,
            base_scores=row["base_scores"],
            reranked_case_ids=reranked_ids,
            reranker_scores=scores,
            **policy,
        )
        decision_row = {
            "qid": qid,
            "target_case_id": row["target_case_id"],
            "selected_case_id": decision.selected_case_id,
            "correct": decision.selected_case_id == row["target_case_id"],
            **decision.__dict__,
            "base_case_ids": base_ids,
            "base_scores": row["base_scores"],
            "reranked_case_ids": reranked_ids,
            "reranker_scores": scores,
        }
        decisions.append(decision_row)
        hybrid_rankings[qid] = base_ids
        reranked_rankings[qid] = reranked_ids + base_ids[candidate_depth:]
        adaptive_rankings[qid] = [decision.selected_case_id] if decision.selected_case_id else []

    qrels = {str(row["qid"]): {str(row["case_id"])} for row in questions}
    answered = [row for row in decisions if not row["abstained"]]
    retrieval_summary = {
        "hybrid": evaluate_retrieval(qrels, hybrid_rankings, k_values=(1, 3, 5, 10, 20)),
        "reranked": evaluate_retrieval(qrels, reranked_rankings, k_values=(1, 3, 5, 10, 20)),
        "adaptive": {
            **evaluate_retrieval(qrels, adaptive_rankings, k_values=(1,)),
            "coverage": len(answered) / len(decisions),
            "selective_accuracy": mean(bool(row["correct"]) for row in answered),
            "abstention_rate": mean(bool(row["abstained"]) for row in decisions),
            "agreement_rate": mean(row["source"] == "agreement" for row in decisions),
            "hybrid_selection_rate": mean(row["source"] == "hybrid" for row in decisions),
            "reranker_selection_rate": mean(row["source"] == "reranker" for row in decisions),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = args.output_dir / "adaptive_decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as handle:
        for row in decisions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    prompts_path = args.output_dir / "direct_prompt_pack.jsonl"
    question_by_qid = {str(row["qid"]): row for row in questions}
    with prompts_path.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            question = question_by_qid[str(decision["qid"])]
            selected_id = decision["selected_case_id"]
            record = {
                **question,
                "system": "locked_adaptive_case_rag_replication",
                "prompt_mode": "direct",
                "retriever": "locked_adaptive_hybrid_medcpt_reranker",
                "retrieved_case_ids": [selected_id] if selected_id else [],
                "retrieval_abstained": bool(decision["abstained"]),
                "prompt": _prompt(question, case_by_id.get(selected_id) if selected_id else None),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "experiment": "locked_untouched_openi_replication",
        "status": "retrieval_complete_generation_pending",
        "no_replication_tuning": True,
        "cohort": {key: cohort[key] for key in cohort if key != "questions"},
        "excluded_case_count": len(excluded),
        "locked_configuration": {
            "hybrid_alpha": alpha,
            "reranker_model": reranker_config["model"],
            "reranker_candidate_depth": candidate_depth,
            "adaptive_policy": policy,
            "source_hashes_sha256": {
                str(hybrid_config_path.relative_to(ROOT)): _sha256(hybrid_config_path),
                str(reranker_config_path.relative_to(ROOT)): _sha256(reranker_config_path),
                str(policy_config_path.relative_to(ROOT)): _sha256(policy_config_path),
            },
        },
        "retrieval": retrieval_summary,
        "decisions_path": str(decisions_path.relative_to(ROOT)),
        "prompt_pack": str(prompts_path.relative_to(ROOT)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
