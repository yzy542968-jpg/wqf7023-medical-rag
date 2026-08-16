from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.metrics import evaluate_retrieval
from medical_rag.retrieval.medcpt_reranker import (
    DEFAULT_RERANKER_MODEL,
    MedCPTReranker,
    case_document,
)
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def evaluate_subset(
    questions: list[dict], rankings: dict[str, list[str]], selected_qids: set[str]
) -> dict[str, float]:
    qrels = {
        str(item["qid"]): {str(value) for value in item["relevant_case_ids"]}
        for item in questions
        if str(item["qid"]) in selected_qids
    }
    return evaluate_retrieval(
        qrels,
        {qid: rankings[qid] for qid in qrels},
        k_values=(1, 3, 5, 10, 20),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MedCPT Cross-Encoder reranking.")
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "data" / "processed" / "openi_cases.jsonl"
    )
    parser.add_argument(
        "--qa",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_qa_seed_clean.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "openi_qa_grouped_case_seed7023.json",
    )
    parser.add_argument(
        "--retrieval-selection",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "retrieval"
        / "hybrid_alpha_selection.json",
    )
    parser.add_argument("--model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidate-depths", nargs="+", type=int, default=[3, 5, 10, 20])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "final_optimized" / "reranking",
    )
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    case_by_id = {str(case["case_id"]): case for case in cases}
    questions = json.loads(args.qa.read_text(encoding="utf-8"))["questions"]
    question_by_qid = {str(item["qid"]): item for item in questions}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    development_qids = set(split["development"]["qids"])
    test_qids = set(split["test"]["qids"])
    retrieval = json.loads(args.retrieval_selection.read_text(encoding="utf-8"))
    base_rankings = {
        str(qid): [str(case_id) for case_id in ranking]
        for qid, ranking in retrieval["selected_rankings"].items()
    }

    max_depth = min(max(args.candidate_depths), len(next(iter(base_rankings.values()))))
    flattened_pairs: list[tuple[str, str]] = []
    pair_keys: list[tuple[str, str]] = []
    for qid, ranking in base_rankings.items():
        query = str(question_by_qid[qid]["question"])
        for case_id in ranking[:max_depth]:
            flattened_pairs.append((query, case_document(case_by_id[case_id])))
            pair_keys.append((qid, case_id))

    reranker = MedCPTReranker(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=False,
    )
    scores = reranker.score(flattened_pairs)
    score_by_pair = {
        pair_key: score for pair_key, score in zip(pair_keys, scores, strict=True)
    }

    development_results = []
    rankings_by_depth: dict[int, dict[str, list[str]]] = {}
    for depth in args.candidate_depths:
        reranked = {}
        for qid, ranking in base_rankings.items():
            candidates = ranking[:depth]
            reordered = sorted(
                candidates,
                key=lambda case_id: score_by_pair[(qid, case_id)],
                reverse=True,
            )
            reranked[qid] = reordered + ranking[depth:]
        rankings_by_depth[depth] = reranked
        development_results.append(
            {
                "candidate_depth": depth,
                "split": "development",
                **evaluate_subset(questions, reranked, development_qids),
            }
        )

    selected = max(
        development_results,
        key=lambda row: (row["mrr"], row["hit@1"], row["hit@5"], -row["candidate_depth"]),
    )
    selected_depth = int(selected["candidate_depth"])
    selected_rankings = rankings_by_depth[selected_depth]
    held_out_test = {
        "candidate_depth": selected_depth,
        "split": "test",
        **evaluate_subset(questions, selected_rankings, test_qids),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "reranker_development_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(development_results[0]))
        writer.writeheader()
        writer.writerows(development_results)

    output = {
        "model": args.model,
        "base_selected_alpha": retrieval["selected_alpha"],
        "selection_rule": "maximize development MRR",
        "selected_candidate_depth": selected_depth,
        "selected_development_metrics": selected,
        "held_out_test_metrics": held_out_test,
        "split_manifest": str(args.split),
        "selected_rankings": selected_rankings,
        "reranker_scores": {
            qid: {
                case_id: score_by_pair[(qid, case_id)]
                for case_id in base_rankings[qid][:max_depth]
            }
            for qid in base_rankings
        },
    }
    json_path = args.output_dir / "medcpt_reranker_selection.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "selected_candidate_depth": selected_depth,
                "selected_development_metrics": selected,
                "held_out_test_metrics": held_out_test,
                "development_sweep_csv": str(csv_path),
                "selection_json": str(json_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
