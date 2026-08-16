from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from medical_rag.evaluation.answer_metrics import extract_final_answer
from scripts.run_grouped_statistical_analysis import grouped_bootstrap_ci


GENERATION_SYSTEMS = {
    "llm_only": ROOT / "experiments" / "generations_llm_only_qwen15_full360.jsonl",
    "report_bm25_draft": ROOT / "experiments" / "generations_report_rag_bm25_qwen15_full360.jsonl",
    "case_bm25_draft": ROOT
    / "experiments"
    / "generations_case_rag_bm25_top1_qwen15_full360.jsonl",
    "case_hybrid_a050_draft": ROOT
    / "experiments"
    / "generations_case_rag_hybrid_top1_qwen15_full360.jsonl",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out answers with F1-RadGraph.")
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "openi_qa_grouped_case_seed7023.json",
    )
    parser.add_argument(
        "--semantic-test-rows",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "semantic_agent"
        / "semantic_agent_selected_test_rows.jsonl",
    )
    parser.add_argument(
        "--final-test-rows",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "final_test"
        / "final_optimized_test_rows.jsonl",
    )
    parser.add_argument("--model-type", default="modern-radgraph-xl")
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7023)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "final_optimized" / "radgraph",
    )
    args = parser.parse_args()

    from radgraph import F1RadGraph

    split = json.loads(args.split.read_text(encoding="utf-8"))
    test_qids = set(split["test"]["qids"])
    rows = []
    for system, path in GENERATION_SYSTEMS.items():
        for row in read_jsonl(path):
            if str(row["qid"]) not in test_qids:
                continue
            rows.append(
                {
                    "system": system,
                    "qid": str(row["qid"]),
                    "case_id": str(row["case_id"]),
                    "prediction": extract_final_answer(row.get("answer", "")),
                    "reference": str(row.get("reference_answer", "")),
                }
            )
    for row in read_jsonl(args.semantic_test_rows):
        rows.append(
            {
                "system": f"{row['system']}_semantic_agent",
                "qid": str(row["qid"]),
                "case_id": str(row["case_id"]),
                "prediction": str(row["final_answer"]),
                "reference": str(row["reference_answer"]),
            }
        )
    if args.final_test_rows.exists():
        for row in read_jsonl(args.final_test_rows):
            rows.append(
                {
                    "system": "final_adaptive_direct_draft",
                    "qid": str(row["qid"]),
                    "case_id": str(row["case_id"]),
                    "prediction": str(row["draft_answer"]),
                    "reference": str(row["reference_answer"]),
                }
            )
            rows.append(
                {
                    "system": "final_adaptive_direct_semantic_agent",
                    "qid": str(row["qid"]),
                    "case_id": str(row["case_id"]),
                    "prediction": str(row["final_answer"]),
                    "reference": str(row["reference_answer"]),
                }
            )

    scorer = F1RadGraph(reward_level="all", model_type=args.model_type)
    _, reward_lists, _, _ = scorer(
        hyps=[row["prediction"] for row in rows],
        refs=[row["reference"] for row in rows],
    )
    entity_scores, entity_relation_scores, complete_scores = reward_lists
    for row, entity, entity_relation, complete in zip(
        rows,
        entity_scores,
        entity_relation_scores,
        complete_scores,
        strict=True,
    ):
        row["radgraph_entity_f1"] = float(entity)
        row["radgraph_entity_relation_f1"] = float(entity_relation)
        row["radgraph_complete_f1"] = float(complete)

    metric_columns = [
        "radgraph_entity_f1",
        "radgraph_entity_relation_f1",
        "radgraph_complete_f1",
    ]
    summary = []
    systems = sorted({row["system"] for row in rows})
    for system_index, system in enumerate(systems):
        system_rows = [row for row in rows if row["system"] == system]
        output = {
            "system": system,
            "case_count": len({row["case_id"] for row in system_rows}),
            "question_count": len(system_rows),
        }
        for metric_index, metric in enumerate(metric_columns):
            by_case: dict[str, list[float]] = defaultdict(list)
            for row in system_rows:
                by_case[row["case_id"]].append(float(row[metric]))
            observed, low, high = grouped_bootstrap_ci(
                dict(by_case),
                iterations=args.bootstrap_iterations,
                seed=args.seed + system_index * 10 + metric_index,
            )
            output[f"{metric}_mean"] = observed
            output[f"{metric}_ci_low"] = low
            output[f"{metric}_ci_high"] = high
        summary.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "held_out_radgraph_per_answer.csv"
    with rows_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = args.output_dir / "held_out_radgraph_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    output = {
        "model_type": args.model_type,
        "split_manifest": str(args.split),
        "rows": len(rows),
        "summary": summary,
        "per_answer_csv": str(rows_path),
        "summary_csv": str(summary_path),
    }
    json_path = args.output_dir / "held_out_radgraph_summary.json"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
