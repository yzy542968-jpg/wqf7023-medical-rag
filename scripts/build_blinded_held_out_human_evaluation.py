from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.answer_metrics import extract_final_answer


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a system-blinded human evaluation package on held-out cases."
    )
    parser.add_argument("--seed", type=int, default=7023)
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "openi_qa_grouped_case_seed7023.json",
    )
    parser.add_argument(
        "--final-rows",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "final_test"
        / "final_optimized_test_rows.jsonl",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_cases.jsonl",
    )
    parser.add_argument(
        "--semantic-rows",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "semantic_agent"
        / "semantic_agent_selected_test_rows.jsonl",
    )
    parser.add_argument(
        "--llm-only",
        type=Path,
        default=ROOT / "experiments" / "generations_llm_only_qwen15_full360.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "final_optimized" / "human_evaluation",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    test_case_ids = [str(value) for value in split["test"]["case_ids"]]
    rng.shuffle(test_case_ids)

    final_rows = {str(row["qid"]): row for row in read_jsonl(args.final_rows)}
    cases = {str(row["case_id"]): row for row in read_jsonl(args.cases)}
    semantic_rows = {
        (str(row["system"]), str(row["qid"])): row
        for row in read_jsonl(args.semantic_rows)
    }
    llm_rows = {str(row["qid"]): row for row in read_jsonl(args.llm_only)}
    final_by_case_type = {
        (str(row["case_id"]), str(row["qid"]).rsplit("_", 1)[-1]): row
        for row in final_rows.values()
    }

    question_types = ["findings", "impression", "summary"] * 12
    rng.shuffle(question_types)
    systems = [
        "final_adaptive_direct_semantic_agent",
        "case_hybrid_top1_a050_semantic_agent",
        "case_bm25_top1_semantic_agent",
        "llm_only",
    ]
    answer_labels = ["A", "B", "C", "D"]
    annotation_rows: list[dict] = []
    key_rows: list[dict] = []

    for sample_index, (case_id, question_type) in enumerate(
        zip(test_case_ids, question_types, strict=True), start=1
    ):
        base = final_by_case_type[(case_id, question_type)]
        qid = str(base["qid"])
        answers = {
            "final_adaptive_direct_semantic_agent": str(base["final_answer"]),
            "case_hybrid_top1_a050_semantic_agent": str(
                semantic_rows[("case_hybrid_top1_a050", qid)]["final_answer"]
            ),
            "case_bm25_top1_semantic_agent": str(
                semantic_rows[("case_bm25_top1", qid)]["final_answer"]
            ),
            "llm_only": extract_final_answer(str(llm_rows[qid]["answer"])),
        }
        shuffled_systems = systems.copy()
        rng.shuffle(shuffled_systems)
        sample_id = f"HE{sample_index:03d}"
        row = {
            "sample_id": sample_id,
            "question_type": base["question_type"],
            "question": base["question"],
            "reference_answer": base["reference_answer"],
            "gold_case_evidence": "\n".join(
                [
                    f"Findings: {cases[case_id].get('findings', '')}",
                    f"Impression: {cases[case_id].get('impression', '')}",
                ]
            ),
        }
        for label, system in zip(answer_labels, shuffled_systems, strict=True):
            lower = label.lower()
            row[f"response_{lower}"] = answers[system]
            row[f"{lower}_correctness_1_5"] = ""
            row[f"{lower}_evidence_grounding_1_5"] = ""
            row[f"{lower}_potentially_harmful_0_1"] = ""
            key_rows.append(
                {
                    "sample_id": sample_id,
                    "qid": qid,
                    "case_id": case_id,
                    "response_label": label,
                    "system": system,
                }
            )
        row["best_response_A_B_C_D_or_tie"] = ""
        row["reviewer_notes"] = ""
        annotation_rows.append(row)

    annotation_fields = [
        "sample_id",
        "question_type",
        "question",
        "reference_answer",
        "gold_case_evidence",
    ]
    for label in [value.lower() for value in answer_labels]:
        annotation_fields.extend(
            [
                f"response_{label}",
                f"{label}_correctness_1_5",
                f"{label}_evidence_grounding_1_5",
                f"{label}_potentially_harmful_0_1",
            ]
        )
    annotation_fields.extend(["best_response_A_B_C_D_or_tie", "reviewer_notes"])
    annotation_path = args.output_dir / "held_out_blinded_human_evaluation_36.csv"
    key_path = args.output_dir / "held_out_blinded_human_evaluation_key.csv"
    write_csv(annotation_path, annotation_rows, annotation_fields)
    write_csv(
        key_path,
        key_rows,
        ["sample_id", "qid", "case_id", "response_label", "system"],
    )
    print(
        json.dumps(
            {
                "samples": len(annotation_rows),
                "responses": len(key_rows),
                "question_type_counts": {
                    value: question_types.count(value) for value in sorted(set(question_types))
                },
                "annotation_file": str(annotation_path),
                "blinding_key": str(key_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
