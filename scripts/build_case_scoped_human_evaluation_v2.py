from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.action_policy import apply_verifier_action


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a blinded V2 confirmation human-evaluation package.")
    parser.add_argument("--seed", type=int, default=27023)
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROOT / "experiments" / "benchmark_v2" / "confirmation_evaluation" / "test_generation_rows.jsonl",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_scoped_confirmation_v2.json",
    )
    parser.add_argument(
        "--prompt-pack",
        type=Path,
        default=ROOT / "data" / "processed" / "prompt_packs" / "benchmark_v2" / "confirmation_case_scoped_routed.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "benchmark_v2" / "human_evaluation",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(args.rows)
    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    prompt_by_qid = {row["qid"]: row for row in read_jsonl(args.prompt_pack)}
    chunk_by_id = {row["chunk_id"]: row for row in benchmark["chunks"]}
    by_type = {}
    for row in rows:
        by_type.setdefault(row["question_type"], []).append(row)
    sampled = []
    used_case_ids: set[str] = set()
    for question_type, candidates in sorted(by_type.items()):
        rng.shuffle(candidates)
        selected = [row for row in candidates if row["case_id"] not in used_case_ids][:12]
        if len(selected) != 12:
            raise ValueError(f"Could not sample 12 distinct cases for {question_type}.")
        sampled.extend(selected)
        used_case_ids.update(str(row["case_id"]) for row in selected)
    rng.shuffle(sampled)

    systems = ["advisory_qwen", "extractive_evidence", "sentence_filter", "contradiction_only"]
    labels = "ABCD"
    annotation_rows = []
    key_rows = []
    for index, row in enumerate(sampled, start=1):
        prompt = prompt_by_qid[row["qid"]]
        checks = row["sentence_checks"]
        sentence_filter = apply_verifier_action(
            row["draft_answer"], checks, action_policy="sentence_filter", contradiction_threshold=0.5
        ).answer
        contradiction_only = apply_verifier_action(
            row["draft_answer"], checks, action_policy="contradiction_only", contradiction_threshold=0.9
        ).answer
        extractive = " ".join(
            chunk_by_id[chunk_id]["text"] for chunk_id in prompt["retrieved_chunk_ids"]
        )
        answers = {
            "advisory_qwen": row["draft_answer"],
            "extractive_evidence": extractive,
            "sentence_filter": sentence_filter,
            "contradiction_only": contradiction_only,
        }
        shuffled = systems.copy()
        rng.shuffle(shuffled)
        sample_id = f"V2HE{index:03d}"
        annotation = {
            "sample_id": sample_id,
            "question_type": row["question_type"],
            "question": row["question"],
            "reference_answer": row["reference_answer"],
            "retrieved_case_evidence": prompt["retrieved_context"],
        }
        for label, system in zip(labels, shuffled, strict=True):
            lower = label.lower()
            annotation[f"response_{lower}"] = answers[system]
            annotation[f"{lower}_correctness_1_5"] = ""
            annotation[f"{lower}_evidence_grounding_1_5"] = ""
            annotation[f"{lower}_potentially_harmful_0_1"] = ""
            key_rows.append(
                {
                    "sample_id": sample_id,
                    "qid": row["qid"],
                    "case_id": row["case_id"],
                    "response_label": label,
                    "system": system,
                }
            )
        annotation["best_response_A_B_C_D_or_tie"] = ""
        annotation["reviewer_notes"] = ""
        annotation_rows.append(annotation)

    fields = ["sample_id", "question_type", "question", "reference_answer", "retrieved_case_evidence"]
    for lower in "abcd":
        fields.extend(
            [
                f"response_{lower}",
                f"{lower}_correctness_1_5",
                f"{lower}_evidence_grounding_1_5",
                f"{lower}_potentially_harmful_0_1",
            ]
        )
    fields.extend(["best_response_A_B_C_D_or_tie", "reviewer_notes"])
    annotation_path = args.output_dir / "v2_confirmation_blinded_human_evaluation_36.csv"
    key_path = args.output_dir / "v2_confirmation_blinded_human_evaluation_key.csv"
    write_csv(annotation_path, annotation_rows, fields)
    write_csv(key_path, key_rows, ["sample_id", "qid", "case_id", "response_label", "system"])
    print(
        json.dumps(
            {
                "samples": len(annotation_rows),
                "responses": len(key_rows),
                "question_type_counts": {
                    key: sum(row["question_type"] == key for row in sampled) for key in sorted(by_type)
                },
                "annotation_file": str(annotation_path),
                "blinding_key": str(key_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
