from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.answer_metrics import extract_final_answer, token_f1


def _load_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        return payload.get("answers", payload.get("records", []))
    if isinstance(payload, list):
        return payload
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _parse_answer_source(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Expected NAME=PATH for --answers, got: {raw}")
    name, path = raw.split("=", 1)
    if not name.strip():
        raise ValueError(f"Missing system name in --answers value: {raw}")
    return name.strip(), Path(path)


def _answer(row: dict) -> str:
    return row.get("final_answer") or extract_final_answer(row.get("answer", ""))


def _agent_support(row: dict) -> float | str:
    agent = row.get("agent", {})
    evidence_check = agent.get("evidence_check", agent)
    value = evidence_check.get("support_rate")
    return value if value is not None else ""


def _unsupported_sentences(row: dict) -> str:
    agent = row.get("agent", {})
    evidence_check = agent.get("evidence_check", agent)
    sentences = evidence_check.get("unsupported_sentences", [])
    return " | ".join(sentences)


def _top1_hit(row: dict) -> bool | str:
    if "top1_hit" in row:
        return row["top1_hit"]
    retrieved = row.get("retrieved_case_ids", [])
    relevant = set(row.get("relevant_case_ids", []))
    if not retrieved or not relevant:
        return ""
    return retrieved[0] in relevant


def _retrieved_hit(row: dict) -> bool | str:
    if "retrieved_hit" in row:
        return row["retrieved_hit"]
    retrieved = set(row.get("retrieved_case_ids", []))
    relevant = set(row.get("relevant_case_ids", []))
    if not retrieved or not relevant:
        return ""
    return bool(retrieved.intersection(relevant))


def _answer_f1(row: dict) -> float | str:
    if "answer_token_f1" in row:
        return row["answer_token_f1"]
    answer = _answer(row)
    reference = row.get("reference_answer", "")
    if not answer or not reference:
        return ""
    return token_f1(answer, reference)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a paired manual annotation CSV across multiple answer files.")
    parser.add_argument("--answers", action="append", required=True, help="System label and file path as NAME=PATH.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", default=50, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    answer_sets: list[tuple[str, dict[str, dict]]] = []
    for raw_source in args.answers:
        name, path = _parse_answer_source(raw_source)
        records = _load_records(path)
        answer_sets.append((name, {record["qid"]: record for record in records}))

    if not answer_sets:
        raise ValueError("No answer sets provided.")

    common_qids = set(answer_sets[0][1])
    for _, records_by_qid in answer_sets[1:]:
        common_qids.intersection_update(records_by_qid)
    if not common_qids:
        raise ValueError("No common qids found across answer files.")

    rng = random.Random(args.seed)
    sampled_qids = sorted(rng.sample(sorted(common_qids), min(args.sample_size, len(common_qids))))

    fieldnames = [
        "qid",
        "case_id",
        "question_type",
        "question",
        "reference_answer",
        "system",
        "system_answer",
        "retrieved_case_ids",
        "top1_hit",
        "retrieved_hit",
        "auto_answer_token_f1",
        "auto_evidence_support_rate",
        "auto_unsupported_sentences",
        "relevance_0_2",
        "evidence_support_0_2",
        "hallucination_control_0_2",
        "completeness_0_2",
        "case_contamination_yes_no",
        "notes",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for qid in sampled_qids:
            for system_name, records_by_qid in answer_sets:
                row = records_by_qid[qid]
                writer.writerow(
                    {
                        "qid": row.get("qid", ""),
                        "case_id": row.get("case_id", ""),
                        "question_type": row.get("question_type", ""),
                        "question": row.get("question", ""),
                        "reference_answer": row.get("reference_answer", ""),
                        "system": system_name,
                        "system_answer": _answer(row),
                        "retrieved_case_ids": " | ".join(row.get("retrieved_case_ids", [])),
                        "top1_hit": _top1_hit(row),
                        "retrieved_hit": _retrieved_hit(row),
                        "auto_answer_token_f1": _answer_f1(row),
                        "auto_evidence_support_rate": _agent_support(row),
                        "auto_unsupported_sentences": _unsupported_sentences(row),
                        "relevance_0_2": "",
                        "evidence_support_0_2": "",
                        "hallucination_control_0_2": "",
                        "completeness_0_2": "",
                        "case_contamination_yes_no": "",
                        "notes": "",
                    }
                )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "questions": len(sampled_qids),
                "systems": len(answer_sets),
                "rows": len(sampled_qids) * len(answer_sets),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
