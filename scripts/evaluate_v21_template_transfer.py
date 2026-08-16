from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from medical_rag.agentic.closed_loop_agent import ClosedLoopEvidenceAgent
from medical_rag.retrieval.scoped_chunk_retriever import ScopedBM25ChunkRetriever
from scripts.evaluate_case_scoped_hard_v21 import _agent_row, _system_metrics


OUT_OF_SCOPE_PARAPHRASES = {
    "troponin": "Provide the post-imaging cardiac enzyme result.",
    "medication": "Name any medicine given after imaging.",
    "pathology": "Report the tissue analysis result.",
    "discharge": "Describe subsequent management after the patient left.",
    "hounsfield": "Give the numerical density measurement from cross-sectional imaging.",
    "oxygen saturation": "State the post-intervention blood oxygen reading.",
}


def transfer_question(row: dict[str, Any]) -> str:
    qid = str(row["qid"])
    original = str(row["question"])
    if qid.endswith("_v21_observation"):
        return "Detail the visual evidence noted by the reporting clinician."
    if qid.endswith("_v21_conclusion"):
        return "State the reporting clinician's overall interpretation."
    if qid.endswith("_v21_fact_probe") or qid.endswith(
        "_v21_near_domain_negative"
    ):
        match = re.search(r"about (.+?)\?$", original, flags=re.IGNORECASE)
        subject = match.group(1) if match else "the requested condition"
        return f"Regarding {subject}, what information is available?"
    lowered = original.lower()
    for keyword, paraphrase in OUT_OF_SCOPE_PARAPHRASES.items():
        if keyword in lowered:
            return paraphrase
    return "Give the requested post-examination clinical information."


def family(qid: str) -> str:
    for suffix in (
        "observation",
        "conclusion",
        "fact_probe",
        "unanswerable_a",
        "unanswerable_b",
        "near_domain_negative",
    ):
        if qid.endswith(suffix):
            return suffix
    raise ValueError(f"Unknown v2.1 question family: {qid}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen Agent wording transfer without tuning."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_scoped_hard_v21.json",
    )
    parser.add_argument(
        "--v21-summary",
        type=Path,
        default=ROOT / "experiments" / "post_submission_v21" / "summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "post_submission_v21" / "template_transfer",
    )
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    prior_summary = json.loads(args.v21_summary.read_text(encoding="utf-8"))
    threshold = float(
        prior_summary["threshold_selection"]["closed_loop_agent_v2"]["threshold"]
    )
    source_rows = [
        row for row in benchmark["questions"] if row["split"] == "test"
    ]
    stress_questions = [
        {
            **row,
            "qid": f"{row['qid']}_transfer",
            "source_qid": row["qid"],
            "source_question": row["question"],
            "question": transfer_question(row),
        }
        for row in source_rows
    ]
    if any(row["question"] == row["source_question"] for row in stress_questions):
        raise ValueError("Every transfer question must use unseen wording.")

    retriever = ScopedBM25ChunkRetriever().fit(benchmark["chunks"])
    agent = ClosedLoopEvidenceAgent(retriever, first_pass_k=3, retry_k=3)
    rows = [
        _agent_row(question, agent, "closed_loop_agent_v2_template_transfer")
        for question in stress_questions
    ]
    metrics = _system_metrics(rows, threshold)
    by_family = {
        name: _system_metrics(
            [row for row in rows if family(str(row["source_qid"])) == name],
            threshold,
        )
        for name in sorted({family(str(row["source_qid"])) for row in rows})
    }
    route_confusion: dict[str, dict[str, int]] = {}
    error_counts: Counter[str] = Counter()
    error_examples: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        expected = str(row["expected_intent"])
        predicted = str(row["final_intent"])
        route_confusion.setdefault(expected, {}).setdefault(predicted, 0)
        route_confusion[expected][predicted] += 1
        predicts_answer = float(row["answer_probability"]) >= threshold
        retrieved_hit = bool(
            set(row["relevant_chunk_ids"]) & set(row["retrieved_chunk_ids"])
        )
        if not row["is_answerable"]:
            category = "false_answer_unanswerable" if predicts_answer else "correct_abstention"
        elif not predicts_answer:
            category = "missed_answerable"
        elif not retrieved_hit:
            category = "retrieval_miss"
        elif predicted != expected:
            category = "wrong_section_route_with_evidence_hit"
        else:
            category = "correct_answer_action"
        error_counts[category] += 1
        if category not in {"correct_abstention", "correct_answer_action"}:
            examples = error_examples.setdefault(category, [])
            if len(examples) < 5:
                examples.append(
                    {
                        "qid": row["qid"],
                        "question": row["question"],
                        "expected_intent": expected,
                        "final_intent": predicted,
                        "answer_probability": row["answer_probability"],
                        "retrieved_sections": row["retrieved_sections"],
                    }
                )

    canonical = json.dumps(
        [(row["source_qid"], row["question"]) for row in stress_questions],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    output = {
        "experiment": "v21_template_transfer_stress_test",
        "protocol": {
            "cases": "frozen v2.1 test cases only",
            "question_wording": "reserved paraphrases not used by development or calibration",
            "parameter_tuning_after_evaluation": False,
            "answerability_threshold_source": "v2.1 development split",
            "answerability_threshold": threshold,
            "question_count": len(rows),
            "template_fingerprint_sha256": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
        },
        "original_wording_test": prior_summary["systems"]["closed_loop_agent_v2"][
            "test"
        ],
        "transfer_wording_test": metrics,
        "by_question_family": by_family,
        "route_confusion": route_confusion,
        "failure_taxonomy": {
            "counts": dict(sorted(error_counts.items())),
            "examples": error_examples,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
