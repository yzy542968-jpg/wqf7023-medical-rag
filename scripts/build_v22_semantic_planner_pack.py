from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_v21_template_transfer import transfer_question


PLANNER_INSTRUCTION = """Classify a question about one known chest-radiograph report.

Return exactly one label and no explanation:
FINDINGS - asks for visual observations or abnormalities in the examination.
IMPRESSION - asks for the radiologist's conclusion, interpretation, assessment, or report summary.
REPORT_FACT - asks what the report says about one named radiologic condition or finding.
OUTSIDE_REPORT - asks for laboratory results, medication, pathology, treatment, vital signs, or other information not normally contained in a radiology report.

Question: {question}
Label:"""


def planner_prompt(question: str) -> str:
    return PLANNER_INSTRUCTION.format(question=question)


def _record(row: dict[str, Any], *, transfer: bool) -> dict[str, Any]:
    question = transfer_question(row) if transfer else str(row["question"])
    qid = f"{row['qid']}_transfer" if transfer else str(row["qid"])
    return {
        "qid": qid,
        "case_id": row["case_id"],
        "question_type": "semantic_route_classification",
        "system": "v22_constrained_semantic_planner",
        "prompt_mode": "zero_shot_constrained_route",
        "retriever": "none_planning_only",
        "question": question,
        "reference_answer": row["expected_intent"],
        "relevant_case_ids": [row["case_id"]],
        "retrieved_case_ids": [],
        "source_qid": row["qid"],
        "source_split": row["split"],
        "transfer": transfer,
        "prompt": planner_prompt(question),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen V2.2 semantic planner pack.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_scoped_hard_v21.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "data"
        / "processed"
        / "prompt_packs"
        / "v22_semantic_planner.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "experiments"
        / "post_submission_v22"
        / "planner_pack_manifest.json",
    )
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    original = [_record(row, transfer=False) for row in benchmark["questions"]]
    transfer = [
        _record(row, transfer=True)
        for row in benchmark["questions"]
        if row["split"] == "test"
    ]
    records = original + transfer
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "experiment": "v22_constrained_semantic_planner",
        "planner_prompt_sha256": hashlib.sha256(
            PLANNER_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "prompt_frozen_before_generation": True,
        "record_count": len(records),
        "original_record_count": len(original),
        "transfer_record_count": len(transfer),
        "development_use": "answerability threshold selection only",
        "calibration_use": "Platt probability calibration only",
        "test_use": "evaluation only",
        "post_test_prompt_changes_permitted": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
