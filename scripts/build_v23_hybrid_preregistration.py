from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_v22_semantic_planner_pack import PLANNER_INSTRUCTION, planner_prompt
from scripts.evaluate_v21_template_transfer import transfer_question


POLICY_ID = "lexical_known_forms_else_semantic_v1"
POLICY_DESCRIPTION = (
    "Use the lexical route when infer_report_intent is not unknown; preserve the "
    "development-known 'what does this report state about' report-fact frame; "
    "otherwise use the frozen V2.2 semantic route."
)

OBSERVATION_TEMPLATES = (
    "What did the image appearance reveal?",
    "Summarize the radiographic observations in the study.",
    "Which visible features were documented on inspection of the film?",
)
CONCLUSION_TEMPLATES = (
    "How did the reader synthesize the study?",
    "State the diagnostic assessment made by the reader.",
    "Give the reader's overall judgment.",
)
FACT_TEMPLATES = (
    "Does the account address {subject}, and if so how?",
    "What, if anything, is documented concerning {subject}?",
    "Locate any statement concerning {subject}.",
)
OUTSIDE_REPORT_TEMPLATES = {
    "troponin": "What was the cardiac biomarker value after the study?",
    "medication": "Which drug was administered afterward?",
    "pathology": "What did the tissue examination determine?",
    "discharge": "What follow-up disposition was arranged?",
    "hounsfield": "What numerical CT density was measured?",
    "oxygen saturation": "What was the pulse oximetry reading after care?",
}


def _variant(qid: str, count: int) -> int:
    digest = hashlib.sha256(qid.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % count


def second_transfer_question(row: dict[str, Any]) -> str:
    qid = str(row["qid"])
    original = str(row["question"])
    if qid.endswith("_v21_observation"):
        return OBSERVATION_TEMPLATES[_variant(qid, len(OBSERVATION_TEMPLATES))]
    if qid.endswith("_v21_conclusion"):
        return CONCLUSION_TEMPLATES[_variant(qid, len(CONCLUSION_TEMPLATES))]
    if qid.endswith("_v21_fact_probe") or qid.endswith(
        "_v21_near_domain_negative"
    ):
        match = re.search(r"about (.+?)\?$", original, flags=re.IGNORECASE)
        subject = match.group(1) if match else "the requested condition"
        template = FACT_TEMPLATES[_variant(qid, len(FACT_TEMPLATES))]
        return template.format(subject=subject)
    lowered = original.lower()
    for keyword, paraphrase in OUTSIDE_REPORT_TEMPLATES.items():
        if keyword in lowered:
            return paraphrase
    raise ValueError(f"Unsupported question family: {qid}")


def build_records(benchmark: dict[str, Any]) -> list[dict[str, Any]]:
    source_rows = [
        row for row in benchmark["questions"] if row["split"] == "test"
    ]
    records = []
    for row in source_rows:
        question = second_transfer_question(row)
        if question in {str(row["question"]), transfer_question(row)}:
            raise ValueError(f"Second transfer wording is not reserved: {row['qid']}")
        records.append(
            {
                "qid": f"{row['qid']}_transfer2",
                "case_id": row["case_id"],
                "question_type": "semantic_route_classification",
                "system": "v23_preregistered_hybrid_planner",
                "prompt_mode": "zero_shot_constrained_route_v22_frozen",
                "retriever": "none_planning_only",
                "question": question,
                "reference_answer": row["expected_intent"],
                "relevant_case_ids": [row["case_id"]],
                "retrieved_case_ids": [],
                "source_qid": row["qid"],
                "source_split": row["split"],
                "transfer": "reserved_wording_set_2",
                "prompt": planner_prompt(question),
            }
        )
    return records


def _template_fingerprint(records: list[dict[str, Any]]) -> str:
    canonical = json.dumps(
        [(row["source_qid"], row["question"]) for row in records],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze V2.3 hybrid policy and second reserved wording set."
    )
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
        / "v23_hybrid_transfer2_planner.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "experiments"
        / "post_submission_v23"
        / "preregistration_manifest.json",
    )
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    records = build_records(benchmark)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "experiment": "v23_preregistered_hybrid_planner",
        "policy_id": POLICY_ID,
        "policy_description": POLICY_DESCRIPTION,
        "policy_frozen_before_generation": True,
        "semantic_planner_prompt_sha256": hashlib.sha256(
            PLANNER_INSTRUCTION.encode("utf-8")
        ).hexdigest(),
        "second_transfer_template_sha256": _template_fingerprint(records),
        "record_count": len(records),
        "case_count": len({row["case_id"] for row in records}),
        "source_split": "v2.1 test cases with independently reserved wording set 2",
        "answerability_threshold_source": "frozen v2.1 development selection",
        "calibration_source": "frozen v2.1 calibration-only Platt model",
        "semantic_planner_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "test_or_transfer_tuning": False,
        "post_evaluation_policy_changes_permitted": False,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "prompt_pack": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
