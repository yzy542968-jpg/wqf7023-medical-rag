from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.splits import build_grouped_case_split, filter_questions_for_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Build case-disjoint development/test QA splits by case ID.")
    parser.add_argument(
        "--qa",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_qa_seed_clean.json",
    )
    parser.add_argument("--development-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=7023)
    parser.add_argument(
        "--split-output",
        type=Path,
        default=ROOT / "data" / "splits" / "openi_qa_grouped_case_seed7023.json",
    )
    parser.add_argument(
        "--development-output",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_qa_seed_clean_development.json",
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_case_qa_seed_clean_test.json",
    )
    args = parser.parse_args()

    payload = json.loads(args.qa.read_text(encoding="utf-8"))
    questions = payload["questions"]
    split = build_grouped_case_split(
        questions,
        development_fraction=args.development_fraction,
        seed=args.seed,
    )
    split["source_qa"] = str(args.qa)

    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    args.split_output.write_text(json.dumps(split, indent=2), encoding="utf-8")

    outputs = {
        "development": args.development_output,
        "test": args.test_output,
    }
    for split_name, output_path in outputs.items():
        selected = filter_questions_for_split(questions, split, split_name)
        selected_payload = {
            **{key: value for key, value in payload.items() if key != "questions"},
            "parent_qa": str(args.qa),
            "split_manifest": str(args.split_output),
            "split_name": split_name,
            "case_count": split[split_name]["case_count"],
            "question_count": len(selected),
            "questions": selected,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(selected_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "split_manifest": str(args.split_output),
                "development": {
                    "output": str(args.development_output),
                    **{
                        key: split["development"][key]
                        for key in ("case_count", "question_count", "question_type_counts")
                    },
                },
                "test": {
                    "output": str(args.test_output),
                    **{
                        key: split["test"][key]
                        for key in ("case_count", "question_count", "question_type_counts")
                    },
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
