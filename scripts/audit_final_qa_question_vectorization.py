from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.qa.question_vectorizer import (  # noqa: E402
    RadReStructQuestionVectorizer,
)
from medical_rag.qa.radrestruct import iter_radrestruct_cases  # noqa: E402
from medical_rag.qa.radrestruct_hierarchy import RadReStructHierarchy  # noqa: E402
from medical_rag.qa.structured_metrics import (  # noqa: E402
    load_answer_vector,
    load_report_keys,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    hierarchy = RadReStructHierarchy(args.radrestruct_root)
    vectorizer = RadReStructQuestionVectorizer(hierarchy)
    report_keys = load_report_keys(args.radrestruct_root)
    case_count = 0
    question_count = 0
    mismatched_cases: list[dict[str, object]] = []
    question_ids_seen: set[int] = set()
    for case in iter_radrestruct_cases(args.radrestruct_root):
        reconstructed = vectorizer.vectorize_answers(case.questions)
        reference_path = (
            args.radrestruct_root
            / f"{case.official_split}_vectorized_answers"
            / f"{case.source_report_id}.json"
        )
        reference = load_answer_vector(reference_path, report_keys)
        different = int(np.count_nonzero(reconstructed != reference))
        if different:
            mismatched_cases.append(
                {
                    "case_id": case.case_id,
                    "official_split": case.official_split,
                    "different_elements": different,
                }
            )
        question_ids_seen.update(vectorizer.question_ids(case.questions))
        case_count += 1
        question_count += len(case.questions)

    result = {
        "study": "Final QA ordered-question vectorization audit",
        "radrestruct_commit": "b293158f0c5c1c5fa27dd615c28005eb54d7b1de",
        "report_keys_sha256": _sha256(args.radrestruct_root / "report_keys.json"),
        "case_count": case_count,
        "question_count": question_count,
        "question_ids_seen": len(question_ids_seen),
        "report_vector_dimensions": len(report_keys),
        "mismatched_case_count": len(mismatched_cases),
        "mismatched_cases_preview": mismatched_cases[:20],
        "all_exact": not mismatched_cases,
        "input_boundary": (
            "Each QA row is mapped from its own text/options/path and order. Gold history "
            "is not an input. This audit uses reference answers only to verify mapping."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if mismatched_cases:
        raise RuntimeError(
            f"Question vectorization mismatched {len(mismatched_cases)} cases"
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radrestruct-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments/final_qa_development/question_vectorization_audit.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
