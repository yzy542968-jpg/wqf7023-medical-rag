from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def _placeholder_ratio(text: str) -> float:
    tokens = _clean(text).split()
    if not tokens:
        return 1.0
    return sum(1 for token in tokens if "XXXX" in token.upper()) / len(tokens)


def _eligible(case: dict[str, Any], min_findings_chars: int, min_impression_chars: int, require_images: bool) -> bool:
    if require_images and not case.get("images"):
        return False
    findings = _clean(case.get("findings", ""))
    impression = _clean(case.get("impression", ""))
    if len(findings) < min_findings_chars:
        return False
    if len(impression) < min_impression_chars:
        return False
    return True


def _problem_hint(case: dict[str, Any]) -> str:
    problems = _clean(case.get("problems", ""))
    if not problems:
        return "the main radiology finding"
    return problems.replace(";", ", ")


def _question_items(case: dict[str, Any]) -> list[dict[str, Any]]:
    case_id = case["case_id"]
    indication = _clean(case.get("indication", "")) or "not provided"
    findings = _clean(case.get("findings", ""))
    impression = _clean(case.get("impression", ""))
    problems = _clean(case.get("problems", ""))
    problem_hint = _problem_hint(case)

    return [
        {
            "qid": f"{case_id}_impression",
            "case_id": case_id,
            "question_type": "impression_from_indication",
            "question": f"For a chest X-ray case with the indication '{indication}', what is the radiology impression?",
            "reference_answer": impression,
            "answer_source": "impression",
            "relevant_case_ids": [case_id],
            "problems": problems,
            "images": case.get("images", []),
        },
        {
            "qid": f"{case_id}_findings",
            "case_id": case_id,
            "question_type": "findings_from_indication",
            "question": f"For a chest X-ray case with the indication '{indication}', what are the main report findings?",
            "reference_answer": findings,
            "answer_source": "findings",
            "relevant_case_ids": [case_id],
            "problems": problems,
            "images": case.get("images", []),
        },
        {
            "qid": f"{case_id}_summary",
            "case_id": case_id,
            "question_type": "abnormality_summary",
            "question": f"What does the chest X-ray report say about {problem_hint}?",
            "reference_answer": impression if impression else findings,
            "answer_source": "impression_or_findings",
            "relevant_case_ids": [case_id],
            "problems": problems,
            "images": case.get("images", []),
        },
    ]


def build_qa_seed(
    cases: list[dict[str, Any]],
    max_cases: int,
    seed: int,
    min_findings_chars: int,
    min_impression_chars: int,
    require_images: bool,
    clean_only: bool = False,
) -> list[dict[str, Any]]:
    eligible = [
        case
        for case in cases
        if _eligible(case, min_findings_chars, min_impression_chars, require_images)
    ]
    if clean_only:
        eligible = [
            case
            for case in eligible
            if _placeholder_ratio(case.get("indication", "")) <= 0.5
            and _clean(case.get("problems", "")).lower() not in {"", "normal"}
        ]
    random.Random(seed).shuffle(eligible)
    selected_cases = sorted(eligible[:max_cases], key=lambda case: case["case_id"])

    questions: list[dict[str, Any]] = []
    for case in selected_cases:
        questions.extend(_question_items(case))
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic case-grounded QA seed set from OpenI reports.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-cases", default=120, type=int)
    parser.add_argument("--seed", default=7023, type=int)
    parser.add_argument("--min-findings-chars", default=40, type=int)
    parser.add_argument("--min-impression-chars", default=8, type=int)
    parser.add_argument("--allow-missing-images", action="store_true")
    parser.add_argument("--clean-only", action="store_true")
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    questions = build_qa_seed(
        cases=cases,
        max_cases=args.max_cases,
        seed=args.seed,
        min_findings_chars=args.min_findings_chars,
        min_impression_chars=args.min_impression_chars,
        require_images=not args.allow_missing_images,
        clean_only=args.clean_only,
    )

    payload = {
        "source_cases": str(args.cases),
        "seed": args.seed,
        "case_count": len({item["case_id"] for item in questions}),
        "question_count": len(questions),
        "questions": questions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"case_count": payload["case_count"], "question_count": payload["question_count"]}, indent=2))


if __name__ == "__main__":
    main()
