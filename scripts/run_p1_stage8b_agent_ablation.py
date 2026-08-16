from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.evidence_checker import NEGATION_RE, check_evidence_support


DEFAULT_P1_ROOT = Path(r"C:\Users\yz542_dntjhas\Documents\New project 2\radiology-rag")
ABSTENTION = "The retrieved report evidence is insufficient to answer this question."


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evidence_text(row: dict) -> str:
    return "\n".join(
        str(unit.get("retrieved_text", ""))
        for unit in row.get("retrieved_units", [])
        if unit.get("retrieved_text")
    )


def is_cannot_determine(text: str) -> bool:
    lowered = (text or "").lower()
    return any(
        phrase in lowered
        for phrase in [
            "cannot be determined",
            "cannot determine",
            "insufficient",
            "not enough information",
            "unable to determine",
            "no retrieved evidence",
        ]
    )


def question_target(question: str) -> str:
    target = (question or "").lower().strip().rstrip("?")
    patterns = [
        r"^is there (?:evidence of |a |an )?",
        r"^are there (?:evidence of |a |an )?",
        r"^is the ",
        r"^are the ",
    ]
    for pattern in patterns:
        target = re.sub(pattern, "", target)
    return target.replace(" present", "").strip()


def answer_for_check(answer: str, question: str) -> tuple[str, bool]:
    stripped = (answer or "").strip()
    binary_match = re.fullmatch(r"\s*(yes|no)[.!]?\s*", stripped, flags=re.IGNORECASE)
    target = question_target(question)
    if target and binary_match:
        alternatives = [value.strip() for value in re.split(r"\s+or\s+", target) if value.strip()]
        prefix = "There is" if binary_match.group(1).lower() == "yes" else "There is no"
        return " ".join(f"{prefix} {value}." for value in alternatives), True
    return stripped, False


def claim_polarity(answer: str, category: str) -> str:
    if is_cannot_determine(answer):
        return "cannot_determine"
    lowered = (answer or "").lower().strip()
    first = re.split(r"[.;\n]", lowered, maxsplit=1)[0]
    if re.match(r"^yes\b", first):
        return "positive"
    if re.match(r"^no\b", first):
        return "normal" if category == "normal_or_no_acute_disease" else "negative"
    if category == "location_and_attribute":
        return "negative" if NEGATION_RE.search(lowered) else "location_or_attribute"
    if NEGATION_RE.search(lowered):
        return "normal" if category == "normal_or_no_acute_disease" else "negative"
    if re.search(r"\b(normal|no acute|no active disease|unremarkable)\b", lowered):
        return "normal"
    if re.search(r"\b(there is|there are|present|seen|shows|demonstrates|compatible with)\b", lowered):
        return "positive"
    return "uncertain"


def correctness(answer: str, row: dict) -> str:
    category = row.get("human_verified_category", "")
    polarity = claim_polarity(answer, category)
    if category == "positive_finding":
        return "true" if polarity == "positive" else "false" if polarity != "uncertain" else "needs_manual_review"
    if category in {"negative_finding", "negation_sensitive"}:
        return "true" if polarity in {"negative", "normal"} else "false" if polarity != "uncertain" else "needs_manual_review"
    if category == "normal_or_no_acute_disease":
        return "true" if polarity in {"negative", "normal"} else "false" if polarity in {"positive", "cannot_determine"} else "needs_manual_review"
    if category == "location_and_attribute":
        answer_tokens = set(re.findall(r"[a-z0-9]+", answer.lower()))
        expected_tokens = set(re.findall(r"[a-z0-9]+", row.get("human_verified_answer", "").lower()))
        if expected_tokens and expected_tokens <= answer_tokens:
            return "true"
        if answer_tokens.intersection(expected_tokens):
            return "needs_manual_review"
        return "false"
    return "needs_manual_review"


def evaluate_row(row: dict, threshold: float) -> dict:
    draft = str(row.get("parsed_answer", "")).strip()
    check_answer, binary_answer = answer_for_check(draft, row.get("question", ""))
    check = check_evidence_support(
        check_answer,
        evidence_text(row),
        min_sentence_support=threshold,
    )
    if check.abstained:
        final = ABSTENTION
    elif binary_answer:
        final = draft
    else:
        final = check.revised_answer

    draft_correct = str(row.get("answer_correct_strict", "needs_manual_review")).lower()
    return {
        "eval_id": row.get("eval_id"),
        "system_name": row.get("system_name"),
        "prompt_type": row.get("prompt_type"),
        "category": row.get("human_verified_category"),
        "question": row.get("question"),
        "ground_truth_case_id": row.get("ground_truth_case_id"),
        "human_verified_answer": row.get("human_verified_answer"),
        "draft_answer": draft,
        "answer_checked_as": check_answer,
        "final_answer": final,
        "draft_correctness": draft_correct,
        "final_correctness": correctness(final, row),
        "support_rate": check.support_rate,
        "revised": final.strip() != draft.strip(),
        "abstained": check.abstained,
        "sentence_checks": [asdict(value) for value in check.sentence_checks],
        "unsupported_sentence_count": len(check.unsupported_sentences),
        "sentence_count": len(check.sentence_checks),
        "negation_conflict_count": sum(
            1 for value in check.sentence_checks if not value.negation_consistent
        ),
    }


def summarize(rows: list[dict], threshold: float) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["system_name"], row["prompt_type"])].append(row)

    output: list[dict] = []
    for (system_name, prompt_type), subset in sorted(grouped.items()):
        sentences = sum(row["sentence_count"] for row in subset)
        unsupported = sum(row["unsupported_sentence_count"] for row in subset)
        output.append(
            {
                "system_name": system_name,
                "prompt_type": prompt_type,
                "threshold": threshold,
                "n": len(subset),
                "draft_correct_true": sum(row["draft_correctness"] == "true" for row in subset),
                "draft_correct_false": sum(row["draft_correctness"] == "false" for row in subset),
                "draft_correct_manual": sum(row["draft_correctness"] == "needs_manual_review" for row in subset),
                "final_correct_true": sum(row["final_correctness"] == "true" for row in subset),
                "final_correct_false": sum(row["final_correctness"] == "false" for row in subset),
                "final_correct_manual": sum(row["final_correctness"] == "needs_manual_review" for row in subset),
                "errors_corrected": sum(
                    row["draft_correctness"] == "false" and row["final_correctness"] == "true"
                    for row in subset
                ),
                "correct_answers_lost": sum(
                    row["draft_correctness"] == "true" and row["final_correctness"] != "true"
                    for row in subset
                ),
                "mean_support_rate": mean([row["support_rate"] for row in subset]),
                "revision_rate": mean([float(row["revised"]) for row in subset]),
                "abstention_rate": mean([float(row["abstained"]) for row in subset]),
                "unsupported_sentence_rate": unsupported / sentences if sentences else 0.0,
                "negation_conflict_count": sum(row["negation_conflict_count"] for row in subset),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the P2 evidence-checking agent to P1 Stage 8B outputs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_P1_ROOT / "results" / "generations" / "stage8b_case_conditioned_generations.jsonl",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.40, 0.50, 0.60, 0.65, 0.70],
    )
    parser.add_argument("--export-threshold", type=float, default=0.40)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "final_p2" / "p1_stage8b_agent",
    )
    args = parser.parse_args()

    source_rows = [row for row in read_jsonl(args.input) if row.get("system_name") != "llm_only"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for threshold in args.thresholds:
        checked = [evaluate_row(row, threshold) for row in source_rows]
        summaries.extend(summarize(checked, threshold))
        if abs(threshold - args.export_threshold) < 1e-9:
            output_path = args.output_dir / f"p1_stage8b_agent_checked_t{threshold:.2f}.jsonl"
            with output_path.open("w", encoding="utf-8") as handle:
                for row in checked:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    json_path = args.output_dir / "p1_stage8b_agent_threshold_sweep.json"
    json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = args.output_dir / "p1_stage8b_agent_threshold_sweep.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(
        json.dumps(
            {
                "source_rows": len(source_rows),
                "summary_rows": len(summaries),
                "json": str(json_path),
                "csv": str(csv_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
