from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.evidence_checker import check_evidence_support, split_sentences
from medical_rag.agentic.semantic_evidence_checker import (
    MedicalNLIPredictor,
    check_semantic_evidence_support,
)
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


FLIP_PATTERNS = [
    (re.compile(r"\b[Tt]here is no\b"), "There is"),
    (re.compile(r"\b[Tt]here are no\b"), "There are"),
    (re.compile(r"^\s*[Nn]o evidence of\b"), "Evidence of"),
    (re.compile(r"^\s*[Nn]o\b"), ""),
    (re.compile(r"\bwithout evidence of\b", re.IGNORECASE), "with evidence of"),
    (re.compile(r"\bwithout demonstration of\b", re.IGNORECASE), "with demonstration of"),
    (re.compile(r"\bnegative for\b", re.IGNORECASE), "positive for"),
]


def flip_explicit_polarity(sentence: str) -> str | None:
    for pattern, replacement in FLIP_PATTERNS:
        flipped, count = pattern.subn(replacement, sentence, count=1)
        flipped = " ".join(flipped.split()).strip()
        if count and flipped and flipped.lower() != sentence.lower():
            return flipped[0].upper() + flipped[1:]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only polarity stress test for evidence checkers."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=ROOT / "data" / "processed" / "openi_cases.jsonl",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=ROOT / "data" / "splits" / "openi_qa_grouped_case_seed7023.json",
    )
    parser.add_argument(
        "--semantic-config",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "semantic_agent"
        / "semantic_agent_selection.json",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=120)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "verifier_stress_test"
        / "development_polarity_stress_test.json",
    )
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    development_cases = set(str(value) for value in split["development"]["case_ids"])
    cases = [
        case
        for case in load_cases_jsonl(args.cases)
        if str(case["case_id"]) in development_cases
    ]
    examples = []
    for case in sorted(cases, key=lambda value: str(value["case_id"])):
        evidence = " ".join(
            [str(case.get("findings", "")), str(case.get("impression", ""))]
        ).strip()
        for sentence in split_sentences(evidence):
            flipped = flip_explicit_polarity(sentence)
            if flipped:
                examples.append(
                    {
                        "case_id": str(case["case_id"]),
                        "evidence": evidence,
                        "entailed_claim": sentence,
                        "contradictory_claim": flipped,
                    }
                )
                if len(examples) >= args.max_pairs:
                    break
        if len(examples) >= args.max_pairs:
            break
    if not examples:
        raise RuntimeError("no explicit polarity examples were found")

    semantic_selection = json.loads(args.semantic_config.read_text(encoding="utf-8"))
    config = semantic_selection["selected_config"]
    predictor = MedicalNLIPredictor(
        semantic_selection["nli_model"],
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=True,
    )
    counts = {
        "rule_positive_accepted": 0,
        "rule_contradiction_rejected": 0,
        "semantic_positive_accepted": 0,
        "semantic_contradiction_rejected": 0,
    }
    rows = []
    for example in examples:
        rule_positive = check_evidence_support(
            example["entailed_claim"], example["evidence"], min_sentence_support=0.40
        )
        rule_negative = check_evidence_support(
            example["contradictory_claim"], example["evidence"], min_sentence_support=0.40
        )
        semantic_positive = check_semantic_evidence_support(
            example["entailed_claim"],
            example["evidence"],
            predictor,
            min_combined_support=float(config["support_threshold"]),
            entailment_threshold=float(config["entailment_threshold"]),
            contradiction_threshold=float(config["contradiction_threshold"]),
            lexical_weight=float(config["lexical_weight"]),
        )
        semantic_negative = check_semantic_evidence_support(
            example["contradictory_claim"],
            example["evidence"],
            predictor,
            min_combined_support=float(config["support_threshold"]),
            entailment_threshold=float(config["entailment_threshold"]),
            contradiction_threshold=float(config["contradiction_threshold"]),
            lexical_weight=float(config["lexical_weight"]),
        )
        decisions = {
            "rule_positive_accepted": not rule_positive.abstained,
            "rule_contradiction_rejected": rule_negative.abstained,
            "semantic_positive_accepted": not semantic_positive.abstained,
            "semantic_contradiction_rejected": semantic_negative.abstained,
        }
        for key, value in decisions.items():
            counts[key] += int(value)
        rows.append({**example, **decisions})

    pair_count = len(rows)
    summary = {
        "split": "development_only",
        "pair_count": pair_count,
        **{key + "_rate": value / pair_count for key, value in counts.items()},
        "synthetic_construction": "explicit negation removal or polarity phrase replacement",
        "limitation": "This stress test validates polarity sensitivity, not general clinical verifier accuracy.",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
