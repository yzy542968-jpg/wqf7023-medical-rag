from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.agentic.evidence_checker import score_sentence_pair, split_sentences
from medical_rag.agentic.semantic_evidence_checker import MedicalNLIPredictor
from medical_rag.evaluation.answer_metrics import extract_final_answer
from medical_rag.retrieval.tfidf_retriever import load_cases_jsonl


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def case_evidence(case: dict) -> str:
    return " ".join(
        [str(case.get("findings", "")), str(case.get("impression", ""))]
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure sentence-level cross-case contamination.")
    parser.add_argument(
        "--generations",
        type=Path,
        default=ROOT / "experiments" / "generations_report_rag_bm25_qwen15_full360.jsonl",
    )
    parser.add_argument(
        "--cases", type=Path, default=ROOT / "data" / "processed" / "openi_cases.jsonl"
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
    parser.add_argument("--entailment-threshold", type=float, default=0.75)
    parser.add_argument("--max-candidates", type=int, default=6)
    parser.add_argument("--minimum-lexical-anchor", type=float, default=0.20)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "experiments"
        / "final_optimized"
        / "contamination"
        / "report_rag_cross_case_contamination.json",
    )
    args = parser.parse_args()

    cases = load_cases_jsonl(args.cases)
    case_by_id = {str(case["case_id"]): case for case in cases}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    test_qids = set(split["test"]["qids"])
    rows = [row for row in read_jsonl(args.generations) if str(row["qid"]) in test_qids]
    semantic_config = json.loads(args.semantic_config.read_text(encoding="utf-8"))
    predictor = MedicalNLIPredictor(
        semantic_config["nli_model"],
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=True,
    )

    config = semantic_config["selected_config"]
    lexical_weight = float(config["lexical_weight"])
    support_threshold = float(config["support_threshold"])
    contradiction_threshold = float(config["contradiction_threshold"])
    sentence_records = []
    pairs = []
    pair_index = []
    for row in rows:
        retrieved_ids = [str(value) for value in row.get("retrieved_case_ids", [])]
        answer_sentences = split_sentences(extract_final_answer(row.get("answer", "")))
        for sentence_index, sentence in enumerate(answer_sentences):
            record = {
                "qid": row["qid"],
                "case_id": row["case_id"],
                "sentence_index": sentence_index,
                "sentence": sentence,
                "retrieved_case_ids": retrieved_ids,
                "support_by_case": {},
            }
            record_index = len(sentence_records)
            sentence_records.append(record)
            for retrieved_case_id in retrieved_ids:
                case = case_by_id.get(retrieved_case_id)
                if case:
                    candidates = []
                    for evidence_sentence in split_sentences(case_evidence(case)):
                        lexical_score, negation_consistent = score_sentence_pair(
                            sentence, evidence_sentence
                        )
                        candidates.append(
                            (evidence_sentence, lexical_score, negation_consistent)
                        )
                    candidates.sort(key=lambda value: value[1], reverse=True)
                    for candidate_index, candidate in enumerate(
                        candidates[: args.max_candidates]
                    ):
                        evidence_sentence, lexical_score, negation_consistent = candidate
                        pairs.append((evidence_sentence, sentence))
                        pair_index.append(
                            (
                                record_index,
                                retrieved_case_id,
                                candidate_index,
                                evidence_sentence,
                                lexical_score,
                                negation_consistent,
                            )
                        )

    predictions = predictor.predict(pairs)
    candidates_by_record_case = defaultdict(list)
    for metadata, prediction in zip(pair_index, predictions, strict=True):
        (
            record_index,
            retrieved_case_id,
            candidate_index,
            evidence_sentence,
            lexical_score,
            negation_consistent,
        ) = metadata
        candidates_by_record_case[(record_index, retrieved_case_id)].append(
            {
                "candidate_index": candidate_index,
                "evidence_sentence": evidence_sentence,
                "lexical_score": lexical_score,
                "negation_consistent": negation_consistent,
                **prediction,
            }
        )

    for (record_index, retrieved_case_id), candidates in candidates_by_record_case.items():
        polarity_conflicts = [
            value
            for value in candidates
            if not value["negation_consistent"] and value["lexical_score"] >= 0.45
        ]
        if polarity_conflicts:
            selected = max(polarity_conflicts, key=lambda value: value["lexical_score"])
            supported = False
            reason = "rule_polarity_conflict"
        else:
            selected = max(
                candidates,
                key=lambda value: 0.30 * value["lexical_score"]
                + 0.70 * max(value["entailment"], value["contradiction"]),
            )
            combined = (
                lexical_weight * selected["lexical_score"]
                + (1.0 - lexical_weight) * selected["entailment"]
            )
            supported = (
                selected["negation_consistent"]
                and selected["contradiction"] < contradiction_threshold
                and (
                    combined >= support_threshold
                    or selected["entailment"] >= args.entailment_threshold
                )
            )
            reason = "supported" if supported else "insufficient_support"
        selected["supported"] = supported
        selected["reason"] = reason
        sentence_records[record_index]["support_by_case"][retrieved_case_id] = selected

    contaminated_sentences = 0
    top1_supported_sentences = 0
    unsupported_sentences = 0
    contaminated_answers = set()
    anchored_contaminated_sentences = 0
    anchored_top1_supported_sentences = 0
    anchored_unsupported_sentences = 0
    anchored_contaminated_answers = set()
    for record in sentence_records:
        retrieved_ids = record["retrieved_case_ids"]
        support = record["support_by_case"]
        top1_supported = (
            support.get(retrieved_ids[0], {}).get("supported", False)
            if retrieved_ids
            else False
        )
        cross_case_supported = (
            not top1_supported
            and any(
                support.get(case_id, {}).get("supported", False)
                for case_id in retrieved_ids[1:]
            )
        )
        unsupported = not top1_supported and not cross_case_supported
        record["top1_supported"] = top1_supported
        record["cross_case_contaminated"] = cross_case_supported
        record["unsupported_by_retrieved_cases"] = unsupported
        top1_supported_sentences += int(top1_supported)
        contaminated_sentences += int(cross_case_supported)
        unsupported_sentences += int(unsupported)
        if cross_case_supported:
            contaminated_answers.add(record["qid"])

        def anchored(case_id: str) -> bool:
            value = support.get(case_id, {})
            return bool(value.get("supported", False)) and float(
                value.get("lexical_score", 0.0)
            ) >= args.minimum_lexical_anchor

        anchored_top1 = bool(retrieved_ids and anchored(retrieved_ids[0]))
        anchored_cross_case = not anchored_top1 and any(
            anchored(case_id) for case_id in retrieved_ids[1:]
        )
        anchored_unsupported = not anchored_top1 and not anchored_cross_case
        record["lexically_anchored_top1_supported"] = anchored_top1
        record["lexically_anchored_cross_case_contaminated"] = anchored_cross_case
        anchored_top1_supported_sentences += int(anchored_top1)
        anchored_contaminated_sentences += int(anchored_cross_case)
        anchored_unsupported_sentences += int(anchored_unsupported)
        if anchored_cross_case:
            anchored_contaminated_answers.add(record["qid"])

    sentence_count = len(sentence_records)
    output = {
        "system": "report_rag_bm25_top5",
        "split": "held_out_test",
        "answer_count": len(rows),
        "sentence_count": sentence_count,
        "top1_supported_sentence_rate": top1_supported_sentences / sentence_count,
        "cross_case_contaminated_sentence_rate": contaminated_sentences / sentence_count,
        "unsupported_sentence_rate": unsupported_sentences / sentence_count,
        "answers_with_cross_case_contamination": len(contaminated_answers),
        "answer_cross_case_contamination_rate": len(contaminated_answers) / len(rows),
        "lexically_anchored_top1_supported_sentence_rate": anchored_top1_supported_sentences
        / sentence_count,
        "lexically_anchored_cross_case_contaminated_sentence_rate": anchored_contaminated_sentences
        / sentence_count,
        "lexically_anchored_unsupported_sentence_rate": anchored_unsupported_sentences
        / sentence_count,
        "answers_with_lexically_anchored_cross_case_contamination": len(
            anchored_contaminated_answers
        ),
        "lexically_anchored_answer_cross_case_contamination_rate": len(
            anchored_contaminated_answers
        )
        / len(rows),
        "entailment_threshold": args.entailment_threshold,
        "support_threshold": support_threshold,
        "lexical_weight": lexical_weight,
        "method": "sentence_aligned_hybrid_semantic_checker",
        "minimum_lexical_anchor": args.minimum_lexical_anchor,
        "sentence_records": sentence_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "sentence_records"}, indent=2))


if __name__ == "__main__":
    main()
