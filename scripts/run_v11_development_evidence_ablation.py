"""Run the deterministic, no-GPU V11 evidence-selection development audit.

This script evaluates retrieval and evidence compression on V10 train/validation
partitions only. It does not run V10 confirmation, generate confirmation IDs, or
change any frozen artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from medical_rag.retrieval.bm25_retriever import BM25Retriever
from medical_rag.similar_case.v10_evidence import sentence_units
from medical_rag.similar_case.v11_evidence import evidence_profile, select_hierarchical_evidence
from medical_rag.similar_case.v11_qrel import qrel_v2_profile
from medical_rag.similar_case.v11_question_planner import plan_question
from medical_rag.similar_case.v11_selective import compute_retrieval_confidence, fit_proxy_threshold, risk_coverage_curve


QUESTIONS = {
    "findings": "What are the main radiographic findings?",
    "impression": "What is the most likely radiographic impression?",
    "acute": "Is there an acute cardiopulmonary abnormality? Explain briefly.",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256_ids(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _dcg(values: list[float], k: int = 10) -> float:
    return sum((2.0**value - 1.0) / __import__("math").log2(index + 2.0) for index, value in enumerate(values[:k]))


def _ndcg(ranked_values: list[float], ideal_values: list[float], k: int = 10) -> float:
    ideal = _dcg(sorted(ideal_values, reverse=True), k)
    return _dcg(ranked_values, k) / ideal if ideal > 0 else 0.0


def _whole_report_units(case: dict[str, Any]) -> list[Any]:
    case_id = str(case["case_id"])
    return sentence_units(case_id, "findings", case.get("findings")) + sentence_units(case_id, "impression", case.get("impression"))


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=ROOT / "data/processed/openi_cases.jsonl")
    parser.add_argument("--facts", type=Path, default=ROOT / "data/processed/v9_radgraph_modern_xl.jsonl")
    parser.add_argument("--split", type=Path, default=ROOT / "data/splits/v10/v10_cluster_disjoint_split.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/splits/v11/v11_development_evidence_ablation_summary.json")
    args = parser.parse_args()

    cases = _jsonl(args.cases)
    by_id = {str(row["case_id"]): row for row in cases}
    fact_rows = _jsonl(args.facts)
    facts_by_case = {str(row["case_id"]): tuple(row.get("facts", ())) for row in fact_rows if row.get("status") == "ok"}
    split = json.loads(args.split.read_text(encoding="utf-8"))
    train_ids = [str(value) for value in split["partitions"]["train"]["case_ids"]]
    validation_ids = [str(value) for value in split["partitions"]["validation"]["case_ids"]]
    train_cases = [by_id[case_id] for case_id in train_ids]
    retriever = BM25Retriever().fit(train_cases)
    rows: list[dict[str, Any]] = []

    for query_id in validation_ids:
        query = by_id[query_id]
        for question_type, question in QUESTIONS.items():
            query_text = "\n".join(part for part in (query.get("indication", ""), question) if part)
            scores = retriever.score_all(query_text)
            order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
            shortlist_indices = order[:100]
            candidates = [train_cases[index] for index in shortlist_indices]
            # Compute relevance against the complete development bank.  A
            # qrel computed only inside Top-100 makes nDCG's ideal list
            # invalid and turns the Top-100 diagnostic into circular evidence.
            full_profiles = {
                str(candidate["case_id"]): qrel_v2_profile(query, candidate, facts_by_case)
                for candidate in train_cases
            }
            qrels = {case_id: float(profile["qrel_v2"]) for case_id, profile in full_profiles.items()}
            ranked_case_ids = [str(train_cases[index]["case_id"]) for index in order]
            ranked_qrels = [qrels[case_id] for case_id in ranked_case_ids]
            shortlist_qrel_profiles = [full_profiles[str(candidate["case_id"])] for candidate in candidates]
            plan = plan_question(question, str(query.get("indication", "")))
            top_cases = candidates[:3]
            hierarchical = select_hierarchical_evidence(top_cases, query=query_text, facts_by_case=facts_by_case, plan=plan)
            whole_units = [unit for case in top_cases for unit in _whole_report_units(case)]
            hierarchical_profile = evidence_profile(hierarchical.units)
            whole_profile = evidence_profile(whole_units)
            confidence = compute_retrieval_confidence(scores, evidence_coverage=min(1.0, hierarchical_profile["unit_count"] / 6.0))
            top1_qrel = ranked_qrels[0] if ranked_qrels else 0.0
            relevant_ids = {case_id for case_id, value in qrels.items() if value >= 0.5}
            relevant_in_top100 = relevant_ids.intersection(ranked_case_ids[:100])
            relevant_recall_at100 = len(relevant_in_top100) / len(relevant_ids) if relevant_ids else 0.0
            rows.append({
                "query_case_id": query_id,
                "question_type": question_type,
                "query_text_sha256": hashlib.sha256(query_text.encode("utf-8")).hexdigest(),
                "planner_intent": plan.intent,
                "top1_qrel_v2": top1_qrel,
                "top3_mean_qrel_v2": sum(ranked_qrels[:3]) / min(3, len(ranked_qrels)) if ranked_qrels else 0.0,
                "full_bank_qrel_ndcg10": _ndcg(ranked_qrels, list(qrels.values()), 10),
                "proxy_relevant_count": len(relevant_ids),
                "proxy_relevant_in_top100_count": len(relevant_in_top100),
                "proxy_relevant_in_top100": float(bool(relevant_in_top100)),
                "proxy_relevant_recall_at100": relevant_recall_at100,
                "proxy_relevant_outside_top100": float(bool(relevant_ids - set(ranked_case_ids[:100]))),
                "qrel_v2_availability_fraction_top100": sum(float(profile["availability_fraction"]) for profile in shortlist_qrel_profiles) / len(shortlist_qrel_profiles) if shortlist_qrel_profiles else 0.0,
                "qrel_v2_available_normalized_top1": float(full_profiles[ranked_case_ids[0]]["qrel_v2_available_normalized"]) if ranked_case_ids else 0.0,
                "bm25_top_score": float(scores[order[0]]) if order else 0.0,
                "bm25_margin": float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 0.0,
                "confidence": confidence.confidence,
                "confidence_raw_top_score": confidence.top_score,
                "confidence_score_range": confidence.score_range,
                "confidence_normalized_top_score": confidence.normalized_top_score,
                "confidence_normalized_margin": confidence.normalized_margin,
                "whole_report": whole_profile,
                "case_to_fact": hierarchical_profile,
                "selected_case_ids": list(hierarchical.retrieved_case_ids),
                "selected_provenance_ids": [unit.provenance_id for unit in hierarchical.units],
            })

    confidences = [float(row["confidence"]) for row in rows]
    proxy_labels = [bool(row["proxy_relevant_in_top100"]) for row in rows]
    threshold = fit_proxy_threshold(confidences, proxy_labels, minimum_coverage=0.80)
    for row in rows:
        row["selective_use_history"] = float(row["confidence"]) >= float(threshold["threshold"])
    whole = [row["whole_report"] for row in rows]
    fact = [row["case_to_fact"] for row in rows]
    output = {
        "study": "v11_development_evidence_ablation",
        "status": "development_only_no_confirmation",
        "inputs": {
            "cases": str(args.cases.relative_to(ROOT)),
            "facts": str(args.facts.relative_to(ROOT)),
            "split": str(args.split.relative_to(ROOT)),
            "train_case_count": len(train_ids),
            "validation_case_count": len(validation_ids),
            "train_case_ids_sha256": _sha256_ids(train_ids),
            "validation_case_ids_sha256": _sha256_ids(validation_ids),
        },
        "counts": {"rows": len(rows), "questions_per_case": len(QUESTIONS), "shortlist_k": 100},
        "proxy_metrics": {
            "top1_qrel_v2": _mean(rows, "top1_qrel_v2"),
            "top3_mean_qrel_v2": _mean(rows, "top3_mean_qrel_v2"),
            "full_bank_qrel_ndcg10": _mean(rows, "full_bank_qrel_ndcg10"),
            "proxy_relevant_in_top100_rate": _mean(rows, "proxy_relevant_in_top100"),
            "proxy_relevant_recall_at100": _mean(rows, "proxy_relevant_recall_at100"),
            "proxy_relevant_outside_top100_rate": _mean(rows, "proxy_relevant_outside_top100"),
            "mean_proxy_relevant_count": _mean(rows, "proxy_relevant_count"),
            "mean_qrel_v2_availability_fraction_top100": _mean(rows, "qrel_v2_availability_fraction_top100"),
            "mean_qrel_v2_available_normalized_top1": _mean(rows, "qrel_v2_available_normalized_top1"),
        },
        "evidence_compression": {
            "whole_report": {key: _mean(whole, key) for key in ("unit_count", "character_count", "provenance_complete_rate", "duplicate_text_rate")},
            "case_to_fact": {key: _mean(fact, key) for key in ("unit_count", "character_count", "provenance_complete_rate", "duplicate_text_rate")},
            "character_reduction_fraction": 1.0 - _mean(fact, "character_count") / _mean(whole, "character_count") if _mean(whole, "character_count") else 0.0,
        },
        "planner_intent_counts": {intent: sum(row["planner_intent"] == intent for row in rows) for intent in sorted({row["planner_intent"] for row in rows})},
        "selective_gate": {"fit": threshold, "risk_coverage_curve": risk_coverage_curve(confidences, proxy_labels), "accepted_rate": sum(bool(row["selective_use_history"]) for row in rows) / len(rows)},
        "interpretation_boundary": "All qrel and selective labels are report-derived development proxies; no clinical correctness, safety, human review, or external validation is claimed.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    rows_path = args.output.with_name("v11_development_evidence_ablation_rows.jsonl")
    rows_path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
