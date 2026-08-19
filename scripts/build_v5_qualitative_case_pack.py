from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_PATH = ROOT / "experiments/post_submission_v5/confirmation_retrieval_rows.jsonl"
REPORT_QA_PATH = ROOT / "experiments/post_submission_v5/qa_report_only/final_optimized_test_rows.jsonl"
MULTIMODAL_QA_PATH = ROOT / "experiments/post_submission_v5/qa_multimodal/final_optimized_test_rows.jsonl"
COHORT_PATH = ROOT / "data/processed/openi_multimodal_v5_cohort.json"
CASES_PATH = ROOT / "data/processed/openi_cases.jsonl"
PUBLIC_PATH = ROOT / "experiments/post_submission_v5/qualitative_case_pack.csv"
LOCAL_PATH = ROOT / "outputs/v5_qualitative_review_local.jsonl"

PROTOCOL_VERSION = "1.0"
V5_COMMIT = "10f57ba"
PROTOCOL_COMMIT = "d3b0765"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def question_type(qid: str) -> str:
    for suffix in ("findings", "impression", "summary"):
        if qid.endswith(f"_{suffix}"):
            return suffix
    return "unknown"


def as_bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else str(value).lower() == "true"


def add_category(categories: set[str], category: str) -> None:
    categories.add(category)


def build_retrieval_index() -> dict[str, dict[str, Any]]:
    rows = read_jsonl(RETRIEVAL_PATH)
    by_qid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_qid[str(row["qid"])][str(row["system"])] = row

    index: dict[str, dict[str, Any]] = {}
    improvement_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    degradation_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qid, systems in by_qid.items():
        base = systems["indication_question_bm25"]
        multimodal = systems["indication_question_correct_image"]
        base_rank = int(base["target_rank"])
        multimodal_rank = int(multimodal["target_rank"])
        rank_gain = base_rank - multimodal_rank
        mrr_delta = float(multimodal["mrr"]) - float(base["mrr"])
        f1_delta = float(multimodal["token_f1"]) - float(base["token_f1"])
        qtype = question_type(qid)
        if rank_gain > 0 or (rank_gain == 0 and f1_delta > 0):
            status = "improvement"
            improvement_by_type[qtype].append({"qid": qid, "rank_gain": rank_gain, "mrr_delta": mrr_delta, "case_id": str(base["case_id"])})
        elif rank_gain < 0 or (rank_gain == 0 and f1_delta < 0):
            status = "degradation"
            degradation_by_type[qtype].append({"qid": qid, "rank_loss": -rank_gain, "mrr_delta": mrr_delta, "case_id": str(base["case_id"])})
        else:
            status = "no_change"
        index[qid] = {
            "case_id": str(base["case_id"]),
            "qid": qid,
            "question_type": qtype,
            "retrieval_status": status,
            "report_only_rank": base_rank,
            "multimodal_rank": multimodal_rank,
            "rank_delta_multimodal_minus_report": multimodal_rank - base_rank,
            "retrieval_mrr_delta": round(mrr_delta, 8),
            "retrieval_token_f1_delta": round(f1_delta, 8),
            "report_only_selected_case_id": str(base.get("selected_case_id") or ""),
            "multimodal_selected_case_id": str(multimodal.get("selected_case_id") or ""),
            "retrieval_categories": set(),
        }
        if status == "improvement":
            add_category(index[qid]["retrieval_categories"], "retrieval_improvement")
        elif status == "degradation":
            add_category(index[qid]["retrieval_categories"], "retrieval_degradation")

    for entries, category in ((improvement_by_type, "representative_retrieval_improvement"), (degradation_by_type, "representative_retrieval_degradation")):
        for qtype, values in entries.items():
            if category.endswith("improvement"):
                ordered = sorted(values, key=lambda row: (-row["rank_gain"], -row["mrr_delta"], row["case_id"], row["qid"]))
            else:
                ordered = sorted(values, key=lambda row: (-row["rank_loss"], row["mrr_delta"], row["case_id"], row["qid"]))
            for selected in ordered[:2]:
                add_category(index[selected["qid"]]["retrieval_categories"], category)
    return index


def merge_qa(index: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    report_rows = {str(row["qid"]): row for row in read_jsonl(REPORT_QA_PATH)}
    multimodal_rows = {str(row["qid"]): row for row in read_jsonl(MULTIMODAL_QA_PATH)}
    qa_details: dict[str, dict[str, Any]] = {}
    gain_support_loss: dict[str, list[dict[str, Any]]] = defaultdict(list)
    correct_generation_error: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for qid, multimodal in multimodal_rows.items():
        report = report_rows[qid]
        base = index[qid]
        final_delta = float(multimodal["final_token_f1"]) - float(report["final_token_f1"])
        support_delta = float(multimodal["support_rate"]) - float(report["support_rate"])
        retrieved = [str(value) for value in multimodal.get("retrieved_case_ids", [])]
        correct_retrieval = str(multimodal["case_id"]) in retrieved
        categories = set(base["retrieval_categories"])
        if final_delta > 0 and support_delta < 0:
            add_category(categories, "qa_gain_support_loss")
            gain_support_loss[base["question_type"]].append({"qid": qid, "f1_delta": final_delta, "support_delta": support_delta})
        if correct_retrieval and float(multimodal["final_token_f1"]) < 0.5:
            add_category(categories, "correct_retrieval_generation_error")
            correct_generation_error[base["question_type"]].append({"qid": qid, "final_f1": float(multimodal["final_token_f1"])})
        if as_bool(multimodal.get("revised")):
            add_category(categories, "possible_generation_unsupported_addition")
        if float(multimodal["final_token_f1"]) < float(multimodal["draft_token_f1"]):
            add_category(categories, "possible_verifier_over_rejection")
        if as_bool(multimodal.get("revised")) and abs(support_delta) > 0:
            add_category(categories, "verifier_evidence_disagreement")
        if as_bool(multimodal.get("agent_abstained")) or as_bool(report.get("agent_abstained")):
            add_category(categories, "abstention_case")
        if not categories:
            add_category(categories, "no_obvious_error")
        base["qa_final_token_f1_delta"] = round(final_delta, 8)
        base["qa_support_rate_delta"] = round(support_delta, 8)
        base["report_only_final_token_f1"] = float(report["final_token_f1"])
        base["multimodal_final_token_f1"] = float(multimodal["final_token_f1"])
        base["report_only_support_rate"] = float(report["support_rate"])
        base["multimodal_support_rate"] = float(multimodal["support_rate"])
        base["multimodal_retrieved_case_ids"] = retrieved
        base["multimodal_retrieval_correct"] = correct_retrieval
        base["qa_categories"] = categories
        qa_details[qid] = {
            "case_id": base["case_id"],
            "qid": qid,
            "question_type": base["question_type"],
            "question": multimodal.get("question", ""),
            "indication": "",
            "reference_answer": multimodal.get("reference_answer", ""),
            "report_only_answer": report.get("final_answer", ""),
            "multimodal_draft_answer": multimodal.get("draft_answer", ""),
            "multimodal_final_answer": multimodal.get("final_answer", ""),
            "report_only_selected_case_id": base["report_only_selected_case_id"],
            "multimodal_selected_case_id": base["multimodal_selected_case_id"],
            "multimodal_retrieved_case_ids": retrieved,
            "multimodal_retrieval_correct": correct_retrieval,
            "report_only_rank": base["report_only_rank"],
            "multimodal_rank": base["multimodal_rank"],
            "report_only_final_token_f1": report["final_token_f1"],
            "multimodal_final_token_f1": multimodal["final_token_f1"],
            "qa_final_token_f1_delta": final_delta,
            "report_only_support_rate": report["support_rate"],
            "multimodal_support_rate": multimodal["support_rate"],
            "qa_support_rate_delta": support_delta,
            "report_only_abstained": as_bool(report.get("agent_abstained")),
            "multimodal_abstained": as_bool(multimodal.get("agent_abstained")),
            "multimodal_sentence_checks": multimodal.get("sentence_checks", []),
            "provisional_categories": sorted(categories),
            "researcher_review_status": "not_reviewed",
        }

    for values, category in ((gain_support_loss, "representative_qa_gain_support_loss"), (correct_generation_error, "representative_generation_error")):
        for qtype, entries in values.items():
            if category.endswith("support_loss"):
                ordered = sorted(entries, key=lambda row: (-row["f1_delta"], row["support_delta"], row["qid"]))
            else:
                ordered = sorted(entries, key=lambda row: (row["final_f1"], row["qid"]))
            for selected in ordered[:2]:
                qid = selected["qid"]
                add_category(index[qid]["qa_categories"], category)
                if category not in qa_details[qid]["provisional_categories"]:
                    qa_details[qid]["provisional_categories"].append(category)
                    qa_details[qid]["provisional_categories"].sort()
    return index, qa_details


def main() -> None:
    index, qa_details = merge_qa(build_retrieval_index())
    case_by_id = {str(row["case_id"]): row for row in read_jsonl(CASES_PATH)}
    for qid, detail in qa_details.items():
        case = case_by_id.get(detail["case_id"], {})
        detail["indication"] = str(case.get("indication", ""))
        selected_case = case_by_id.get(detail["multimodal_selected_case_id"], {})
        detail["selected_report_findings"] = str(selected_case.get("findings", ""))
        detail["selected_report_impression"] = str(selected_case.get("impression", ""))
        report_case = case_by_id.get(detail["report_only_selected_case_id"], {})
        detail["report_only_selected_report_findings"] = str(report_case.get("findings", ""))
        detail["report_only_selected_report_impression"] = str(report_case.get("impression", ""))

    PUBLIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    public_fields = [
        "protocol_version", "v5_commit", "protocol_commit", "case_id", "qid", "question_type",
        "retrieval_status", "report_only_rank", "multimodal_rank", "rank_delta_multimodal_minus_report",
        "retrieval_mrr_delta", "retrieval_token_f1_delta", "qa_final_token_f1_delta",
        "report_only_final_token_f1", "multimodal_final_token_f1", "qa_support_rate_delta",
        "report_only_support_rate", "multimodal_support_rate", "report_only_selected_case_id",
        "multimodal_selected_case_id", "multimodal_retrieved_case_ids", "multimodal_retrieval_correct",
        "retrieval_categories",
        "qa_categories", "researcher_review_status",
    ]
    with PUBLIC_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=public_fields)
        writer.writeheader()
        for qid in sorted(index):
            row = index[qid]
            output = {field: row.get(field, "") for field in public_fields}
            output["protocol_version"] = PROTOCOL_VERSION
            output["v5_commit"] = V5_COMMIT
            output["protocol_commit"] = PROTOCOL_COMMIT
            output["researcher_review_status"] = "not_reviewed"
            output["retrieval_categories"] = ";".join(sorted(row.get("retrieval_categories", set())))
            output["qa_categories"] = ";".join(sorted(row.get("qa_categories", set())))
            output["multimodal_retrieved_case_ids"] = ";".join(row.get("multimodal_retrieved_case_ids", []))
            writer.writerow(output)

    LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL_PATH.open("w", encoding="utf-8") as handle:
        for qid in sorted(qa_details):
            handle.write(json.dumps(qa_details[qid], ensure_ascii=False) + "\n")

    print(json.dumps({
        "public_output": str(PUBLIC_PATH),
        "local_output": str(LOCAL_PATH),
        "rows": len(index),
        "local_review_rows": len(qa_details),
        "protocol_version": PROTOCOL_VERSION,
        "v5_commit": V5_COMMIT,
        "protocol_commit": PROTOCOL_COMMIT,
    }, indent=2))


if __name__ == "__main__":
    main()
