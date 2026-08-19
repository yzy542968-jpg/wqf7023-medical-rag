from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOCAL_REVIEW_PATH = ROOT / "outputs/v5_qualitative_review_local.jsonl"
CASES_PATH = ROOT / "data/processed/openi_cases.jsonl"
PUBLIC_OUTPUT_PATH = ROOT / "experiments/post_submission_v5/qualitative_representative_cases.csv"
PRIVATE_OUTPUT_PATH = ROOT / "outputs/v5_qualitative_researcher_review.csv"

REPRESENTATIVE_MARKERS = (
    "representative_retrieval_improvement",
    "representative_retrieval_degradation",
    "representative_qa_gain_support_loss",
    "representative_generation_error",
)
EXPECTED_PER_MARKER = 6
EXPECTED_TOTAL = EXPECTED_PER_MARKER * len(REPRESENTATIVE_MARKERS)
RESEARCHER_DECISION = "accepted"
RESEARCHER_INITIALS = "ZY"
RESEARCHER_REVIEW_DATE = "2026-08-19"
RESEARCHER_REVIEW_NOTE = (
    "Accepted the assistant-proposed taxonomy v1.1 labels without modification after reviewing "
    "the 24-case evidence summaries."
)

# These notes are evidence-based assistant suggestions for researcher confirmation.
# They are not human adjudications and never populate the researcher-owned fields.
ASSISTANT_PROVISIONAL_REVIEWS: dict[str, tuple[str, str, str]] = {
    "CXR2197_v2_findings": (
        "revise",
        "retrieval_improvement;generation_omission;possible_verifier_over_rejection;abstention_case",
        "Target rank improves from 59 to 10 but the paired report is not selected. The selected CXR3997 report supports the draft's right-upper-lobe granuloma statement, yet that statement is removed and only the negative finding remains; report-only abstains.",
    ),
    "CXR2433_v2_findings": (
        "confirm",
        "retrieval_improvement",
        "Target rank improves from 98 to 27 but remains below top-1. The answer follows the incorrectly selected CXR1076 report rather than the paired CXR2433 report.",
    ),
    "CXR2285_v2_impression": (
        "confirm",
        "retrieval_improvement",
        "Target rank improves from 83 to 12 but CXR1076 remains the selected report, producing an answer about pneumonia and pleural effusion that does not match the paired CXR2285 reference.",
    ),
    "CXR343_v2_impression": (
        "confirm",
        "retrieval_improvement",
        "Target rank improves from 96 to 21 without reaching top-1. The generated impression is grounded in the incorrectly selected CXR1076 report.",
    ),
    "CXR2285_v2_summary": (
        "confirm",
        "retrieval_improvement",
        "Target rank improves from 63 to 10 but the selected CXR1076 report remains mismatched, so the summary describes pneumonia and effusions rather than the paired report.",
    ),
    "CXR3784_v2_summary": (
        "confirm",
        "retrieval_improvement",
        "Target rank improves from 59 to 18 but does not reach top-1. The unchanged answer follows the mismatched CXR1076 report instead of the no-acute-abnormality reference.",
    ),
    "CXR257_v2_findings": (
        "revise",
        "retrieval_degradation;generation_omission;possible_verifier_over_rejection",
        "Target rank worsens from 20 to 38 and CXR3997 is selected. The removed granuloma sentence is visibly supported by that selected report, leaving only a negative finding in the final answer.",
    ),
    "CXR3174_v2_findings": (
        "revise",
        "retrieval_degradation;generation_omission;possible_verifier_over_rejection",
        "Target rank worsens from 19 to 40 and the wrong CXR3997 report is selected. Its granuloma statement supports the filtered draft sentence, so retrieval mismatch and possible checker over-rejection coexist.",
    ),
    "CXR1508_v2_impression": (
        "confirm",
        "retrieval_degradation",
        "Target rank worsens from 7 to 21. The selected CXR1076 report supports the generated pneumonia/effusion impression, but that report is not the paired no-active-disease case.",
    ),
    "CXR257_v2_impression": (
        "confirm",
        "retrieval_degradation",
        "Target rank worsens from 22 to 34 and CXR3997 is selected. The answer is grounded in that wrong report and conflicts with the paired developing-infection reference.",
    ),
    "CXR1120_v2_summary": (
        "confirm",
        "retrieval_degradation",
        "Target rank worsens from 9 to 29. The selected report also states no acute abnormality, creating partial answer overlap despite failure to retrieve the paired case.",
    ),
    "CXR1508_v2_summary": (
        "confirm",
        "retrieval_degradation",
        "Target rank worsens from 5 to 10 and CXR1076 remains selected. The summary is faithful to the wrong report rather than the paired no-active-disease report.",
    ),
    "CXR1120_v2_findings": (
        "revise",
        "retrieval_degradation;qa_gain_support_loss;unsupported_addition",
        "The paired target falls from rank 87 to 88 and remains unselected. Token-F1 improves through partial incidental overlap, while the checker removes an internally inconsistent cardiomegaly sentence and support falls from 1.0 to 0.6.",
    ),
    "CXR3005_v2_findings": (
        "revise",
        "retrieval_improvement;qa_gain_support_loss;verifier_evidence_disagreement",
        "Target rank improves from 18 to 5 but remains unselected. The checker removes 'Lungs are clear'; the selected report is redacted as 'Lungs are XXXX', so support cannot be resolved confidently from the displayed text.",
    ),
    "CXR2553_v2_impression": (
        "revise",
        "retrieval_improvement;qa_gain_support_loss;no_obvious_error",
        "The paired case reaches rank 1 and the final answer exactly matches 'No evidence of active disease.' The support-rate drop is caused by filtering the generic answer preamble rather than substantive clinical content.",
    ),
    "CXR3506_v2_impression": (
        "revise",
        "retrieval_improvement;qa_gain_support_loss;no_obvious_error",
        "The paired case reaches rank 1 and the final answer exactly matches 'No active disease.' Only the generic prompt-derived preamble is filtered, so the lower support rate does not indicate loss of the answer content.",
    ),
    "CXR2818_v2_summary": (
        "revise",
        "retrieval_improvement;qa_gain_support_loss;generation_omission;verifier_evidence_disagreement",
        "The paired case reaches rank 1, but the report findings describe left-upper-lobe opacity while the impression/reference says right upper lobe. The checker removes the infection/reactivation sentence, leaving an incomplete final summary amid internal laterality disagreement.",
    ),
    "CXR3018_v2_summary": (
        "revise",
        "retrieval_improvement;qa_gain_support_loss;possible_verifier_over_rejection",
        "The paired case reaches rank 1. The filtered detailed sentence is directly supported by the selected report, although the retained main conclusion still matches the no-acute-findings reference.",
    ),
    "CXR2505_v2_findings": (
        "revise",
        "possible_verifier_over_rejection;abstention_case",
        "The paired report is rank 1 and the draft closely restates its findings. The checker rejects the complete sentence and the final system abstains despite visible supporting report text.",
    ),
    "CXR2702_v2_findings": (
        "revise",
        "possible_verifier_over_rejection;abstention_case",
        "The paired report is rank 1 and the draft closely paraphrases cardiomegaly, low volumes, costophrenic blunting, and no infiltrate. Filtering removes the whole answer and produces abstention.",
    ),
    "CXR112_v2_impression": (
        "revise",
        "generation_omission;possible_verifier_over_rejection",
        "The paired report is rank 1 and explicitly states 'Hyperexpanded but clear lungs.' The checker removes that same substantive phrase, leaving only the answer preamble and Token-F1 0.",
    ),
    "CXR143_v2_impression": (
        "revise",
        "generation_omission;possible_verifier_over_rejection",
        "The paired report is rank 1 and explicitly supports old granulomatous disease with no acute pulmonary disease. The checker removes the paraphrased conclusion and leaves an empty preamble.",
    ),
    "CXR143_v2_summary": (
        "revise",
        "possible_verifier_over_rejection;abstention_case",
        "The paired report is rank 1 and visibly supports the draft conclusion. The sentence is filtered and the final output abstains, yielding Token-F1 0.",
    ),
    "CXR1897_v2_summary": (
        "revise",
        "generation_omission",
        "The paired report is rank 1, but the answer focuses on pectus deformity and omits the report's principal conclusion 'No acute disease.' No sentence filtering explains this mismatch.",
    ),
}

REFINED_LABELS_V1_1: dict[str, str] = {
    "CXR2197_v2_findings": "target_rank_improvement;top1_retrieval_failure;post_verification_content_loss;possible_verifier_over_rejection;abstention_case",
    "CXR2433_v2_findings": "target_rank_improvement;top1_retrieval_failure",
    "CXR2285_v2_impression": "target_rank_improvement;top1_retrieval_failure",
    "CXR343_v2_impression": "target_rank_improvement;top1_retrieval_failure",
    "CXR2285_v2_summary": "target_rank_improvement;top1_retrieval_failure",
    "CXR3784_v2_summary": "target_rank_improvement;top1_retrieval_failure",
    "CXR257_v2_findings": "target_rank_degradation;top1_retrieval_failure;post_verification_content_loss;possible_verifier_over_rejection",
    "CXR3174_v2_findings": "target_rank_degradation;top1_retrieval_failure;post_verification_content_loss;possible_verifier_over_rejection",
    "CXR1508_v2_impression": "target_rank_degradation;top1_retrieval_failure",
    "CXR257_v2_impression": "target_rank_degradation;top1_retrieval_failure",
    "CXR1120_v2_summary": "target_rank_degradation;top1_retrieval_failure",
    "CXR1508_v2_summary": "target_rank_degradation;top1_retrieval_failure",
    "CXR1120_v2_findings": "target_rank_degradation;top1_retrieval_failure;qa_gain_support_loss;generation_inconsistency",
    "CXR3005_v2_findings": "target_rank_improvement;top1_retrieval_failure;qa_gain_support_loss;verifier_evidence_disagreement;deidentification_ambiguity",
    "CXR2553_v2_impression": "target_rank_improvement;top1_retrieval_success;qa_gain_support_loss;template_prefix_filtering;no_substantive_answer_loss",
    "CXR3506_v2_impression": "target_rank_improvement;top1_retrieval_success;qa_gain_support_loss;template_prefix_filtering;no_substantive_answer_loss",
    "CXR2818_v2_summary": "target_rank_improvement;top1_retrieval_success;qa_gain_support_loss;post_verification_content_loss;verifier_evidence_disagreement;report_internal_inconsistency",
    "CXR3018_v2_summary": "target_rank_improvement;top1_retrieval_success;qa_gain_support_loss;post_verification_content_loss;possible_verifier_over_rejection",
    "CXR2505_v2_findings": "top1_retrieval_success;post_verification_content_loss;possible_verifier_over_rejection;abstention_case;suspected_unnecessary_abstention",
    "CXR2702_v2_findings": "top1_retrieval_success;post_verification_content_loss;possible_verifier_over_rejection;abstention_case;suspected_unnecessary_abstention",
    "CXR112_v2_impression": "top1_retrieval_success;post_verification_content_loss;possible_verifier_over_rejection",
    "CXR143_v2_impression": "top1_retrieval_success;post_verification_content_loss;possible_verifier_over_rejection",
    "CXR143_v2_summary": "top1_retrieval_success;post_verification_content_loss;possible_verifier_over_rejection;abstention_case;suspected_unnecessary_abstention",
    "CXR1897_v2_summary": "top1_retrieval_success;generation_omission;generation_focus_error",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def representative_marker(row: dict[str, Any]) -> str | None:
    matches = [marker for marker in REPRESENTATIVE_MARKERS if marker in row["provisional_categories"]]
    if len(matches) > 1:
        raise ValueError(f"Representative strata overlap for {row['qid']}: {matches}")
    return matches[0] if matches else None


def unsupported_sentences(row: dict[str, Any]) -> list[str]:
    return [
        str(check.get("sentence", "")).strip()
        for check in row.get("multimodal_sentence_checks", [])
        if not check.get("supported", False) and str(check.get("sentence", "")).strip()
    ]


def provisional_observation(row: dict[str, Any], marker: str) -> str:
    base_rank = int(row["report_only_rank"])
    multimodal_rank = int(row["multimodal_rank"])
    unsupported = unsupported_sentences(row)

    if marker == "representative_retrieval_improvement":
        gain = base_rank - multimodal_rank
        if multimodal_rank == 1:
            return f"The aligned image improves target rank by {gain} positions and reaches top-1."
        return (
            f"The aligned image improves target rank by {gain} positions, but the target remains at rank "
            f"{multimodal_rank}; the top-ranked report therefore still requires mismatch inspection."
        )
    if marker == "representative_retrieval_degradation":
        loss = multimodal_rank - base_rank
        return (
            f"The aligned image worsens target rank by {loss} positions (rank {base_rank} to "
            f"{multimodal_rank}); inspect whether the selected report is a visually similar mismatch."
        )
    if marker == "representative_qa_gain_support_loss":
        return (
            f"Final Token-F1 increases by {row['qa_final_token_f1_delta']:.3f} while automated support "
            f"decreases by {abs(row['qa_support_rate_delta']):.3f}; {len(unsupported)} sentence(s) are "
            "flagged for checking against the selected report."
        )

    if unsupported:
        outcome = "abstention" if row.get("multimodal_abstained") else "a lower-scoring final answer"
        return (
            f"Retrieval contains the paired case, but {len(unsupported)} sentence(s) are filtered and the "
            f"pipeline produces {outcome}; verify possible over-rejection versus a genuine generation error."
        )
    return (
        "Retrieval contains the paired case but final Token-F1 is below 0.5 without sentence filtering; "
        "inspect omission, focus, or polarity error in generation."
    )


def select_rows(rows: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    selected = [(marker, row) for row in rows if (marker := representative_marker(row))]
    counts = Counter(marker for marker, _ in selected)
    expected = Counter({marker: EXPECTED_PER_MARKER for marker in REPRESENTATIVE_MARKERS})
    if counts != expected:
        raise ValueError(f"Unexpected representative counts: {counts}; expected {expected}")
    if len(selected) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} unique representatives, found {len(selected)}")
    return sorted(
        selected,
        key=lambda item: (
            REPRESENTATIVE_MARKERS.index(item[0]),
            str(item[1]["question_type"]),
            str(item[1]["case_id"]),
            str(item[1]["qid"]),
        ),
    )


def write_public(rows: list[tuple[str, dict[str, Any]]]) -> None:
    fields = [
        "selection_order",
        "selection_stratum",
        "case_id",
        "qid",
        "question_type",
        "report_only_rank",
        "multimodal_rank",
        "rank_delta_multimodal_minus_report",
        "report_only_final_token_f1",
        "multimodal_final_token_f1",
        "qa_final_token_f1_delta",
        "report_only_support_rate",
        "multimodal_support_rate",
        "qa_support_rate_delta",
        "multimodal_retrieval_correct",
        "protocol_labels_v1_0",
        "assistant_action_vs_v1_0",
        "assistant_proposed_labels_v1_1",
        "assistant_review_note",
        "researcher_reviewed_labels_v1_1",
        "researcher_decision_on_v1_1",
        "review_note",
        "reviewer_initials",
        "review_date",
    ]
    PUBLIC_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PUBLIC_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for order, (marker, row) in enumerate(rows, start=1):
            assistant_decision, _, assistant_note = ASSISTANT_PROVISIONAL_REVIEWS[row["qid"]]
            output = {
                "selection_order": order,
                "selection_stratum": marker,
                **{
                    field: row.get(field, "")
                    for field in fields
                    if field
                    not in {
                        "selection_order",
                        "selection_stratum",
                        "protocol_labels_v1_0",
                    }
                },
                "protocol_labels_v1_0": ";".join(row["provisional_categories"]),
                "assistant_action_vs_v1_0": {"confirm": "unchanged", "revise": "refined"}[assistant_decision],
                "assistant_proposed_labels_v1_1": REFINED_LABELS_V1_1[row["qid"]],
                "assistant_review_note": assistant_note,
                "researcher_reviewed_labels_v1_1": REFINED_LABELS_V1_1[row["qid"]],
                "researcher_decision_on_v1_1": RESEARCHER_DECISION,
                "review_note": RESEARCHER_REVIEW_NOTE,
                "reviewer_initials": RESEARCHER_INITIALS,
                "review_date": RESEARCHER_REVIEW_DATE,
            }
            output["rank_delta_multimodal_minus_report"] = int(row["multimodal_rank"]) - int(row["report_only_rank"])
            writer.writerow(output)


def write_private(rows: list[tuple[str, dict[str, Any]]]) -> None:
    fields = [
        "selection_order",
        "selection_stratum",
        "case_id",
        "qid",
        "question_type",
        "indication",
        "question",
        "reference_answer",
        "target_image_files",
        "report_only_selected_case_id",
        "report_only_selected_report_findings",
        "report_only_selected_report_impression",
        "report_only_answer",
        "multimodal_selected_case_id",
        "selected_image_files",
        "selected_report_findings",
        "selected_report_impression",
        "multimodal_draft_answer",
        "multimodal_final_answer",
        "unsupported_sentences",
        "report_only_rank",
        "multimodal_rank",
        "report_only_final_token_f1",
        "multimodal_final_token_f1",
        "report_only_support_rate",
        "multimodal_support_rate",
        "multimodal_abstained",
        "protocol_labels_v1_0",
        "tool_assisted_provisional_observation",
        "assistant_action_vs_v1_0",
        "assistant_proposed_labels_v1_1",
        "assistant_review_note",
        "researcher_reviewed_labels_v1_1",
        "researcher_decision_on_v1_1",
        "review_note",
        "reviewer_initials",
        "review_date",
    ]
    PRIVATE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PRIVATE_OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for order, (marker, row) in enumerate(rows, start=1):
            assistant_decision, _, assistant_note = ASSISTANT_PROVISIONAL_REVIEWS[row["qid"]]
            output = {field: row.get(field, "") for field in fields}
            output.update(
                {
                    "selection_order": order,
                    "selection_stratum": marker,
                    "unsupported_sentences": " || ".join(unsupported_sentences(row)),
                    "protocol_labels_v1_0": ";".join(row["provisional_categories"]),
                    "tool_assisted_provisional_observation": provisional_observation(row, marker),
                    "assistant_action_vs_v1_0": {"confirm": "unchanged", "revise": "refined"}[assistant_decision],
                    "assistant_proposed_labels_v1_1": REFINED_LABELS_V1_1[row["qid"]],
                    "assistant_review_note": assistant_note,
                    "researcher_reviewed_labels_v1_1": REFINED_LABELS_V1_1[row["qid"]],
                    "researcher_decision_on_v1_1": RESEARCHER_DECISION,
                    "review_note": RESEARCHER_REVIEW_NOTE,
                    "reviewer_initials": RESEARCHER_INITIALS,
                    "review_date": RESEARCHER_REVIEW_DATE,
                }
            )
            writer.writerow(output)


def main() -> None:
    rows = read_jsonl(LOCAL_REVIEW_PATH)
    case_by_id = {str(case["case_id"]): case for case in read_jsonl(CASES_PATH)}
    for row in rows:
        target_case = case_by_id.get(str(row["case_id"]), {})
        selected_case = case_by_id.get(str(row["multimodal_selected_case_id"]), {})
        row["target_image_files"] = ";".join(
            str(image.get("filename", "")) for image in target_case.get("images", [])
        )
        row["selected_image_files"] = ";".join(
            str(image.get("filename", "")) for image in selected_case.get("images", [])
        )
    selected = select_rows(rows)
    selected_qids = {str(row["qid"]) for _, row in selected}
    review_qids = set(ASSISTANT_PROVISIONAL_REVIEWS)
    taxonomy_qids = set(REFINED_LABELS_V1_1)
    if selected_qids != review_qids or selected_qids != taxonomy_qids:
        missing = sorted(selected_qids - review_qids)
        extra = sorted(review_qids - selected_qids)
        taxonomy_missing = sorted(selected_qids - taxonomy_qids)
        taxonomy_extra = sorted(taxonomy_qids - selected_qids)
        raise ValueError(
            "Assistant review coverage mismatch; "
            f"review_missing={missing}, review_extra={extra}, "
            f"taxonomy_missing={taxonomy_missing}, taxonomy_extra={taxonomy_extra}"
        )
    write_public(selected)
    write_private(selected)
    print(
        json.dumps(
            {
                "source_rows": len(rows),
                "selected_rows": len(selected),
                "public_output": str(PUBLIC_OUTPUT_PATH),
                "private_output": str(PRIVATE_OUTPUT_PATH),
                "researcher_decision_on_v1_1": RESEARCHER_DECISION,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
