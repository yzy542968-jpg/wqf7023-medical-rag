from scripts.finalize_v9_qualitative_review import accept_proposals


def test_accept_proposals_preserves_assistant_labels() -> None:
    rows = [
        {
            "case_id": f"CXR{index}",
            "assistant_proposed_labels_v1_0": ["retrieval_relevance_gain"],
            "researcher_reviewed_labels_v1_0": [],
            "review_status": "pending_researcher_review",
        }
        for index in range(24)
    ]
    accepted = accept_proposals(
        rows, reviewer_initials="ZY", review_date="2026-08-19"
    )
    assert len(accepted) == 24
    assert {row["review_status"] for row in accepted} == {"accepted"}
    assert all(
        row["researcher_reviewed_labels_v1_0"]
        == row["assistant_proposed_labels_v1_0"]
        for row in accepted
    )
    assert {row["reviewer_initials"] for row in accepted} == {"ZY"}
