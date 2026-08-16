from __future__ import annotations

from scripts.evaluate_v21_template_transfer import family, transfer_question


def test_transfer_templates_do_not_repeat_original_wording() -> None:
    rows = [
        {
            "qid": "CXR1_v21_observation",
            "question": "What did the radiographic examination show?",
        },
        {
            "qid": "CXR1_v21_fact_probe",
            "question": "What does this report state about opacity?",
        },
        {
            "qid": "CXR1_v21_unanswerable_a",
            "question": "What was the patient's serum troponin concentration?",
        },
    ]
    assert all(transfer_question(row) != row["question"] for row in rows)
    assert family("CXR1_v21_fact_probe") == "fact_probe"
