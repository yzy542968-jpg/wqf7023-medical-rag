from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from medical_rag.evaluation.human_rating import (
    LETTERS,
    METRICS,
    completed_mask,
    existing_index,
    save_evaluation,
)


EVALUATIONS = {
    "V1 held-out stress test": ROOT
    / "experiments"
    / "final_optimized"
    / "human_evaluation"
    / "held_out_blinded_human_evaluation_36.csv",
    "V2 confirmation workflow": ROOT
    / "experiments"
    / "benchmark_v2"
    / "human_evaluation"
    / "v2_confirmation_blinded_human_evaluation_36.csv",
}


st.set_page_config(page_title="Blinded Human Evaluation", layout="wide")
st.title("Blinded Human Evaluation")
st.caption("Independent scoring interface. System identities are not loaded by this application.")

dataset = st.segmented_control(
    "Evaluation set",
    options=list(EVALUATIONS),
    default=list(EVALUATIONS)[0],
)
path = EVALUATIONS[str(dataset)]
frame = pd.read_csv(path, keep_default_na=False)
complete = completed_mask(frame)
completed_count = int(complete.sum())

progress_columns = st.columns(3)
progress_columns[0].metric("Completed", f"{completed_count}/{len(frame)}")
progress_columns[1].metric("Remaining", str(len(frame) - completed_count))
progress_columns[2].metric("Completion", f"{completed_count / len(frame):.1%}")
st.progress(completed_count / len(frame))

default_index = int((~complete).idxmax()) if not complete.all() else len(frame) - 1
sample_number = st.number_input(
    "Sample",
    min_value=1,
    max_value=len(frame),
    value=default_index + 1,
    step=1,
    key=f"sample_{dataset}",
)
row_index = int(sample_number) - 1
row = frame.iloc[row_index]

st.subheader(f"{row['sample_id']} | {row['question_type']}")
st.markdown("**Question**")
st.write(row["question"])
reference_column, evidence_column = st.columns(2, gap="large")
with reference_column:
    st.markdown("**Reference answer**")
    st.write(row["reference_answer"])
with evidence_column:
    st.markdown("**Case evidence**")
    evidence_field = (
        "gold_case_evidence" if "gold_case_evidence" in frame.columns else "retrieved_case_evidence"
    )
    st.write(row[evidence_field])

with st.form(f"rating_{dataset}_{row['sample_id']}"):
    entered: dict[str, object] = {}
    response_tabs = st.tabs([f"Response {letter.upper()}" for letter in LETTERS])
    for letter, tab in zip(LETTERS, response_tabs):
        with tab:
            st.write(row[f"response_{letter}"])
            metric_columns = st.columns(3)
            for column, (metric, label, options) in zip(metric_columns, METRICS):
                field = f"{letter}_{metric}"
                with column:
                    entered[field] = st.selectbox(
                        label,
                        options=options,
                        index=existing_index(row[field], options),
                        placeholder="Select",
                        key=f"{dataset}_{row['sample_id']}_{field}",
                    )

    best_options = ["A", "B", "C", "D", "tie"]
    entered["best_response_A_B_C_D_or_tie"] = st.selectbox(
        "Best response",
        options=best_options,
        index=existing_index(row["best_response_A_B_C_D_or_tie"], best_options),
        placeholder="Select A, B, C, D, or tie",
    )
    notes = st.text_area("Reviewer notes", value=str(row.get("reviewer_notes", "")))
    save_clicked = st.form_submit_button("Save rating", type="primary", icon=":material/save:")

if save_clicked:
    missing = [field for field, value in entered.items() if value is None or not str(value).strip()]
    if missing:
        st.error("Complete all 12 ratings and select the best response before saving.")
    else:
        for field, value in entered.items():
            frame.at[row_index, field] = value
        frame.at[row_index, "reviewer_notes"] = notes
        save_evaluation(frame, path)
        st.success(f"Saved {row['sample_id']}. Select the next incomplete sample.")

st.caption(f"Ratings are saved directly to: {path}")

