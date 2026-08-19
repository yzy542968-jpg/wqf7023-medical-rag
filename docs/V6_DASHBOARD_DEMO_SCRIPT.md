# V6 Dashboard Demonstration Script

## Purpose

This dashboard demonstrates the deployed form of the V6 model-modernized
confirmation pipeline. It is an interactive research prototype, not a new
evaluation run and not a clinical decision-support system.

The demonstration accepts a chest X-ray image, a clinical indication, and a
report-grounded question. It then retrieves the top-ranked candidate report
from the frozen 240-case V6 confirmation pool. The page exposes the retrieval
trace and produces either a transparent extractive answer or a Qwen2.5
report-grounded answer.

The wording **top-ranked candidate report** is intentional. For a newly
uploaded image, the system does not establish that the selected report belongs
to the same patient as the image. The page therefore demonstrates closed-set
image-to-report retrieval, not patient identification or autonomous diagnosis.

## Before the demonstration

1. Start the application from the repository root:

   ```powershell
   streamlit run app.py
   ```

2. Open the local URL printed by Streamlit.

3. Select **V6 confirmation demo**.

4. Use a local PNG or JPEG chest X-ray. The application requires the local
   OpenI source cases, the cached V6 report chunks, and the locally available
   `google/medsiglip-448` model revision specified in
   `config/v6_confirmation.json`.

5. For the first live run, use **Extractive report answer**. This keeps the
   demonstration focused on retrieval and does not require loading the Qwen
   generator. Use the Qwen option only when the local generator weights are
   available and a slower generation step is acceptable.

## Demonstration sequence

### 1. Introduce the task

Say:

> This prototype receives a chest X-ray and a report-grounded question. It
> searches a fixed candidate pool of 240 indexed OpenI cases, ranks candidate
> reports using text and image evidence, and then answers from the selected
> report. The selected item is a candidate report, not a verified patient
> match.

Enter an indication such as `Chest pain` and use one of these questions:

- `What is the final radiology impression for this examination?`
- `What radiographic findings are documented for this examination?`
- `Summarize the principal abnormality or conclusion in this report.`

### 2. Show the uploaded image

Point out that the image is shown before retrieval. Explain that the image is
encoded by the MedSigLIP image tower at request time. The uploaded pixels are
used for the interactive request and are not added to the frozen benchmark
artifacts.

### 3. Run candidate retrieval

Click **Run V6 candidate retrieval**. The status panel shows the workflow:

1. Load the frozen V6 candidate pool and report chunk manifest.
2. Encode the uploaded image with MedSigLIP-448.
3. Generate a BM25 top-100 report shortlist using the indication and question.
4. Rerank the shortlist with maximum image-to-report-chunk cosine similarity.
5. Fuse normalized text and image scores with equal weights.
6. Select the top-ranked candidate report and audit the answer against that
   report only.

The text query is:

```text
Clinical indication: <indication>
Question: <question>
```

The V6 fusion policy is fixed at development/confirmation time. It uses the
BM25 top-100 shortlist, independent min-max normalization inside that
shortlist, a 0.5 text weight, a 0.5 image weight, and the maximum similarity
between the uploaded image and any frozen report chunk for each candidate.

### 4. Explain the selected report

Read the **Top-ranked candidate report** section. The page displays the
candidate case ID, findings, and impression. Say:

> This is the report that ranked first under the frozen V6 retrieval policy.
> The system is faithful to this selected report for the downstream answer,
> but this screen alone does not prove that the report belongs to the person
> represented by the uploaded image.

This distinction demonstrates the central alignment boundary in the study:

```text
Faithfulness to retrieved report
does not by itself prove
correct image-report or patient alignment.
```

### 5. Explain the answer and audit

The extractive mode returns the findings or impression section that matches the
question type. The Qwen mode generates one concise answer using only the
selected report. The evidence support value is a lexical report-support
signal. It is not a physician rating, clinical gold standard, or diagnosis
confidence.

If the audit reports insufficient support, say:

> This is an automated evidence-audit flag. It indicates that the answer and
> the selected report did not align strongly under the configured lexical
> check. It is not a clinical adjudication.

### 6. Show the retrieval trace

The table contains the top ten candidate reports and exposes:

- final fused score;
- raw BM25 score;
- normalized BM25 score;
- maximum image-to-report-chunk similarity;
- normalized image score.

Use the trace to explain that the image does not search the complete report
corpus by itself. Text retrieval first establishes the shortlist, after which
image evidence changes the order within that shortlist.

The five-step trace below the table is the compact version for a presentation:

```text
Encode uploaded image
→ BM25 top-100 shortlist
→ MedSigLIP max-chunk reranking
→ top-ranked candidate report
→ report-support audit
```

## What to say about the research results

The dashboard is a live demonstration of the technical route. The formal V6
claims come from the frozen confirmation outputs and their recorded statistics,
not from a single interactive upload. The formal retrieval comparison found a
positive MedSigLIP-over-BM25 MRR difference on the confirmation cohort, and the
downstream QA comparison showed a larger multimodal gain under MedGemma than
under Qwen2.5. The exact values should be read from the versioned V6 result
record rather than improvised during the presentation.

The dashboard should not be used to claim:

- external validation;
- patient-level independence;
- image diagnosis;
- clinical utility or safety;
- physician-authored question performance;
- proof that the top-ranked report belongs to the uploaded patient;
- that one successful interactive example establishes population-level
  performance.

## Recommended closing statement

> The system demonstrates how image evidence can improve candidate-report
> ranking and how that ranking can feed report-grounded question answering.
> The important limitation is that a report can be faithfully used after
> retrieval while still being misaligned with the intended case. Therefore the
> prototype is best understood as an auditable research workflow for
> image-report retrieval and evidence grounding, not as an autonomous medical
> diagnostic tool.
