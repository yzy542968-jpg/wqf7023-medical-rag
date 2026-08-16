# Project Charter

## Working Title

Evidence-Checking Agentic RAG for Radiology Report-Grounded Question Answering with Linked X-ray Cases

Administrative title, if the submitted form must remain unchanged:

Retrieval-Augmented Medical Question Answering over Paired Radiology Images and Reports

## Core Idea

The project studies medical question answering in radiology using retrieval-augmented generation and an evidence-checking agent loop. Instead of retrieving isolated report chunks only, the proposed system retrieves complete case-level units where each unit contains a radiology report and its associated image files. The agent then plans the query, retrieves evidence, drafts an answer, checks whether the answer is supported by retrieved report evidence, and revises or abstains when support is insufficient.

## What Counts as Success

The project should produce:

1. A P1 proposal report.
2. A P1 proposal presentation.
3. A reproducible dataset processing pipeline.
4. An LLM-only baseline.
5. A report-based RAG baseline.
6. A case-based RAG baseline using linked image-report cases.
7. An evidence-checking agentic RAG proposed system.
8. Prompt engineering comparisons.
9. Retrieval, answer, evidence-support, and hallucination-control evaluation results.
10. A final report.
11. A final presentation with selected case examples.

## Scope Boundaries

Included:

- Public paired radiology image-report data.
- Text-grounded medical question answering.
- Retrieval over reports and case-level image-report records.
- Hybrid lexical and biomedical dense retrieval.
- Evidence checking, answer revision, and abstention logic.
- Prompt engineering and comparative evaluation.

Excluded from the main scope:

- Training a full multimodal medical model from scratch.
- Large-scale fine-tuning of LLMs.
- Private clinical data.
- Ethics-sensitive human subject data collection.
- Raw-image diagnosis as the primary task.
- Autonomous clinical decision-making.

## Main Risk

The largest project risk is over-scope. The project must remain a feasible MAI research project, so the image is used as part of case-level presentation and contextual analysis, while the expert-written report remains the main evidence source for generated answers. The agentic component should focus on evidence grounding and hallucination control, not on replacing radiologists or diagnosing raw images.
