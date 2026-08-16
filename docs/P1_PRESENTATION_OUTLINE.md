# P1 Presentation Outline

## Deck Title

Evidence-Checking Agentic RAG for Radiology Report-Grounded Question Answering with Linked X-ray Cases

## Target Length

10 minutes, 9-10 slides.

## Slide 1: Title

Claim: This project studies safer medical RAG by preserving radiology case context and checking generated answers against report evidence.

Content:

- Name: ZHANG YUE
- Matric No: 22097191
- Programme: Master of Artificial Intelligence
- Supervisor: Dr. Uzair Ishtiaq
- Course: WQF7023 Artificial Intelligence Research Project

Speaker note:

Introduce the project as report-grounded medical QA, not autonomous diagnosis. The image is linked as case context, while the radiology report remains the main evidence.

## Slide 2: Problem

Claim: Medical QA needs evidence grounding because fluent answers can still be unsupported.

Proof object:

- Simple flow: question -> LLM answer -> risk of unsupported claim
- Contrast: LLM-only versus retrieval-grounded answer

Speaker note:

Explain hallucination risk in healthcare-oriented QA. The project focuses on traceability to expert-written radiology reports.

## Slide 3: Research Gap

Claim: Basic RAG may retrieve evidence but still lose case structure or mix evidence across cases.

Proof object:

- Comparison table:
  - Report chunk retrieval
  - Top-k case retrieval
  - Top-1 case-grounded retrieval with evidence checking

Speaker note:

Point out that radiology data naturally comes as image-report cases. Treating retrieved text as a loose top-k pool can create cross-case contamination.

## Slide 4: Research Questions

Claim: The project evaluates whether case-level retrieval and evidence checking improve grounded medical QA.

Questions:

1. Does retrieval improve medical QA compared with LLM-only answering?
2. Does hybrid BM25 + MedCPT retrieval improve case retrieval?
3. Can evidence checking identify unsupported or cross-case claims?
4. Can linked image-report cases improve case-level presentation?

Speaker note:

Keep the scope conservative: no raw-image diagnosis and no clinical deployment claim.

## Slide 5: Dataset And Case Unit

Claim: OpenI can be normalized into complete image-report retrieval units.

Proof object:

- Dataset counts:
  - 3,851 reports
  - 7,466 image mappings
  - 120 clean QA-seed cases
  - 360 clean QA-seed questions

Speaker note:

Each case stores indication, findings, impression, problem labels, and linked X-ray filenames.

## Slide 6: System Design

Claim: The proposed system adds an evidence-checking agent loop on top of hybrid case retrieval.

Proof object:

Pipeline diagram:

Question -> Query planning -> Hybrid BM25 + MedCPT retrieval -> Top-1 case-grounded prompt -> LLM draft -> Evidence checker -> Final answer or abstention

Speaker note:

Explain that the agent is not an autonomous doctor. It plans, retrieves, checks, revises, and abstains.

## Slide 7: Retrieval Results

Claim: Hybrid retrieval improves case recall over BM25 and standalone MedCPT.

Proof object:

| Retriever | Hit@1 | Hit@5 | Hit@20 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.231 | 0.383 | 0.486 | 0.297 |
| MedCPT | 0.108 | 0.253 | 0.406 | 0.172 |
| Hybrid alpha 0.50 | 0.242 | 0.422 | 0.553 | 0.324 |

Speaker note:

The important finding is not that dense retrieval alone wins. MedCPT alone is weaker here, but it adds complementary signal when fused with BM25.

## Slide 8: Generation Pilot

Claim: Retrieval improves the small-model pilot, and top-1 case prompting is the strongest local setting.

Proof object:

| System | Token-F1 | Top-1 | Retrieved Hit |
|---|---:|---:|---:|
| Qwen2.5-1.5B LLM-only | 0.097 | 0.000 | 0.000 |
| Report RAG BM25 | 0.148 | 0.300 | 0.533 |
| Case RAG BM25 top-1 | 0.209 | 0.300 | 0.533 |
| Case RAG Hybrid top-1 | 0.218 | 0.333 | 0.600 |

Speaker note:

State clearly that this is a 30-question pilot, not final thesis-scale evaluation.

## Slide 9: Key Error Analysis

Claim: The pilot identified cross-case contamination as a concrete medical RAG risk.

Proof object:

- Before: model combines impressions from several retrieved cases.
- After: top-1 checker removes claims not supported by the selected case.

Speaker note:

This is the main research insight so far. Evidence checking should be case-specific, not just supported somewhere in top-k evidence.

## Slide 10: Plan To Completion

Claim: The remaining work is evaluation depth, not basic feasibility.

Next steps:

1. Complete manual annotation on 30-50 generated answers.
2. Run larger generation experiment if compute allows.
3. Compare LLM-only, report-RAG, case-RAG, and agentic RAG.
4. Write final analysis of retrieval failures, cross-case contamination, revision, and abstention.
5. Prepare final report and defense examples.

Speaker note:

Close with feasibility: the data pipeline, retrieval baselines, hybrid retriever, LLM pilot, and agentic checker already run locally.
