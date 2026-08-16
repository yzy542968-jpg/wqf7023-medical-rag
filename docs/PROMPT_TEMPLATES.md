# Prompt Templates

These templates are draft prompts for the planned LLM-only, report-based RAG, and case-based RAG experiments. They should be revised after the first small generation test.

## Direct Prompting

```text
You are answering a medical question for research purposes.

Question:
{question}

Answer clearly and concisely.
```

## Evidence-Guided Prompting

```text
You are answering a medical question for research purposes.
Use only the provided radiology report evidence.
If the evidence is insufficient, say that the evidence is insufficient.
Do not add unsupported medical claims.

Question:
{question}

Retrieved report evidence:
{retrieved_reports}

Answer:
```

## Structured Evidence Prompting

```text
You are answering a medical question for research purposes.
Use only the provided radiology report evidence.
Do not use outside clinical knowledge unless it is explicitly supported by the evidence.

Question:
{question}

Retrieved case evidence:
{retrieved_cases}

Respond in this structure:
Evidence:
- List the relevant evidence from the retrieved report.

Reasoning:
- Briefly connect the evidence to the answer.

Final answer:
- Provide the answer in one concise paragraph.
```

## Case-Based RAG Presentation Template

```text
Question:
{question}

Retrieved case:
- Case ID: {case_id}
- Projection images: {image_list}
- Findings: {findings}
- Impression: {impression}

Answer using the retrieved report as the main evidence. Mention that the linked images are part of the retrieved case context, but do not claim visual findings unless they are supported by the report.
```

## Agentic Evidence-Checking Output Template

```text
Draft answer:
{generated_answer}

Evidence check:
- Supported claims: {supported_sentences}
- Unsupported claims: {unsupported_sentences}

Final answer:
{revised_or_abstained_answer}
```

## Top-1 Case-Grounded Prompting

```text
You are answering a medical question for a case-grounded research experiment.
Use only the selected top-ranked radiology case evidence below.
Do not summarize other cases.
Do not combine findings or impressions from multiple cases.
If the selected case evidence is insufficient, say that the selected case evidence is insufficient.

Question:
{question}

Selected top-ranked case evidence:
{top1_case_context}

Respond in this structure:
Evidence:
- Quote or paraphrase only the relevant evidence from the selected case.

Final answer:
- Answer the question for the selected case only in one concise paragraph.
```
