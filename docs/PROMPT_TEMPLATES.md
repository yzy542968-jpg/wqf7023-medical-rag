# Prompt Templates

This registry preserves historical templates and the final V10/V11 prompt contracts. Historical prompts are retained for traceability; only the explicitly versioned final contracts describe the completed study.

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

## V10 Frozen Target-Case Answer Contract

The target report is not available to the generator. Historical cases are analogies, not facts about the target patient.

```text
You are answering a research question about the TARGET chest radiograph.

Use the TARGET image and indication as the primary evidence. Historical reports are other-patient analogies and may be irrelevant. Never state a historical finding as though it belongs to the target patient. If the target image does not support a definite answer, say that the available evidence is insufficient.

TARGET indication:
{target_indication}

Question:
{question}

Historical evidence, with immutable provenance:
{selected_historical_evidence}

Return no more than two complete sentences containing only the answer. Do not invent case IDs, section names, or evidence IDs.
```

The generator output is treated as untrusted text. Case IDs, section labels, evidence IDs, abstention state and provenance-validity fields are assembled deterministically after generation.

## V11 Development-Only Case-to-Fact Contract

```text
You are answering a research question about the TARGET chest radiograph.

The evidence units below were selected inside retrieved historical cases. They remain other-patient analogies. Use them only when they are compatible with the target image and question. Do not merge details across patients. If no reliable answer is supported, state that the available evidence is insufficient.

TARGET indication:
{target_indication}

Question intent:
{planned_intent}

Question:
{question}

Selected historical facts:
{case_scoped_fact_units}

Return no more than two complete sentences containing only the answer.
```

V11 retains the same deterministic provenance assembly used by V10. The planner and fact selector are development components; their outputs do not establish clinical correctness.

## V10/V11 Output and Provenance Rules

1. The target image, indication and question are always separated from historical context.
2. Historical evidence is always labeled with a case ID and source section.
3. A generated answer cannot create, modify or delete provenance identifiers.
4. Unknown evidence IDs, malformed output and token-ceiling truncation are marked invalid rather than silently repaired into a valid result.
5. Retrieval confidence is a research signal about report-derived relevance, not diagnostic confidence.
6. No prompt claims physician validation, clinical safety or external generalization.
