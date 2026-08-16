# Evaluation Plan

## Retrieval Evaluation

The retrieval stage will be evaluated using two datasets:

1. Keyword qrels derived from the OpenI `Problems` field.
2. Case-grounded QA seed questions generated from report findings and impressions.

Metrics:

- Hit@k
- Recall@k
- Mean Reciprocal Rank

Current keyword qrels:

```text
data/processed/openi_keyword_qrels.json
```

Current case-grounded QA seed:

```text
data/processed/openi_case_qa_seed.json
```

Current QA seed size:

- Source cases: 120
- Questions: 360

Preferred first-generation QA seed:

```text
data/processed/openi_case_qa_seed_clean.json
```

The clean seed also contains 120 cases and 360 questions, but it prioritizes non-normal problem-labeled cases and reduces weak placeholder-heavy questions.

Current QA seed lexical retrieval results:

| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.189 | 0.267 | 0.281 | 0.308 | 0.342 | 0.232 |
| BM25 | 0.233 | 0.286 | 0.303 | 0.322 | 0.344 | 0.264 |

Current clean QA seed lexical retrieval results:

| Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 0.167 | 0.239 | 0.281 | 0.339 | 0.400 | 0.220 |
| BM25 | 0.231 | 0.344 | 0.383 | 0.428 | 0.486 | 0.297 |
| MedCPT | 0.108 | 0.197 | 0.253 | 0.300 | 0.406 | 0.172 |
| Hybrid BM25 + MedCPT, alpha = 0.50 | 0.242 | 0.378 | 0.422 | 0.478 | 0.553 | 0.324 |

## Answer Evaluation

Generated answers will be evaluated on:

1. Answer relevance: whether the answer addresses the question.
2. Evidence support: whether answer claims are supported by retrieved reports.
3. Hallucination tendency: whether the answer introduces unsupported or contradictory medical claims.
4. Case completeness: whether the answer preserves useful case-level context.

The generated-answer evidence-checking script is:

```text
scripts/run_agentic_evidence_check_on_generations.py
```

For generated case-grounded QA answers, the preferred automatic evidence scope is top-1 retrieved case evidence. This stricter setting helps identify cross-case contamination, where an answer mixes supported claims from several different retrieved cases but does not answer the selected case correctly.

Current extractive answer baseline:

| System | Answer Token-F1 | Top-1 Case Accuracy | Retrieved Hit Rate | Evidence Support | Revision Rate | Abstention Rate | Non-Empty Answer Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Extractive TF-IDF RAG | 0.488 | 0.189 | N/A | N/A | N/A | N/A | 0.986 |
| Extractive BM25 RAG | 0.341 | 0.233 | N/A | N/A | N/A | N/A | 0.967 |
| Extractive TF-IDF RAG, clean seed | 0.401 | 0.167 | N/A | N/A | N/A | N/A | 0.981 |
| Extractive BM25 RAG, clean seed | 0.364 | 0.231 | N/A | N/A | N/A | N/A | 0.981 |
| Agentic hybrid RAG, clean seed | 0.412 | 0.242 | 0.422 | 0.992 | 0.008 | 0.008 | 1.000 |

These early answer scores should be treated cautiously because token overlap can be high even when the retrieved case is not the exact source case.

## Manual Checking Rubric

Use a 0-2 scale for selected outputs:

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| Relevance | Does not answer the question | Partially answers | Directly answers |
| Evidence support | Unsupported or contradicted | Partially supported | Fully supported |
| Hallucination control | Major unsupported claims | Minor unsupported claims | No unsupported claims |
| Completeness | Missing key case information | Some key information | Key findings/impression covered |

Current annotation sheet generator:

```text
scripts/build_manual_annotation_sheet.py
```

Current 30-item agentic baseline annotation sheet:

```text
experiments/manual_annotation_agentic_sample.csv
```

Current 30-item Qwen2.5-0.5B hybrid RAG top-1 evidence-checking annotation sheet:

```text
experiments/manual_annotation_hybrid_qwen05_pilot30_agentic_top1.csv
```

Current 30-item Qwen2.5-1.5B hybrid top-1 prompt and top-1 evidence-checking annotation sheet:

```text
experiments/manual_annotation_hybrid_qwen15_pilot30_agentic_top1.csv
```

Current 30-item Qwen2.5-1.5B BM25 top-1 prompt and top-1 evidence-checking annotation sheet:

```text
experiments/manual_annotation_bm25_qwen15_pilot30_agentic_top1.csv
```

## Planned System Comparison

| System | Retrieval | Prompting | Expected role |
|---|---|---|---|
| LLM-only | None | Direct | Fluency baseline |
| Report-based RAG | Report text | Evidence-guided | Basic RAG baseline |
| Case-based RAG | Paired image-report case | Evidence-guided / structured | Case-level context baseline |
| Agentic case-based RAG | Hybrid case retrieval | Evidence checking and revision | Proposed system |
