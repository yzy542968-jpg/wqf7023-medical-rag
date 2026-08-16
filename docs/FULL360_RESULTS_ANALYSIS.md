# Full 360-Question Results Analysis

Current date: 2026-06-05

This document summarizes the full local Qwen2.5-1.5B generation experiments over the 360-question clean OpenI QA seed.

## Automatic Generation Results

| System | Token-F1 | Top-1 Case Accuracy | Retrieved Hit Rate | Avg Answer Words | Insufficient Rate |
|---|---:|---:|---:|---:|---:|
| LLM-only Qwen2.5-1.5B | 0.091 | 0.000 | 0.000 | 65.544 | 0.000 |
| Report-RAG BM25 Qwen2.5-1.5B | 0.146 | 0.231 | 0.383 | 84.464 | 0.008 |
| Case-RAG BM25 top-1 Qwen2.5-1.5B | 0.188 | 0.231 | 0.383 | 53.297 | 0.000 |
| Case-RAG Hybrid top-1 Qwen2.5-1.5B | 0.209 | 0.242 | 0.422 | 52.139 | 0.000 |

## Evidence-Checking Results

| System | Draft Token-F1 | Final Token-F1 | Evidence Support | Revision Rate | Abstention Rate | Unsupported Sentence Rate |
|---|---:|---:|---:|---:|---:|---:|
| Report-RAG BM25 Qwen2.5-1.5B | 0.146 | 0.093 | 0.118 | 0.992 | 0.681 | 0.819 |
| Case-RAG BM25 top-1 Qwen2.5-1.5B | 0.188 | 0.139 | 0.374 | 0.833 | 0.325 | 0.674 |
| Case-RAG Hybrid top-1 Qwen2.5-1.5B | 0.209 | 0.145 | 0.305 | 0.886 | 0.419 | 0.711 |

## Interpretation

The full 360-question results support three findings.

First, retrieval improves answer quality over LLM-only generation. Qwen2.5-1.5B without retrieval reaches Token-F1 of 0.091, while all RAG settings improve over it.

Second, case-level top-1 prompting improves over report-RAG BM25. Report-RAG BM25 reaches Token-F1 of 0.146, while case-RAG BM25 top-1 reaches 0.188 with the same BM25 retrieval hit rate. This suggests that prompt structure and case-boundary control matter, not only retrieval.

Third, hybrid BM25 + MedCPT top-1 gives the best automatic draft score and retrieved-hit rate, reaching Token-F1 of 0.209 and retrieved hit rate of 0.422. However, BM25 top-1 has higher evidence-support rate after checking. This creates a useful discussion point: the best answer-overlap system is not necessarily the most conservative evidence-supported system under the current checker.

## Thesis-Relevant Claim

The strongest claim supported by the current results is:

Case-level top-1 RAG improves local small-model medical QA over LLM-only and report-level RAG, but generated answers still contain many unsupported claims. Evidence checking is therefore necessary as a safety-oriented post-generation step, even when retrieval improves automatic answer metrics.

## What Needs Manual Review

The automatic checker is useful but not enough for final medical QA claims. The next manual annotation should focus on:

1. Whether the evidence checker is too strict for paraphrased radiology statements.
2. Whether abstentions are clinically appropriate or overly conservative.
3. Whether hybrid retrieval retrieves semantically similar but wrong cases more often than BM25.
4. Whether top-1 case prompting reduces cross-case contamination compared with top-k report prompting.
5. Whether high Token-F1 examples are genuinely supported or merely lexically similar.

Manual annotation file:

```text
experiments/manual_annotation_hybrid_qwen15_full360_agentic_top1_sample50.csv
```

Comparative manual annotation file:

```text
experiments/manual_annotation_qwen15_full360_comparative_sample50.csv
```

Automated per-answer and grouped analysis:

```text
experiments/full360_analysis/full360_error_analysis.md
experiments/full360_analysis/full360_grouped_metrics.json
experiments/full360_analysis/full360_per_answer_metrics.csv
```

## Question-Type Findings

The automated grouped analysis suggests that `abnormality_summary` questions are the hardest. For both case-RAG systems, this question type has lower Token-F1 than findings/impression questions. This is expected because abnormality-summary questions often ask about problem labels rather than directly repeating a report section.

Findings-from-indication questions have the highest retrieval hit rate for report-RAG and case-RAG systems, but evidence support differs by system. Case-RAG BM25 top-1 has higher evidence support than hybrid top-1 on findings questions under the current checker, while hybrid top-1 has the best overall draft Token-F1 and retrieved-hit rate.

Impression-from-indication questions show the clearest benefit of case-level prompting. Case-RAG BM25 top-1 and case-RAG hybrid top-1 both outperform report-RAG BM25 after evidence checking.

## Representative Automated Cases

The automated analysis selected the following cases for later manual review:

| Case Type | QID | System | Why It Matters |
|---|---|---|---|
| High-support success | `CXR2721_impression` | Case-RAG BM25 top-1 | Shows a clean case where retrieval, answer overlap, and evidence support align. |
| High-F1 but low-support | `CXR533_impression` | Report-RAG BM25 | Shows why lexical answer overlap alone is not enough for evidence-grounded evaluation. |
| Retrieval miss with poor answer | `CXR1027_summary` | Report-RAG BM25 | Shows the expected failure when the relevant case is not retrieved. |
| Heavy revision | `CXR1027_impression` | Report-RAG BM25 | Shows cross-case mixing from multiple retrieved reports and why top-1 checking matters. |
| Hybrid representative failure | `CXR1054_impression` | Case-RAG Hybrid top-1 | Shows that hybrid retrieval can still retrieve a related but not fully correct case. |

## Recommended Final Report Framing

Use the full 360-question Qwen2.5-1.5B experiments as the reproducible local baseline. If a stronger 7B/cloud run is available later, present it as an extension. If not, state clearly that the thesis contribution is evidence-grounding behavior and case-contamination analysis, not achieving state-of-the-art medical answer generation.
