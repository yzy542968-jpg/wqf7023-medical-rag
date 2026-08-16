# Supervisor Brief

## Proposed Title

Evidence-Checking Agentic RAG for Radiology Report-Grounded Question Answering with Linked X-ray Cases

## One-Sentence Project Summary

This project builds and evaluates a medical RAG system that answers radiology questions using expert-written report evidence while preserving linked X-ray case context and checking generated answers for unsupported or cross-case claims.

## Why This Is Not Just a Basic RAG Coursework Project

The project goes beyond a standard retrieve-then-generate demo in three ways:

1. It treats each radiology examination as a case-level image-report unit, not as isolated text chunks.
2. It compares lexical, biomedical dense, and hybrid retrieval using real OpenI data.
3. It studies a concrete medical RAG failure mode: cross-case contamination, where an LLM combines findings from several retrieved cases into one answer.

## Current Evidence of Feasibility

- OpenI/IU X-Ray reports processed: 3,851 cases.
- Linked image-projection mappings retained: 7,466.
- Clean QA seed: 120 cases and 360 questions.
- Full MedCPT index built locally.
- CUDA works on the local RTX 5070 Laptop GPU.
- Qwen2.5-0.5B and Qwen2.5-1.5B pilots completed.
- Full 360-question Qwen2.5-1.5B runs completed for LLM-only, report-RAG BM25, case-RAG BM25 top-1, and case-RAG hybrid top-1.
- Full top-1 evidence checking completed for report-RAG BM25, case-RAG BM25 top-1, and case-RAG hybrid top-1.
- P1 presentation deck completed.

## Preliminary Results

### Retrieval

| Retriever | Hit@1 | Hit@5 | Hit@20 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.231 | 0.383 | 0.486 | 0.297 |
| MedCPT | 0.108 | 0.253 | 0.406 | 0.172 |
| Hybrid BM25 + MedCPT, alpha = 0.50 | 0.242 | 0.422 | 0.553 | 0.324 |

### Local LLM Pilot, 30 Questions

| System | Token-F1 | Top-1 Case Accuracy | Retrieved Hit Rate |
|---|---:|---:|---:|
| LLM-only Qwen2.5-1.5B | 0.097 | 0.000 | 0.000 |
| Report-RAG BM25 Qwen2.5-1.5B | 0.148 | 0.300 | 0.533 |
| Case-RAG BM25 top-1 Qwen2.5-1.5B | 0.209 | 0.300 | 0.533 |
| Case-RAG Hybrid top-1 Qwen2.5-1.5B | 0.218 | 0.333 | 0.600 |

Full 360-question local runs:

| System | Token-F1 | Top-1 Case Accuracy | Retrieved Hit Rate |
|---|---:|---:|---:|
| LLM-only Qwen2.5-1.5B | 0.091 | 0.000 | 0.000 |
| Report-RAG BM25 Qwen2.5-1.5B | 0.146 | 0.231 | 0.383 |
| Case-RAG BM25 top-1 Qwen2.5-1.5B | 0.188 | 0.231 | 0.383 |
| Case-RAG Hybrid top-1 Qwen2.5-1.5B | 0.209 | 0.242 | 0.422 |

Full 360-question top-1 evidence checking:

| System | Draft Token-F1 | Final Token-F1 | Evidence Support | Revision Rate | Abstention Rate | Unsupported Sentence Rate |
|---|---:|---:|---:|---:|---:|---:|
| Report-RAG BM25 Qwen2.5-1.5B | 0.146 | 0.093 | 0.118 | 0.992 | 0.681 | 0.819 |
| Case-RAG BM25 top-1 Qwen2.5-1.5B | 0.188 | 0.139 | 0.374 | 0.833 | 0.325 | 0.674 |
| Case-RAG Hybrid top-1 Qwen2.5-1.5B | 0.209 | 0.145 | 0.305 | 0.886 | 0.419 | 0.711 |

## Proposed Contribution

The main contribution is a reproducible agentic medical RAG framework that makes evidence support case-specific. The project argues that medical RAG should not treat all top-k retrieved evidence as equally valid support when the task is to answer about one radiology case.

## Scope Boundary

The project does not perform autonomous X-ray image diagnosis. Radiology reports are the primary evidence source. X-ray images are retained as linked case context and for presentation of retrieved cases.

## Planned Final Evaluation

1. Run full-generation experiments on all 360 clean QA questions.
2. Compare LLM-only, report-RAG, case-RAG, and agentic case-RAG.
3. Evaluate retrieval metrics, Token-F1, evidence support, revision rate, abstention rate, unsupported sentence rate, and manual hallucination labels.
4. Analyze cross-case contamination examples qualitatively.
