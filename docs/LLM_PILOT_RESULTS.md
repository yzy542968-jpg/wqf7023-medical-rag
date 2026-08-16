# LLM Pilot Results

## Purpose

This document tracks the end-to-end generation and evidence-checking pipeline using small Qwen2.5 instruction models on the local RTX 5070 Laptop GPU. The first results were 30-question pilots. The strongest local setting has now also been run over the full 360-question clean QA seed.

## Environment

- GPU: NVIDIA GeForce RTX 5070 Laptop GPU
- PyTorch: `2.11.0+cu128`
- CUDA visible to PyTorch: yes
- Pilot models: `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-1.5B-Instruct`
- Pilot size: first 30 clean QA-seed questions
- Full local run: 360 clean QA-seed questions for the strongest hybrid top-1 Qwen2.5-1.5B setting
- Max new tokens: 160

## Generation Results

| System | Answer Token-F1 | Top-1 Case Accuracy | Retrieved Hit Rate | Average Answer Words | Insufficient Rate |
|---|---:|---:|---:|---:|---:|
| LLM-only | 0.078 | 0.000 | 0.000 | 84.367 | 0.000 |
| Report RAG, BM25 | 0.187 | 0.300 | 0.533 | 60.633 | 0.033 |
| Case RAG, BM25 | 0.163 | 0.300 | 0.533 | 83.600 | 0.000 |
| Case RAG, hybrid BM25 + MedCPT | 0.187 | 0.333 | 0.600 | 80.867 | 0.000 |
| LLM-only, Qwen2.5-1.5B | 0.097 | 0.000 | 0.000 | 63.567 | 0.000 |
| Report RAG, BM25, Qwen2.5-1.5B | 0.148 | 0.300 | 0.533 | 88.433 | 0.033 |
| Case RAG, BM25 top-1 prompt, Qwen2.5-1.5B | 0.209 | 0.300 | 0.533 | 52.900 | 0.000 |
| Case RAG, hybrid top-1 prompt, Qwen2.5-1.5B | 0.218 | 0.333 | 0.600 | 57.633 | 0.000 |
| LLM-only, Qwen2.5-1.5B, full 360 | 0.091 | 0.000 | 0.000 | 65.544 | 0.000 |
| Report RAG, BM25, Qwen2.5-1.5B, full 360 | 0.146 | 0.231 | 0.383 | 84.464 | 0.008 |
| Case RAG, BM25 top-1 prompt, Qwen2.5-1.5B, full 360 | 0.188 | 0.231 | 0.383 | 53.297 | 0.000 |
| Case RAG, hybrid top-1 prompt, Qwen2.5-1.5B, full 360 | 0.209 | 0.242 | 0.422 | 52.139 | 0.000 |

## Agentic Evidence-Checking Results

The stricter agentic run uses only the top-1 retrieved case as evidence. This avoids a failure mode where the model copies claims from several retrieved cases and receives false support because all copied claims appear somewhere in the top-k evidence pool.

| System | Draft Token-F1 | Final Token-F1 | Evidence Support | Revision Rate | Abstention Rate | Unsupported Sentence Rate |
|---|---:|---:|---:|---:|---:|---:|
| Report RAG, BM25 + top-1 check | 0.187 | 0.143 | 0.293 | 0.833 | 0.600 | 0.709 |
| Case RAG, BM25 + top-1 check | 0.163 | 0.144 | 0.241 | 0.900 | 0.533 | 0.750 |
| Case RAG, hybrid + top-1 check | 0.187 | 0.133 | 0.156 | 0.933 | 0.667 | 0.871 |
| Report RAG, BM25, Qwen2.5-1.5B + top-1 check | 0.148 | 0.134 | 0.098 | 1.000 | 0.700 | 0.844 |
| Case RAG, BM25 top-1 prompt, Qwen2.5-1.5B + top-1 check | 0.209 | 0.169 | 0.358 | 0.867 | 0.300 | 0.648 |
| Case RAG, hybrid top-1 prompt, Qwen2.5-1.5B + top-1 check | 0.218 | 0.161 | 0.300 | 0.900 | 0.333 | 0.705 |
| Report RAG, BM25, Qwen2.5-1.5B, full 360 + top-1 check | 0.146 | 0.093 | 0.118 | 0.992 | 0.681 | 0.819 |
| Case RAG, BM25 top-1 prompt, Qwen2.5-1.5B, full 360 + top-1 check | 0.188 | 0.139 | 0.374 | 0.833 | 0.325 | 0.674 |
| Case RAG, hybrid top-1 prompt, Qwen2.5-1.5B, full 360 + top-1 check | 0.209 | 0.145 | 0.305 | 0.886 | 0.419 | 0.711 |

## Key Observation

The pilot exposed a cross-case contamination issue. In one example, the hybrid case-RAG prompt retrieved CXR1027 as the top case, but the small LLM copied impressions from several retrieved cases into one answer. A broad all-evidence checker considered those extra claims supported because they appeared in the top-5 evidence. The stricter top-1 checker correctly removed the extra impressions and kept only the CXR1027-supported claim.

This is a useful thesis finding: evidence checking must be case-specific, not just top-k-document supported. Otherwise, a medical RAG system can produce a plausible but clinically mixed answer.

The top-1 structured prompt reduces this problem at the prompt level by giving the generator only the selected top-ranked case evidence while still retaining top-5 retrieval metadata for retrieval evaluation. With Qwen2.5-1.5B, BM25 top-1 prompting reached token-F1 of 0.209 and hybrid top-1 prompting reached token-F1 of 0.218. Hybrid retrieval also improved retrieved hit rate from 0.533 to 0.600 in the 30-item pilot.

In the full 360-question runs, LLM-only Qwen2.5-1.5B reached token-F1 of 0.091, report-RAG BM25 reached 0.146, case-RAG BM25 top-1 reached 0.188, and case-RAG hybrid top-1 reached 0.209. The lower retrieval coverage compared with the pilot indicates that the 30-question pilot was optimistic, so the full-run results should be used as the more conservative local baseline.

## Important Limitation

The 0.5B and 1.5B models are too small for final answer-quality claims. The final thesis experiment should rerun this comparison with a stronger instruction model, preferably Qwen2.5-7B-Instruct with quantization/offload or another supervisor-approved model. The small-model pilots should be described as pipeline validation and error-analysis pilots.

## Output Files

```text
experiments/generations_llm_only_qwen05_pilot30.jsonl
experiments/generations_report_rag_bm25_qwen05_pilot30.jsonl
experiments/generations_case_rag_bm25_qwen05_pilot30.jsonl
experiments/generations_case_rag_hybrid_qwen05_pilot30.jsonl
experiments/generations_case_rag_hybrid_qwen05_pilot30_agentic_top1.jsonl
experiments/generations_llm_only_qwen15_pilot30.jsonl
experiments/generations_report_rag_bm25_qwen15_pilot30.jsonl
experiments/generations_case_rag_bm25_top1_qwen15_pilot30.jsonl
experiments/generations_case_rag_hybrid_top1_qwen15_pilot30.jsonl
experiments/generations_case_rag_bm25_top1_qwen15_pilot30_agentic_top1.jsonl
experiments/generations_case_rag_hybrid_top1_qwen15_pilot30_agentic_top1.jsonl
experiments/generations_llm_only_qwen15_full360.jsonl
experiments/generations_llm_only_qwen15_full360_eval.json
experiments/generations_report_rag_bm25_qwen15_full360.jsonl
experiments/generations_report_rag_bm25_qwen15_full360_eval.json
experiments/generations_report_rag_bm25_qwen15_full360_agentic_top1.jsonl
experiments/generations_report_rag_bm25_qwen15_full360_agentic_top1_eval.json
experiments/generations_case_rag_bm25_top1_qwen15_full360.jsonl
experiments/generations_case_rag_bm25_top1_qwen15_full360_eval.json
experiments/generations_case_rag_bm25_top1_qwen15_full360_agentic_top1.jsonl
experiments/generations_case_rag_bm25_top1_qwen15_full360_agentic_top1_eval.json
experiments/generations_case_rag_hybrid_top1_qwen15_full360.jsonl
experiments/generations_case_rag_hybrid_top1_qwen15_full360_eval.json
experiments/generations_case_rag_hybrid_top1_qwen15_full360_agentic_top1.jsonl
experiments/generations_case_rag_hybrid_top1_qwen15_full360_agentic_top1_eval.json
experiments/manual_annotation_hybrid_qwen15_full360_agentic_top1_sample50.csv
```
