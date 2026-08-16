# Experiment Summary

| Dataset | Retriever | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Keyword qrels | TF-IDF | 0.500 | 0.875 | 0.875 | 0.875 | 0.875 | 0.667 | `experiments\tfidf_keyword_eval.json` |
| Keyword qrels | BM25 | 0.500 | 0.875 | 0.875 | 1.000 | 1.000 | 0.688 | `experiments\bm25_keyword_eval.json` |
| QA seed | TF-IDF | 0.189 | 0.267 | 0.281 | 0.308 | 0.342 | 0.232 | `experiments\tfidf_qa_seed_eval.json` |
| QA seed | BM25 | 0.233 | 0.286 | 0.303 | 0.322 | 0.344 | 0.264 | `experiments\bm25_qa_seed_eval.json` |
| Clean QA seed | TF-IDF | 0.167 | 0.239 | 0.281 | 0.339 | 0.400 | 0.220 | `experiments\tfidf_qa_seed_clean_eval.json` |
| Clean QA seed | BM25 | 0.231 | 0.344 | 0.383 | 0.428 | 0.486 | 0.297 | `experiments\bm25_qa_seed_clean_eval.json` |
| Clean QA seed | MedCPT | 0.108 | 0.197 | 0.253 | 0.300 | 0.406 | 0.172 | `experiments\medcpt_qa_seed_clean_eval.json` |
| Clean QA seed | Hybrid a=0.25 | 0.247 | 0.361 | 0.403 | 0.453 | 0.506 | 0.317 | `experiments\hybrid_bm25_medcpt_a025_qa_seed_clean_eval.json` |
| Clean QA seed | Hybrid a=0.50 | 0.242 | 0.378 | 0.422 | 0.478 | 0.553 | 0.324 | `experiments\hybrid_bm25_medcpt_a050_qa_seed_clean_eval.json` |
| Clean QA seed | Hybrid a=0.75 | 0.228 | 0.344 | 0.397 | 0.478 | 0.558 | 0.311 | `experiments\hybrid_bm25_medcpt_a075_qa_seed_clean_eval.json` |

Note: `N/A` means the experiment has not been run in the current environment.

## Extractive Answer Baselines

| Dataset | System | Answer Token-F1 | Top-1 Case Accuracy | Retrieved Hit Rate | Evidence Support | Revision Rate | Abstention Rate | Non-Empty Answer Rate | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| QA seed | Extractive TF-IDF RAG | 0.488 | 0.189 | N/A | N/A | N/A | N/A | 0.986 | `experiments\extractive_tfidf_qa_seed_answers.json` |
| QA seed | Extractive BM25 RAG | 0.341 | 0.233 | N/A | N/A | N/A | N/A | 0.967 | `experiments\extractive_bm25_qa_seed_answers.json` |
| Clean QA seed | Extractive TF-IDF RAG | 0.401 | 0.167 | N/A | N/A | N/A | N/A | 0.981 | `experiments\extractive_tfidf_qa_seed_clean_answers.json` |
| Clean QA seed | Extractive BM25 RAG | 0.364 | 0.231 | N/A | N/A | N/A | N/A | 0.981 | `experiments\extractive_bm25_qa_seed_clean_answers.json` |
| Clean QA seed | Agentic Hybrid RAG a=0.50 | 0.412 | 0.242 | 0.422 | 0.992 | 0.008 | 0.008 | 1.000 | `experiments\agentic_hybrid_a050_qa_seed_clean_answers.json` |

## Qwen2.5 Generation Results

| Dataset | System | Answer Token-F1 | Top-1 Case Accuracy | Retrieved Hit Rate | Avg Answer Words | Insufficient Rate | Source |
|---|---|---:|---:|---:|---:|---:|---|
| Pilot 30 | LLM-only Qwen2.5-0.5B | 0.078 | 0.000 | 0.000 | 84.367 | 0.000 | `experiments\generations_llm_only_qwen05_pilot30_eval.json` |
| Pilot 30 | Report RAG BM25 Qwen2.5-0.5B | 0.187 | 0.300 | 0.533 | 60.633 | 0.033 | `experiments\generations_report_rag_bm25_qwen05_pilot30_eval.json` |
| Pilot 30 | Case RAG BM25 Qwen2.5-0.5B | 0.163 | 0.300 | 0.533 | 83.600 | 0.000 | `experiments\generations_case_rag_bm25_qwen05_pilot30_eval.json` |
| Pilot 30 | Case RAG Hybrid Qwen2.5-0.5B | 0.187 | 0.333 | 0.600 | 80.867 | 0.000 | `experiments\generations_case_rag_hybrid_qwen05_pilot30_eval.json` |
| Pilot 30 | LLM-only Qwen2.5-1.5B | 0.097 | 0.000 | 0.000 | 63.567 | 0.000 | `experiments\generations_llm_only_qwen15_pilot30_eval.json` |
| Pilot 30 | Report RAG BM25 Qwen2.5-1.5B | 0.148 | 0.300 | 0.533 | 88.433 | 0.033 | `experiments\generations_report_rag_bm25_qwen15_pilot30_eval.json` |
| Pilot 30 | Case RAG BM25 top-1 Qwen2.5-1.5B | 0.209 | 0.300 | 0.533 | 52.900 | 0.000 | `experiments\generations_case_rag_bm25_top1_qwen15_pilot30_eval.json` |
| Pilot 30 | Case RAG Hybrid top-1 Qwen2.5-1.5B | 0.218 | 0.333 | 0.600 | 57.633 | 0.000 | `experiments\generations_case_rag_hybrid_top1_qwen15_pilot30_eval.json` |
| Full 360 | LLM-only Qwen2.5-1.5B | 0.091 | 0.000 | 0.000 | 65.544 | 0.000 | `experiments\generations_llm_only_qwen15_full360_eval.json` |
| Full 360 | Report RAG BM25 Qwen2.5-1.5B | 0.146 | 0.231 | 0.383 | 84.464 | 0.008 | `experiments\generations_report_rag_bm25_qwen15_full360_eval.json` |
| Full 360 | Case RAG BM25 top-1 Qwen2.5-1.5B | 0.188 | 0.231 | 0.383 | 53.297 | 0.000 | `experiments\generations_case_rag_bm25_top1_qwen15_full360_eval.json` |
| Full 360 | Case RAG Hybrid top-1 Qwen2.5-1.5B | 0.209 | 0.242 | 0.422 | 52.139 | 0.000 | `experiments\generations_case_rag_hybrid_top1_qwen15_full360_eval.json` |

## Agentic Evidence-Checking Results

| Dataset | System | Draft Token-F1 | Final Token-F1 | Evidence Support | Revision Rate | Abstention Rate | Unsupported Sentence Rate | Source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Pilot 30 | Report RAG BM25 + top-1 evidence check | 0.187 | 0.143 | 0.293 | 0.833 | 0.600 | 0.709 | `experiments\generations_report_rag_bm25_qwen05_pilot30_agentic_top1_eval.json` |
| Pilot 30 | Case RAG BM25 + top-1 evidence check | 0.163 | 0.144 | 0.241 | 0.900 | 0.533 | 0.750 | `experiments\generations_case_rag_bm25_qwen05_pilot30_agentic_top1_eval.json` |
| Pilot 30 | Case RAG Hybrid + top-1 evidence check | 0.187 | 0.133 | 0.156 | 0.933 | 0.667 | 0.871 | `experiments\generations_case_rag_hybrid_qwen05_pilot30_agentic_top1_eval.json` |
| Pilot 30 | Case RAG Hybrid top-1 Qwen2.5-1.5B + top-1 evidence check | 0.218 | 0.161 | 0.300 | 0.900 | 0.333 | 0.705 | `experiments\generations_case_rag_hybrid_top1_qwen15_pilot30_agentic_top1_eval.json` |
| Pilot 30 | Report RAG BM25 Qwen2.5-1.5B + top-1 evidence check | 0.148 | 0.134 | 0.098 | 1.000 | 0.700 | 0.844 | `experiments\generations_report_rag_bm25_qwen15_pilot30_agentic_top1_eval.json` |
| Pilot 30 | Case RAG BM25 top-1 Qwen2.5-1.5B + top-1 evidence check | 0.209 | 0.169 | 0.358 | 0.867 | 0.300 | 0.648 | `experiments\generations_case_rag_bm25_top1_qwen15_pilot30_agentic_top1_eval.json` |
| Full 360 | Report RAG BM25 Qwen2.5-1.5B + top-1 evidence check | 0.146 | 0.093 | 0.118 | 0.992 | 0.681 | 0.819 | `experiments\generations_report_rag_bm25_qwen15_full360_agentic_top1_eval.json` |
| Full 360 | Case RAG BM25 top-1 Qwen2.5-1.5B + top-1 evidence check | 0.188 | 0.139 | 0.374 | 0.833 | 0.325 | 0.674 | `experiments\generations_case_rag_bm25_top1_qwen15_full360_agentic_top1_eval.json` |
| Full 360 | Case RAG Hybrid top-1 Qwen2.5-1.5B + top-1 evidence check | 0.209 | 0.145 | 0.305 | 0.886 | 0.419 | 0.711 | `experiments\generations_case_rag_hybrid_top1_qwen15_full360_agentic_top1_eval.json` |