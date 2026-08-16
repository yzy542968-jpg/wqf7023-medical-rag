# P1 to P2 Alignment and Dashboard Data Plan

Created: 2026-07-08

## Final Thesis Direction

Recommended final title:

`Evidence-Checked Case-Level Retrieval-Augmented Generation for Radiology Report-Grounded Medical Question Answering`

The safest thesis framing is not that P2 replaces P1. The final project should be presented as a continuation:

1. P1 established a report-grounded medical RAG problem over paired radiology image-report cases.
2. P1 implemented and evaluated retrieval-unit choices: chunk-level, report-level, and case-level RAG.
3. P2 extends the P1 system with evidence checking, because retrieval alone does not guarantee supported answers.
4. The final contribution is case-level RAG with post-generation evidence checking for safer radiology report-grounded QA.

## Asset Map

| Asset | Location | Role in Final Project |
|---|---|---|
| P1 official materials | `C:\Users\yz542_dntjhas\Desktop\2025-2026-s2\P1` | Confirms approved title, school requirements, timeline, forms, and P1 submission constraints. |
| P1 engineering project | `C:\Users\yz542_dntjhas\Documents\New project 2\radiology-rag` | Main P1-aligned implementation for retrieval-unit comparison. |
| Current P2 engineering project | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag` | Evidence-checking / agentic extension and final dashboard base. |
| P1 final draft area | `C:\Users\yz542_dntjhas\Desktop\2025-2026-s2\P1` | Existing P1 report drafts and filled chapters are under the P1 thesis draft subfolder. |

Important note: `C:\Users\yz542_dntjhas\Documents\New project 2\radiology-rag\README.md` is stale. It says the project only reached Stage 0/1, but the actual code and result files show that it reached Stage 8B with full generation and evaluation outputs.

## P1 Engineering Evidence

P1 project root:

`C:\Users\yz542_dntjhas\Documents\New project 2\radiology-rag`

Verified data-processing evidence:

| Evidence | Value |
|---|---:|
| Raw XML files processed | 3,955 |
| Kept report cases | 3,927 |
| Skipped empty reports | 28 |
| Cases with linked images | 3,826 |
| Cases without images | 101 |
| Report units | 3,927 |
| Chunk units | 7,817 |
| Case units | 3,927 |
| Verified evaluation items | 100 |
| Stage 8B generations | 1,200 |

Core P1 systems:

| System | Thesis Purpose |
|---|---|
| LLM-only | Shows the baseline without retrieved evidence. |
| Chunk-RAG | Tests whether small evidence chunks fragment report meaning. |
| Report-RAG | Baseline RAG over full radiology report text. |
| Case-level RAG | P1 proposed method, preserving case boundary and traceability. |

Key P1 result files:

| File | Use |
|---|---|
| `data\processed\data_prep_summary.json` | Dataset preparation statistics. |
| `data\processed\corpus_summary.json` | Report/chunk/case corpus statistics. |
| `data\evaluation_set\eval_set_verified_summary.json` | 100-item verified evaluation set composition. |
| `results\generations\stage8b_case_conditioned_generations.jsonl` | 1,200 generated outputs. |
| `results\evaluation\stage8b_case_conditioned_metrics.json` | Main Stage 8B metrics. |
| `results\evaluation\stage8b_system_comparison.json` | System-level comparison. |
| `results\evaluation\stage8b_prompt_comparison.json` | Prompt-level comparison. |
| `results\evaluation\stage8b_failure_mode_counts.json` | Failure mode counts. |
| `results\evaluation\stage8b_case_conditioned_error_examples.json` | Representative P1 success and failure examples. |
| `results\evaluation\stage8b_case_conditioned_item_scores.csv` | Per-output evaluation rows for dashboard filtering. |

P1 Stage 8B model evidence:

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Loading mode | `cuda_4bit_bitsandbytes` |
| Device | `cuda:0` |
| Expected generations | 1,200 |
| Completed generations | 1,200 |
| Generation failures | 0 |

P1 findings that should be used carefully:

1. Report-RAG and case-level RAG have nearly identical automatic scores when both use the same report text. This is not a problem. It means case-level RAG should be framed as improving case-boundary preservation and traceability, not necessarily raw answer accuracy.
2. Chunk-RAG shows more fragmentation indicators, including `chunk_context_missing_full_span = 9` in the system comparison.
3. Some P1 automatic scores are very high, so final claims should mention that these are rule-level and partially manual-review-dependent metrics.
4. P1 is strongest as the Stage A baseline experiment, not as the whole final thesis contribution.

## P2 Engineering Evidence

P2 project root:

`C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag`

Current P2 contribution:

`Case-level RAG + top-1 evidence checking`

P2 system comparison:

| System | Role |
|---|---|
| LLM-only Qwen2.5-1.5B | No-retrieval baseline. |
| Report-RAG BM25 | Report-level retrieval baseline. |
| Case-RAG BM25 top-1 | Case-boundary prompting baseline. |
| Case-RAG Hybrid top-1 | Hybrid BM25 + MedCPT case-level retrieval. |
| Evidence checker | P2 proposed safety-oriented extension. |

Key P2 result files:

| File | Use |
|---|---|
| `docs\FULL360_RESULTS_ANALYSIS.md` | Human-readable summary of the 360-question results. |
| `docs\FULL_PROJECT_COMPLETION_PLAN.md` | P2 completion plan and defensible research claim. |
| `experiments\full360_analysis\full360_per_answer_metrics.csv` | Main per-answer dashboard table. |
| `experiments\full360_analysis\full360_grouped_metrics.json` | Grouped metrics by system/question type. |
| `experiments\full360_analysis\full360_error_analysis.md` | Error analysis summary. |
| `experiments\manual_annotation_qwen15_full360_comparative_sample50.csv` | 50-question comparative manual annotation sheet. |
| `experiments\generations_*_qwen15_full360*.jsonl` | Full generated answer records. |
| `experiments\generations_*_agentic_top1_eval.json` | Evidence-checking summary metrics. |
| `data\processed\openi_cases.jsonl` | Case browser input for dashboard. |

Current full 360-question automatic generation results:

| System | Token-F1 | Top-1 Case Accuracy | Retrieved Hit Rate |
|---|---:|---:|---:|
| LLM-only Qwen2.5-1.5B | 0.091 | 0.000 | 0.000 |
| Report-RAG BM25 Qwen2.5-1.5B | 0.146 | 0.231 | 0.383 |
| Case-RAG BM25 top-1 Qwen2.5-1.5B | 0.188 | 0.231 | 0.383 |
| Case-RAG Hybrid top-1 Qwen2.5-1.5B | 0.209 | 0.242 | 0.422 |

P2 top-k retrieval results were already computed over the 360-question clean QA seed. Generation and evidence checking mainly use top-1, but the retrieval stage has top-k metrics and top-20 ranked results.

| Retriever | Alpha | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF | N/A | 0.167 | 0.239 | 0.281 | 0.339 | 0.400 | 0.220 |
| BM25 | N/A | 0.231 | 0.344 | 0.383 | 0.428 | 0.486 | 0.297 |
| MedCPT dense retrieval | N/A | 0.108 | 0.197 | 0.253 | 0.300 | 0.406 | 0.172 |
| Hybrid BM25 + MedCPT | 0.25 | 0.247 | 0.361 | 0.403 | 0.453 | 0.506 | 0.317 |
| Hybrid BM25 + MedCPT | 0.50 | 0.242 | 0.378 | 0.422 | 0.478 | 0.553 | 0.324 |
| Hybrid BM25 + MedCPT | 0.75 | 0.228 | 0.344 | 0.397 | 0.478 | 0.558 | 0.311 |

Current evidence-checking results:

| System | Draft Token-F1 | Final Token-F1 | Evidence Support | Revision Rate | Abstention Rate | Unsupported Sentence Rate |
|---|---:|---:|---:|---:|---:|---:|
| Report-RAG BM25 | 0.146 | 0.093 | 0.118 | 0.992 | 0.681 | 0.819 |
| Case-RAG BM25 top-1 | 0.188 | 0.139 | 0.374 | 0.833 | 0.325 | 0.674 |
| Case-RAG Hybrid top-1 | 0.209 | 0.145 | 0.305 | 0.886 | 0.419 | 0.711 |

P2 thesis claim:

Case-level RAG improves answer quality over LLM-only and report-level RAG, but generated answers still contain unsupported claims. Evidence checking is therefore necessary as a safety-oriented post-generation step.

## Final Combined Research Questions

| Research Question | Supported By |
|---|---|
| RQ1: Does retrieval improve radiology report-grounded QA compared with LLM-only generation? | P1 Stage 8B and P2 full360 experiments. |
| RQ2: How do chunk-level, report-level, and case-level retrieval units affect correctness, faithfulness, and traceability? | P1 Stage 8B retrieval-unit comparison. |
| RQ3: When does chunk retrieval fragment clinically relevant evidence? | P1 chunk fragmentation indicators and error examples. |
| RQ4: Does top-1 case-level evidence checking reduce unsupported answers and expose cross-case contamination? | P2 evidence-checking and full360 error analysis. |

## Final Thesis Structure

| Chapter | Recommended Content |
|---|---|
| Chapter 1 | Medical RAG problem, hallucination, traceability, case-boundary motivation. |
| Chapter 2 | RAG, medical QA, radiology reports, retrieval units, faithfulness/evidence checking. |
| Chapter 3 | Methodology: dataset, P1 retrieval-unit systems, P2 evidence-checking extension. |
| Chapter 4 | Implementation: data processing, retrieval, generation, evaluation, dashboard. |
| Chapter 5 | Results: Stage A P1 retrieval-unit comparison and Stage B P2 evidence-checking results. |
| Chapter 6 | Discussion: why case boundaries and evidence checking matter; limitations and ethics. |
| Chapter 7 | Conclusion and future work. |

## Dashboard MVP

The dashboard should be a research demonstration, not a clinical upload product.

Recommended tool:

`Streamlit`

Recommended command:

`streamlit run dashboard/app.py`

Dashboard pages:

| Page | Purpose | Primary Data |
|---|---|---|
| Overview | Show final research pipeline and key metrics. | P1/P2 summary JSON and CSV files. |
| Dataset Browser | Browse report cases, linked image references, and evidence spans. | P1 `cases.jsonl` or P2 `openi_cases.jsonl`. |
| Top-k Retrieval Explorer | Show retrieved case rankings, Hit@k behavior, and whether the ground-truth case appears in top-k but not top-1. | P2 retrieval evaluation JSON files. |
| Retrieval Unit Comparison | Compare LLM-only, chunk-RAG, report-RAG, and case-level RAG. | P1 Stage 8B item scores and system comparison. |
| Evidence Checking Demo | Compare draft answers and final evidence-checked answers. | P2 full360 per-answer metrics and generation JSONL files. |
| Failure Gallery | Show selected success/failure cases with explanation. | P1 error examples and P2 representative cases. |

Dashboard should use precomputed files only. It should not run Qwen live during the defense.

## Dashboard Data Sources

P1 dashboard data:

| Need | File |
|---|---|
| Dataset counts | `C:\Users\yz542_dntjhas\Documents\New project 2\radiology-rag\data\processed\data_prep_summary.json` |
| Corpus counts | `C:\Users\yz542_dntjhas\Documents\New project 2\radiology-rag\data\processed\corpus_summary.json` |
| Evaluation set summary | `C:\Users\yz542_dntjhas\Documents\New project 2\radiology-rag\data\evaluation_set\eval_set_verified_summary.json` |
| Stage 8B metrics | `C:\Users\yz542_dntjhas\Documents\New project 2\radiology-rag\results\evaluation\stage8b_case_conditioned_metrics.json` |
| System comparison | `C:\Users\yz542_dntjhas\Documents\New project 2\radiology-rag\results\evaluation\stage8b_system_comparison.json` |
| Prompt comparison | `C:\Users\yz542_dntjhas\Documents\New project 2\radiology-rag\results\evaluation\stage8b_prompt_comparison.json` |
| Failure examples | `C:\Users\yz542_dntjhas\Documents\New project 2\radiology-rag\results\evaluation\stage8b_case_conditioned_error_examples.json` |
| Per-row scores | `C:\Users\yz542_dntjhas\Documents\New project 2\radiology-rag\results\evaluation\stage8b_case_conditioned_item_scores.csv` |
| Generated answers | `C:\Users\yz542_dntjhas\Documents\New project 2\radiology-rag\results\generations\stage8b_case_conditioned_generations.jsonl` |

P2 dashboard data:

| Need | File |
|---|---|
| Case browser | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\data\processed\openi_cases.jsonl` |
| Main per-answer metrics | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\full360_analysis\full360_per_answer_metrics.csv` |
| Grouped metrics | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\full360_analysis\full360_grouped_metrics.json` |
| Error analysis | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\full360_analysis\full360_error_analysis.md` |
| TF-IDF top-k retrieval | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\tfidf_qa_seed_clean_eval.json` |
| BM25 top-k retrieval | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\bm25_qa_seed_clean_eval.json` |
| MedCPT top-k retrieval | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\medcpt_qa_seed_clean_eval.json` |
| Hybrid alpha 0.25 top-k retrieval | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\hybrid_bm25_medcpt_a025_qa_seed_clean_eval.json` |
| Hybrid alpha 0.50 top-k retrieval | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\hybrid_bm25_medcpt_a050_qa_seed_clean_eval.json` |
| Hybrid alpha 0.75 top-k retrieval | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\hybrid_bm25_medcpt_a075_qa_seed_clean_eval.json` |
| Comparative manual annotation sample | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\manual_annotation_qwen15_full360_comparative_sample50.csv` |
| LLM-only generations | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\generations_llm_only_qwen15_full360.jsonl` |
| Report-RAG generations | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\generations_report_rag_bm25_qwen15_full360.jsonl` |
| Case-RAG BM25 generations | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\generations_case_rag_bm25_top1_qwen15_full360.jsonl` |
| Case-RAG Hybrid generations | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\generations_case_rag_hybrid_top1_qwen15_full360.jsonl` |
| Evidence-checked Hybrid generations | `C:\Users\yz542_dntjhas\Documents\New project\wqf7023-medical-rag\experiments\generations_case_rag_hybrid_top1_qwen15_full360_agentic_top1.jsonl` |

## Demo Case Candidates

Use a small set of preselected cases rather than random upload.

P2 candidates from `FULL360_RESULTS_ANALYSIS.md` and `full360_per_answer_metrics.csv`:

| Demo Type | QID | Why It Is Useful |
|---|---|---|
| Clean success | `CXR2721_impression` | Case-RAG BM25 and Hybrid top-1 reach high answer Token-F1 and full evidence support. |
| Lexical match but weak support | `CXR533_impression` | Report-RAG has high Token-F1 but low evidence support, showing why answer overlap is not enough. |
| Retrieval miss | `CXR1027_summary` | Relevant case is missed and answer quality is poor. |
| Evidence checker helps | `CXR1027_impression` | Case-level top-1 answer is supported while report-RAG has many unsupported sentences. |
| Hybrid failure | `CXR1054_impression` | Hybrid retrieves the true case in top-k but not top-1, showing top-1 evidence-scope limitations. |

P1 candidates from `stage8b_case_conditioned_error_examples.json`:

| Demo Type | Eval ID | System | Why It Is Useful |
|---|---|---|---|
| Chunk incomplete evidence | `eval_0002` | Chunk-RAG | Demonstrates incomplete evidence span in chunk retrieval. |
| Over-abstention | `eval_0094` | Chunk-RAG evidence-guided | Shows a case where available evidence exists but the system abstains. |
| Case-level negation success | `eval_0086` | Case-level RAG | Shows correct negation handling for pneumothorax. |
| Location/attribute success | `eval_0151` | Case-level RAG | Shows a case-level answer grounded in report location evidence. |
| Polarity mismatch | `eval_0024` | Report-RAG | Shows a false answer caused by polarity mismatch. |

## Dashboard Non-Goals

Do not make the defense demo about arbitrary medical upload.

Avoid:

1. Real-time Qwen 7B generation during defense.
2. Uploading new patient data.
3. Presenting the system as a clinical diagnostic tool.
4. Claiming that linked X-ray images are used for direct image diagnosis.

Allowed:

1. Preloaded IU X-Ray/OpenI case selector.
2. Optional pasted report text for research demonstration only.
3. Clear label: `Research demo only. Not for clinical diagnosis.`
4. Dashboard screenshots in thesis and slides.

## Immediate Next Tasks

1. Create `dashboard/` under the current P2 project.
2. Add a lightweight data-loading layer that reads P1 and P2 precomputed JSON/CSV/JSONL files.
3. Build the Overview page first with dataset counts, system comparison, and evidence-checking metrics.
4. Build the Demo Case page with the five preselected P2 cases.
5. Add the P1 Retrieval Unit Comparison page after the data loader is stable.
6. Add screenshots to the final report and presentation.
