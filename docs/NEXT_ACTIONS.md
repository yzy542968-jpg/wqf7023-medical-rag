# Next Actions

> Superseded implementation checklist. The optimized experiment is frozen as of 2026-08-14. The only empirical action that remains is independent completion of `experiments/final_optimized/human_evaluation/held_out_blinded_human_evaluation_36.csv`, followed by unblinding and reporting. Do not retune the system from those labels. Use `docs/FINAL_OPTIMIZED_RESEARCH_RESULTS.md` and `docs/HELD_OUT_HUMAN_EVALUATION_GUIDE.md` as the current sources.

## Completed P1 Deliverables

1. Editable P1 presentation deck completed:
   - `deliverables/p1-medical-rag-proposal.pptx`
2. Presentation source workspace and rendered previews:
   - `outputs/manual-20260605-p1-deck/presentations/p1-medical-rag/`
3. Deck QA scorecard:
   - `outputs/manual-20260605-p1-deck/presentations/p1-medical-rag/qa/comeback-scorecard.txt`
4. Full 360-question hybrid top-1 Qwen2.5-1.5B generation:
   - `experiments/generations_case_rag_hybrid_top1_qwen15_full360.jsonl`
   - `experiments/generations_case_rag_hybrid_top1_qwen15_full360_eval.json`
5. Full 360-question comparison baselines:
   - `experiments/generations_llm_only_qwen15_full360_eval.json`
   - `experiments/generations_report_rag_bm25_qwen15_full360_eval.json`
   - `experiments/generations_case_rag_bm25_top1_qwen15_full360_eval.json`
6. Full 360-question top-1 evidence checking:
   - `experiments/generations_report_rag_bm25_qwen15_full360_agentic_top1_eval.json`
   - `experiments/generations_case_rag_bm25_top1_qwen15_full360_agentic_top1_eval.json`
   - `experiments/generations_case_rag_hybrid_top1_qwen15_full360_agentic_top1.jsonl`
   - `experiments/generations_case_rag_hybrid_top1_qwen15_full360_agentic_top1_eval.json`
7. Manual annotation sheet for the full-run sample:
   - `experiments/manual_annotation_hybrid_qwen15_full360_agentic_top1_sample50.csv`
8. Comparative manual annotation sheet across four full-run systems:
   - `experiments/manual_annotation_qwen15_full360_comparative_sample50.csv`
9. P1 formal report DOCX:
   - `deliverables/p1-formal-report.docx`
   - Structural QA: `outputs/manual-20260605-p1-report-docx/qa/p1-report-docx-qa.txt`
10. Full360 automated error analysis:
   - `experiments/full360_analysis/full360_error_analysis.md`
   - `experiments/full360_analysis/full360_per_answer_metrics.csv`
11. Manual annotation guide:
   - `docs/MANUAL_ANNOTATION_GUIDE.md`

## Student-Side Confirmations

These require your access or supervisor communication:

1. Confirm that the title and supervisor appointment were approved by faculty.
2. Confirm whether your supervisor accepts the current title wording.
3. Confirm the expected P1 report submission deadline, because the timeline PDF shows the presentation window clearly but the report submission deadline is not visible in the extracted table.
4. Confirm whether the supervisor expects a working prototype during P1 or only a proposal.

## Data and Compute

1. Download the IU X-Ray / OpenI files into `data/raw/`. Done for metadata.
2. Confirm the filenames and columns match. Done:
   - `indiana_reports.csv`
   - `indiana_projections.csv`
   - image folder
3. Download a small real image subset. Done for 5 pneumonia cases / 10 images.
4. Current local dense-retrieval dependencies:
   - `torch`: installed
   - `transformers`: installed
   - `sentence_transformers`: not required for current MedCPT path
   - `chromadb`: optional, not required for current NumPy index path
5. Decide the compute route for Qwen2.5-7B-Instruct:
   - local GPU, if available;
   - cloud notebook;
   - university machine;
   - smaller fallback model for prototype testing.

## Project Work I Can Continue Next

1. Fill the 50-question comparative manual annotation sheet for relevance, evidence support, hallucination control, completeness, and case contamination.
2. Write the full error-analysis section using full360 outputs.
3. Review the generated P1 formal report DOCX; visual render QA could not be completed because the conversion executable was unavailable.
4. Review and rehearse the completed P1 presentation deck.
5. Optionally test a stronger 7B/quantized/cloud model if supervisor expects a stronger LLM.
6. Prepare a short supervisor-facing explanation of the refined title and contribution.

## Current Working Baseline

The repository already contains a minimal case-level retrieval prototype:

```powershell
python scripts\build_cases.py --reports-csv data\raw\indiana_reports.csv --projections-csv data\raw\indiana_projections.csv --output data\processed\openi_cases.jsonl
python scripts\evaluate_tfidf_retrieval.py --cases data\processed\openi_cases.jsonl --qrels data\processed\openi_keyword_qrels.json --top-k 20 --output experiments\tfidf_keyword_eval.json
python scripts\evaluate_bm25_retrieval.py --cases data\processed\openi_cases.jsonl --qrels data\processed\openi_keyword_qrels.json --top-k 20 --output experiments\bm25_keyword_eval.json
```

This prototype is intentionally simple. Its purpose is to prove the project pipeline before adding heavier dependencies.

## Dense Retrieval Commands

Full MedCPT indexing has already been completed locally. The current full-index commands are:

```powershell
.venv\Scripts\python.exe scripts\build_medcpt_index.py --cases data\processed\openi_cases.jsonl --output data\processed\openi_medcpt_full.npz --batch-size 4 --device cpu
.venv\Scripts\python.exe scripts\evaluate_hybrid_qa_retrieval.py --cases data\processed\openi_cases.jsonl --index data\processed\openi_medcpt_full.npz --qa data\processed\openi_case_qa_seed_clean.json --top-k 20 --alpha 0.50 --output experiments\hybrid_bm25_medcpt_a050_qa_seed_clean_eval.json --device cpu
```

## Agentic Baseline Command

```powershell
.venv\Scripts\python.exe scripts\run_agentic_rag_baseline.py --cases data\processed\openi_cases.jsonl --index data\processed\openi_medcpt_full.npz --qa data\processed\openi_case_qa_seed_clean.json --top-k 5 --batch-size 16 --device cpu --alpha 0.50 --output experiments\agentic_hybrid_a050_qa_seed_clean_answers.json
```

## Completed Local LLM Pilot

The local RTX 5070 Laptop GPU can run Qwen2.5-1.5B-Instruct with CUDA PyTorch. The strongest 30-question local pilot so far is:

```text
experiments/generations_case_rag_hybrid_top1_qwen15_pilot30_eval.json
```

Key result:

- Answer token-F1: 0.218
- Top-1 case accuracy: 0.333
- Retrieved case hit rate: 0.600
- Average answer words: 57.633

## Completed P1 Presentation Outline

```text
docs/P1_PRESENTATION_OUTLINE.md
```

## Completed P1 Presentation Deck

```text
deliverables/p1-medical-rag-proposal.pptx
```
