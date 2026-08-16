# Full Project Completion Plan

Current date: 2026-06-05

This plan turns the current prototype into a complete WQF7023 research project. The project should be managed as three deliverable layers: proposal readiness, final experiment readiness, and final thesis readiness.

## Current Status

Completed:

1. Real OpenI / IU X-Ray metadata processed into 3,851 case-level records.
2. 7,466 linked image-projection mappings retained.
3. Clean QA seed created: 120 cases and 360 questions.
4. TF-IDF, BM25, MedCPT, and hybrid BM25 + MedCPT retrieval evaluated.
5. Full MedCPT index built locally.
6. CUDA PyTorch works locally on NVIDIA GeForce RTX 5070 Laptop GPU.
7. Qwen2.5-0.5B and Qwen2.5-1.5B local pilots completed.
8. Evidence-checking agent implemented and tested.
9. P1 presentation deck completed.
10. Formal P1 report draft created.
11. Full 360-question Qwen2.5-1.5B generation completed for LLM-only, report-RAG BM25, case-RAG BM25 top-1, and case-RAG hybrid top-1.
12. Full 360-question top-1 evidence checking completed for report-RAG BM25, case-RAG BM25 top-1, and case-RAG hybrid top-1.
13. Manual annotation sample CSV created for 50 full-run outputs.
14. Comparative manual annotation CSV created for 50 questions across four systems.

## Research Claim to Defend

The project is not a generic RAG coursework implementation. The defensible thesis claim is:

Case-level medical RAG should verify answer claims against the selected radiology case, because top-k retrieved evidence can hide cross-case contamination when an LLM combines findings from several similar cases.

## P1 Completion Checklist

Minimum ready-to-submit P1 package:

1. Appointment/title form confirmed by student and supervisor.
2. Research intent form/title aligned with the refined title.
3. Formal proposal report reviewed:
   - `docs/P1_FORMAL_REPORT.md`
4. Editable P1 slides ready:
   - `deliverables/p1-medical-rag-proposal.pptx`
5. Supervisor brief ready:
   - `docs/SUPERVISOR_BRIEF.md`
6. Speaking script and likely Q&A ready:
   - `docs/P1_SPEAKING_SCRIPT.md`

## Final Experiment Plan

### Experiment Set A: Retrieval

Already completed, but should be kept as final retrieval baseline unless the QA seed changes:

1. BM25 on clean QA seed.
2. MedCPT on clean QA seed.
3. Hybrid BM25 + MedCPT, alpha in {0.25, 0.50, 0.75}.

Final table should report Hit@1, Hit@5, Hit@20, and MRR.

### Experiment Set B: Full Generation

Run over all 360 clean QA questions if compute time allows:

1. LLM-only Qwen2.5-1.5B or stronger. Completed for Qwen2.5-1.5B.
2. Report-RAG BM25. Completed for Qwen2.5-1.5B.
3. Case-RAG BM25 top-1 prompt. Completed for Qwen2.5-1.5B.
4. Case-RAG Hybrid top-1 prompt. Completed for Qwen2.5-1.5B.

If a 7B model is feasible through quantization or cloud compute, rerun the same four systems with Qwen2.5-7B-Instruct. If not, the thesis can use Qwen2.5-1.5B full-run results and clearly state local-compute limitations.

### Experiment Set C: Evidence Checking

Run top-1 evidence checking over generated outputs:

1. Report-RAG BM25 generated drafts. Completed for Qwen2.5-1.5B full 360.
2. Case-RAG BM25 top-1 generated drafts. Completed for Qwen2.5-1.5B full 360.
3. Case-RAG Hybrid top-1 generated drafts. Completed for Qwen2.5-1.5B full 360.

Report draft Token-F1, final Token-F1, evidence support, revision rate, abstention rate, and unsupported sentence rate.

### Experiment Set D: Manual Annotation

Manually annotate 30 to 50 outputs. Use a stratified sample:

1. 10 cases where retrieval top-1 is correct.
2. 10 cases where the true case appears in top-k but not top-1.
3. 10 cases where retrieval misses the true case.
4. Optional extra cases showing cross-case contamination.

The current 50-item annotation sheet is:

```text
experiments/manual_annotation_hybrid_qwen15_full360_agentic_top1_sample50.csv
```

The paired comparative annotation sheet is:

```text
experiments/manual_annotation_qwen15_full360_comparative_sample50.csv
```

Manual labels:

1. Answer relevance: 0/1/2.
2. Evidence support: supported, partially supported, unsupported.
3. Hallucination: none, minor, major.
4. Completeness: incomplete, adequate, over-complete/mixed.
5. Case contamination: yes/no.

## Final Report Structure

1. Introduction.
2. Research problem and objectives.
3. Literature review.
4. Methodology.
5. Implementation.
6. Experimental results.
7. Error analysis.
8. Discussion.
9. Limitations and ethics.
10. Conclusion.

## Risk Control

| Risk | Control |
|---|---|
| Project looks too simple | Emphasize case-boundary evidence checking, hybrid retrieval, generated-answer failure analysis, and manual evaluation. |
| Dense retrieval underperforms | Present it honestly; the important result is that hybrid fusion improves retrieval. |
| Small LLM quality is weak | Frame local small-model runs as feasibility/pipeline validation; use stronger model if compute allows. |
| Evidence checker lowers Token-F1 | Explain safety tradeoff: removing unsupported claims can reduce lexical overlap while improving faithfulness. |
| Image component is questioned | State clearly that images are linked case context, not raw-image diagnostic evidence. |

## Next Execution Step

The next technical step is to complete manual annotation and error analysis:

1. Fill the 50-question comparative annotation sheet.
2. Compare automatic evidence-checking labels with human judgments.
3. Identify 5 to 8 representative success and failure cases.
4. Decide whether the final thesis needs a stronger 7B/cloud model or whether the local 1.5B results are sufficient with limitations clearly stated.
