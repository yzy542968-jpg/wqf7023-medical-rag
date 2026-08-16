# Project Roadmap

Current date used for planning: 2026-06-05.

> Archived planning record. As of 2026-08-14, the implementation, grouped development/test evaluation, final local generation, Medical NLI agent, RadGraph analysis, contamination analysis, and Dashboard are complete. Current work is independent blinded human review followed by thesis and defense artifact production. See `docs/FINAL_OPTIMIZED_RESEARCH_RESULTS.md`.

## P1 Proposal Phase

### Week 1: 2026-05-17 to 2026-05-24

- Confirm supervisor/title approval email.
- Collect 10-20 core references for RAG, medical QA, biomedical retrieval, radiology report datasets, and RAG evaluation.
- Download or confirm access to IU X-Ray / OpenI data. Done for metadata.
- Build normalized case-level JSONL from reports and image mappings. Done: 3,851 cases and 7,466 image mappings.
- Run a small lexical retrieval prototype. Done: TF-IDF and BM25 baselines.
- Download a small real image subset for P1 feasibility demonstration. Done: 5 pneumonia cases / 10 X-ray images.
- Write preliminary feasibility results for proposal. Done.

### Week 2: 2026-05-25 to 2026-06-01

- Draft P1 report Chapter 1: Introduction.
- Draft Chapter 2: Research Problem.
- Draft Chapter 3: Research Questions and Objectives.
- Draft Chapter 4: Literature Review.
- Draft Chapter 5: Research Design and Methodology.
- Design evaluation set format and annotation rules.
- Prepare dependency plan for dense retrieval: MedCPT embeddings, ChromaDB, and model storage.

### Week 3: 2026-06-02 to 2026-06-08

- Produce P1 report submission version. Draft completed in `docs/P1_FORMAL_REPORT.md`.
- Build proposal presentation slides. Completed in `deliverables/p1-medical-rag-proposal.pptx`.
- Prepare 10-minute speaking script and Q&A notes. Completed in `docs/P1_SPEAKING_SCRIPT.md`.
- Run a feasibility demo with 5-10 test questions using real OpenI cases. Completed as local pilot generation and evidence-checking runs.

## P1 Assessment Window

Proposal Presentation: 2026-06-08 to 2026-07-17.

During this window:

- Revise report and slides after supervisor feedback.
- Strengthen methodology and evaluation design.
- Freeze the implementation plan for P2.

## P2 Implementation Phase

### Late July to August 2026

- Implement LLM-only baseline. Pilot completed; full 360-question run pending.
- Implement report-based RAG baseline. Pilot completed; full 360-question run pending.
- Implement case-based RAG with paired image-report retrieval. Pilot completed; full 360-question run pending.
- Implement MedCPT dense retrieval and hybrid fusion. Completed with NumPy index.
- Add prompt variants. Completed for direct, evidence-guided, structured, and top-1 structured prompts.
- Generate system outputs for the full evaluation set. Pending.

### September 2026

- Run final evaluation.
- Analyze retrieval quality, answer support, and hallucination tendency.
- Write final report.
- Prepare final defense slides and demo examples.
