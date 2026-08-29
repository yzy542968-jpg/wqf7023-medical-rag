# V16 Related-Work Positioning Update

## Purpose

This update fixes the final technical positioning after the V12-V16 extension.
The project is related to retrieval-based radiology report generation, but its
inference contract is not identical: it receives a target chest image, an
available indication, and a question; the hidden target report is unavailable;
and retrieved reports belong to other historical cases.

## Closest reproducible work

### CXR-RePaiR

Endo et al. introduced CLIP-based image-to-report retrieval for chest X-ray
report generation. The method is a foundational retrieval baseline and
supports the use of historical reports as output evidence. It does not provide
the present question-conditioned case-to-fact QA workflow or the same ownership
and provenance contract.

- Paper: <https://proceedings.mlr.press/v158/endo21a.html>

### X-REM

Jeong et al. added multimodal image-text matching to improve retrieval-based
report generation and included expert error assessment. It motivates learned
cross-modal reranking and shows why cosine similarity alone may miss fine-
grained compatibility. Its primary output remains a report, not an arbitrary
question-conditioned answer with deterministic historical-case provenance.

- Paper: <https://proceedings.mlr.press/v227/jeong24a.html>
- Code: <https://github.com/rajpurkarlab/X-REM>

### FactMM-RAG

Sun et al. used RadGraph-derived factual report pairs to train a multimodal
retriever and augment radiology report generation. This is the closest factual
retrieval precedent for the present RadGraph-aware ranking and case-to-fact
evidence stage. The reusable principle is fact-aware supervision; the project
does not copy FactMM-RAG's MIMIC data, LLaVA training, or full-report generation
task.

- NAACL 2025 paper: <https://aclanthology.org/2025.naacl-long.28/>
- Code: <https://github.com/cxcscmu/FactMM-RAG>

### MedProbCLIP

MedProbCLIP models radiograph and report embeddings probabilistically and
emphasizes calibration, risk-coverage behavior, selective retrieval, multiview
encoding, and robustness. It motivates the project's retrieval-confidence and
no-reliable-history analysis. The current project does not claim to reproduce
its probabilistic objective or MIMIC-CXR evaluation.

- Preprint: <https://arxiv.org/abs/2602.16019>

## Final differentiation

The thesis should not claim that multimodal radiology RAG itself is new. Its
defensible methodological contribution is the integrated study design:

1. a target-report-hidden, new-case question-answering contract;
2. candidate generation from BM25, MedCPT, and MedSigLIP followed by a learned
   question-conditioned reranker;
3. duplicate-cluster-disjoint allocation and Train-only historical evidence;
4. case retrieval before fact selection, preserving case and section ownership;
5. aligned-versus-shuffled image and retrieved-versus-random history controls;
6. section-aware MedGemma adaptation evaluated without Test-driven retuning;
7. explicit separation of automated report-reference metrics from clinical
   correctness.

## Reuse boundary

Public projects are used as methodological references, not as sources of
restricted data or unverified outputs. No MIMIC-derived report or image is
redistributed. External MIMIC-CXR replication and independent radiologist review
remain Future Work. The final manuscript must cite primary papers and official
repositories rather than relying on leaderboard values or secondary summaries.
