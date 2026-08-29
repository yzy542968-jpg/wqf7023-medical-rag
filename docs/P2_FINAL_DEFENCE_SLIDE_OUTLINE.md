# Final Defence Slide Outline

This outline is the maintained defence deliverable. It specifies the content of each slide but does not generate a PPTX. All values below come from frozen repository artifacts.

## Slide 1 - Title

**Title:** Retrieval-Augmented Medical Question Answering over Paired Radiology Images and Reports

**Subtitle:** An auditable new-case historical-evidence workflow with learned retrieval and section-aware generation

**Footer:** Zhang Yue | 22097191 | Master of Artificial Intelligence | University of Malaya

**Visual:** One chest radiograph on the left and a simple evidence chain on the right: target image -> historical cases -> concise answer + provenance.

**Speaker focus:** The system does not retrieve the target patient's own report. It uses a target chest image, indication, and question to retrieve other-case historical evidence.

## Slide 2 - Clinical and Technical Problem

**Headline:** A fluent answer can still use the wrong evidence.

**On-slide points:**

- New target case: chest image + indication + question; final report hidden at inference.
- Historical archive: other-case paired images and reports.
- Risks: text shortcut, wrong-case retrieval, irrelevant history, unsupported generation, lost provenance.

**Visual:** Two contrasting paths: correct other-case evidence versus wrong-case evidence that still sounds plausible.

**Speaker focus:** Report-level faithfulness is insufficient if the retrieved report belongs to the wrong case.

## Slide 3 - Research Aim and Questions

**Aim:** Develop and rigorously evaluate an auditable multimodal RAG workflow for other-case historical evidence retrieval and question answering.

**Research questions:**

1. Does correct image alignment add retrieval information beyond indication and question text?
2. Does multi-source candidate generation plus learned reranking improve historical-case retrieval?
3. Does retrieved history improve downstream answer-reference consistency over no history and random history?
4. Does section-aware QLoRA improve generation without weakening the output/provenance contract?

**Visual:** Four numbered research questions mapped to four result panels.

## Slide 4 - Data and Leakage Control

**On-slide numbers:**

- 3,851 OpenI/IU-Xray cases.
- 3,013 exact/near-duplicate report clusters.
- Train 2,510 | Calibration 383 | Validation 384 | Test 574.
- 568 technically executable Test cases; 2,506-case Train historical bank.

**Controls:** target report excluded from the bank; case-ID and duplicate-cluster disjointness verified; Test outcomes prohibited during V12/V16 development.

**Boundary:** reliable patient identifiers were unavailable, so patient-level independence is not claimed.

**Visual:** Cluster first, then partition diagram.

## Slide 5 - Final System Architecture

**Pipeline:**

1. Input target image, indication, and question.
2. Generate BM25, MedCPT, and MedSigLIP candidate rankings.
3. Reciprocal-rank fusion and deterministic Top-200 union.
4. Frozen 17-feature LambdaMART reranking.
5. Select Top-3 other-case reports and question-relevant facts with case/section provenance.
6. Use frozen MedGemma base or QLoRA adapter according to the prespecified section route.
7. Attach deterministic provenance and evaluate against the hidden target report.

**Visual:** Full-width architecture diagram. Keep the target report visibly outside the inference path.

## Slide 6 - Models and Trainable Contributions

**Frozen foundation models:** MedSigLIP, MedCPT, MedGemma 1.5.

**Trainable components:**

- LambdaMART: learns question-conditioned ranking over the Top-200 candidate frame.
- QLoRA adapter: parameter-efficient generation adaptation; selected route applies it only to retrieved-history Impression questions.

**Why this matters:** The thesis includes genuine model training while separating representation reuse, ranking adaptation, and generation adaptation.

**Visual:** Table with model, role, frozen/trainable status, and data split used.

## Slide 7 - Evaluation Design and Controls

**Retrieval primary metric:** case-averaged nDCG@10 under report-derived graded qrels.

**Generation primary metric:** case-averaged Token-F1.

**Secondary metrics:** BLEU-1/4, ROUGE-L, METEOR, CIDEr, BERTScore, RadGraph, CheXbert, contract/provenance validity, token-ceiling rate.

**Negative controls:** 100 fixed-point-free shuffled images; RRF order alone; full-bank LambdaMART; no history; deterministic random history.

**Statistics:** 10,000 case-grouped bootstrap resamples; 95% confidence intervals.

**Visual:** Metric-to-question matrix.

## Slide 8 - Foundation Result: Correct Image Pairing Matters

**Key result:**

- Correctly aligned V10 R5 nDCG@10: 0.36007.
- Mean across 100 shuffled-image assignments: 0.24963.
- No shuffled run reached the aligned result; plus-one Monte Carlo p=0.00990.

**Conclusion:** The visual signal is alignment-specific, not merely an arbitrary image perturbation.

**Boundary:** This is automated report-derived retrieval relevance, not pixel-level diagnostic accuracy.

**Visual:** Distribution of 100 shuffled results with a vertical aligned-score line.

## Slide 9 - Final Retrieval Result

**Primary comparison:**

- Recomputed V10 R5: nDCG@10 0.55313.
- Final V12 RRF Top-200 + LambdaMART: 0.61590.
- Difference: +0.06277; 95% CI [+0.05460,+0.07082].

**Sensitivity:** label-only +0.03928; fact-only +0.01326; both intervals positive.

**Mechanism controls:** RRF ordering alone and full-bank LambdaMART were worse than R5.

**Conclusion:** Both the complementary candidate frame and the learned reranker are necessary.

**Visual:** Five-bar comparison with the two negative controls in muted grey.

## Slide 10 - Final Generation Result

**Retrieved-history comparison:**

- Base MedGemma Token-F1: 0.20570.
- Final V16 impression-gated route: 0.25591.
- Difference: +0.05020; 95% CI [+0.03973,+0.06108].

**Evidence utility controls:** no history 0.16922; deterministic random history 0.19608; retrieved history 0.25591.

**Conclusion:** Relevant retrieved history contributes more than either the target image alone or extra unrelated report text.

**Visual:** Three-condition bars plus a base-versus-route inset.

## Slide 11 - Multi-Metric Interpretation

**Positive evidence:** BLEU-1, BLEU-4, ROUGE-L, METEOR, CIDEr, BERTScore, and RadGraph all favored the final route.

**Mixed evidence:** CheXbert micro-F1 was inconclusive; reference-positive recall decreased by 0.01081 with a fully negative interval.

**Engineering evidence:** answer-contract validity 100%; provenance validity 100%; token-ceiling rate 87.85% -> 56.60%.

**Interpretation:** Better report-reference consistency and compactness do not imply uniform disease-label recall improvement.

**Visual:** Green/mixed/engineering evidence matrix rather than a single composite score.

## Slide 12 - Protocol Deviation and Research Integrity

**Observed issue:** 81 cases had empty Findings references, yielding 243 zero-reference rows per arm; Impression references were complete.

**Action:** Retain the frozen 568-case denominator, disclose the mismatch, and compute a post-hoc non-empty-reference sensitivity.

**Sensitivity:** paired difference +0.04571; 95% CI [+0.03371,+0.05763].

**Why the gain is not created by empty rows:** Findings outputs are identical between routes; only Impression generation changes.

**Visual:** Protocol expectation -> audit finding -> retained primary -> sensitivity analysis.

## Slide 13 - Contributions and Research Value

**Contributions:**

1. Target-report-hidden new-case historical RAG task.
2. Duplicate-aware, Train-only evidence design and alignment control.
3. Complementary multi-source candidate generation plus trained LambdaMART reranking.
4. Section-aware parameter-efficient generator adaptation.
5. Deterministic case/section provenance and honest mixed-metric reporting.

**Value:** The research demonstrates when historical image-report cases improve automated QA and exposes where retrieval, generation, and clinical-label behavior diverge.

**Visual:** Five contribution icons linked into one evidence chain.

## Slide 14 - Limitations and Future Work

**Limitations:** single OpenI source; report-derived qrels/references; no verified patient identifiers; controlled question roles; residual 56.6% token ceilings; mixed CheXbert recall; Test reused across frozen project stages.

**Future Work:**

- Prespecified blinded radiologist review of 80-120 cases.
- Authorized MIMIC-CXR-JPG external replication with reliable subject/study IDs.
- Reference-eligibility fail-fast checks before cohort instantiation.
- Clinician-authored natural questions and clinically grounded risk-coverage analysis.
- Shorter constrained outputs and explicit clinical-recall safeguards.

**Visual:** Current evidence boundary on the left, future validation ladder on the right.

## Slide 15 - Conclusion and Demonstration Transition

**Four takeaways:**

1. Correct image pairing contributes real retrieval signal.
2. V12 improves a strong R5 comparator under multiple report-derived relevance constructs.
3. Relevant historical evidence improves downstream answer-reference consistency.
4. Section-aware QLoRA improves most metrics and reduces truncation, but clinical-label evidence remains mixed.

**Closing boundary:** A reproducible research prototype for auditable historical-evidence QA, not a diagnostic or clinically validated system.

**Demo transition:** Upload a target chest radiograph, enter indication and question, inspect Top-3 historical cases, then show the answer and provenance. State that the live dashboard demonstrates the V10 workflow while the V12/V16 panel reports frozen offline results.

## Backup Slides

Prepare optional backup slides for model revisions and hardware, qrel construction, subgroup results, standard NLG metrics, RadGraph/CheXbert definitions, runtime, version chronology, and repository reproduction commands. Keep independent clinical evaluation and MIMIC-CXR explicitly labelled Future Work with no scores.
