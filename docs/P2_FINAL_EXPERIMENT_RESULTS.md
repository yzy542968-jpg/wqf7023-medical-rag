# Legacy P2 Lexical-Checker Experiment Results

Updated: 2026-08-14

> Status: superseded as the final thesis result source. This file preserves the earlier 360-question lexical-checker and P1/P2 continuity ablations. Use `docs/FINAL_OPTIMIZED_RESEARCH_RESULTS.md` for the case-disjoint grouped split, Medical NLI agent, adaptive retrieval, RadGraph, confidence intervals, and final held-out conclusions. Do not quote the threshold `0.40` results below as the final proposed-system result.

## Scope

This document records the final precomputed experiments for the P2 evidence-checking extension. It separates two experimental settings that must not be merged into one direct leaderboard:

1. Open-retrieval P2 evaluation: 360 OpenI questions, Qwen2.5-1.5B, BM25 and hybrid BM25 + MedCPT retrieval.
2. Case-conditioned P1 ablation: 100 verified questions, Qwen2.5-7B, report/chunk/case units and three prompt types.

The P2 agent was evaluated without rerunning either language model. It operates on existing generated answers, verifies answer sentences against selected report evidence, removes unsupported sentences, and abstains when no sentence remains supported.

## Evidence Checker Revision

The final checker is negation-sensitive and sentence-aligned. It differs from the original lexical checker in four important ways:

1. Negation terms such as `no`, `not`, `without`, and `negative for` are retained.
2. Normality expressions such as `within normal limits` are treated as semantically negative for abnormalities.
3. Claims are matched to the best evidence sentence instead of an unrestricted token union over the whole context.
4. Common radiology word-form variation, such as `opacity` and `opacities`, is normalized.

Eight regression tests cover positive support, negative support, polarity contradiction, normality equivalence, plural normalization, cross-sentence contamination, revision, and abstention.

## Open-Retrieval P2 Results

### Draft Generation

| System | Draft Token-F1 | Top-1 Case Accuracy | Retrieved Hit Rate |
|---|---:|---:|---:|
| Report-RAG BM25 | 0.146 | 0.231 | 0.383 |
| Case-RAG BM25 top-1 | 0.188 | 0.231 | 0.383 |
| Case-RAG Hybrid top-1 | 0.209 | 0.242 | 0.422 |

Hybrid case-level retrieval gives the strongest draft Token-F1 and retrieval hit rate.

### Evidence-Checking Threshold Sensitivity

| System | Threshold | Final Token-F1 | Evidence Support | Abstention | Unsupported Sentence Rate | Negation Conflicts |
|---|---:|---:|---:|---:|---:|---:|
| Report-RAG BM25 | 0.40 | 0.197 | 0.281 | 0.319 | 0.660 | 113 |
| Report-RAG BM25 | 0.50 | 0.194 | 0.242 | 0.375 | 0.695 | 113 |
| Report-RAG BM25 | 0.65 | 0.179 | 0.185 | 0.492 | 0.762 | 113 |
| Case-RAG BM25 top-1 | 0.40 | 0.146 | 0.436 | 0.242 | 0.547 | 31 |
| Case-RAG BM25 top-1 | 0.50 | 0.134 | 0.328 | 0.308 | 0.677 | 31 |
| Case-RAG BM25 top-1 | 0.65 | 0.109 | 0.160 | 0.608 | 0.843 | 31 |
| Case-RAG Hybrid top-1 | 0.40 | 0.171 | 0.469 | 0.228 | 0.527 | 31 |
| Case-RAG Hybrid top-1 | 0.50 | 0.159 | 0.362 | 0.303 | 0.650 | 31 |
| Case-RAG Hybrid top-1 | 0.65 | 0.118 | 0.177 | 0.592 | 0.828 | 31 |

The previous default threshold of 0.65 is too strict. A preliminary threshold of 0.40 gives the strongest utility-safety trade-off, but the final threshold must be confirmed against manually reviewed outputs.

The strongest agent benefit appears in report-RAG. At threshold 0.40, final Token-F1 increases from 0.146 to 0.197 because the checker removes content copied from non-target retrieved cases. This directly supports the cross-case contamination motivation.

For case-level top-1 systems, Token-F1 decreases after checking. This is an expected safety trade-off: the checker removes unsupported or weakly matched explanatory language, which can reduce lexical overlap even when the remaining answer is more conservative.

## P1 Case-Conditioned Agent Ablation

The agent was also applied to the 900 existing P1 RAG outputs:

- 3 retrieval-unit systems: report, chunk, and case-level RAG.
- 3 prompt types: direct, evidence-guided, and structured case-aware.
- 100 verified questions.
- Qwen2.5-7B-Instruct.

LLM-only rows were excluded because no report evidence is available for post-generation verification.

### Results by Retrieval Unit at Threshold 0.40

| System | N | Draft Correct | Final Correct | Errors Corrected | Correct Answers Lost | Mean Support | Revision | Abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Report-RAG | 300 | 287 | 271 | 0 | 19 | 0.940 | 0.063 | 0.047 |
| Chunk-RAG | 300 | 286 | 274 | 2 | 16 | 0.940 | 0.077 | 0.033 |
| Case-level RAG | 300 | 287 | 271 | 0 | 19 | 0.943 | 0.060 | 0.043 |

### Results by Prompt Type at Threshold 0.40

| Prompt | N | Draft Correct | Final Correct | Errors Corrected | Correct Answers Lost | Mean Support | Revision | Abstention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Direct | 300 | 289 | 279 | 0 | 13 | 0.947 | 0.083 | 0.023 |
| Evidence-guided | 300 | 283 | 272 | 2 | 16 | 0.945 | 0.063 | 0.047 |
| Structured case-aware | 300 | 288 | 265 | 0 | 25 | 0.932 | 0.053 | 0.053 |

The case-conditioned experiment already supplies clean target-case evidence, so the agent has little opportunity to improve correctness. It preserves most answers but introduces conservative losses. This result narrows the thesis claim: evidence checking is most useful when retrieval supplies noisy or multi-case evidence, rather than when a verified target case is already provided by construction.

## Prompt Engineering and Agent Contribution

The experiments distinguish prompt effects from agent effects:

1. Direct prompting provides the strongest case-conditioned correctness before checking.
2. Structured case-aware prompting adds explicit traceability and valid case citations.
3. Evidence checking adds a separate post-generation safety layer and detects polarity conflicts that prompt instructions alone do not guarantee against.
4. The agent has the clearest measurable benefit under report-RAG open retrieval, where multiple retrieved cases can be mixed in one generated answer.

The final thesis should therefore present structured case-aware prompting as the traceability mechanism and the evidence-checking agent as the verification, revision, and abstention mechanism.

## Claims Supported by the Current Evidence

The current results support the following claims:

1. Retrieval improves answer quality over LLM-only generation.
2. Hybrid BM25 + MedCPT improves retrieval coverage and draft answer quality.
3. Case-level top-1 prompting reduces exposure to cross-case evidence compared with multi-report prompting.
4. A negation-sensitive evidence checker can identify unsupported and polarity-conflicting claims.
5. Evidence checking improves report-RAG answer Token-F1 in the noisy multi-case setting, but introduces a safety-utility trade-off in already clean case-conditioned settings.

The current results do not support claims of clinical diagnostic performance or state-of-the-art medical QA.

## Remaining Validation Gate

Before these numbers are treated as final thesis results:

1. Manually review a stratified sample of 30 to 50 questions across the four P2 systems.
2. Label answer relevance, evidence support, hallucination control, completeness, and case contamination.
3. Use the labels to select between threshold 0.40 and 0.50.
4. Report the checker precision/recall or agreement against the manual support labels.
5. Resolve or explicitly report the 31 P1 rows marked `needs_manual_review`.

## Reproducibility Files

| Output | Path |
|---|---|
| P2 threshold sweep | `experiments/final_p2/evidence_threshold_sweep.csv` |
| P2 checked rows at threshold 0.50 | `experiments/final_p2/*_checked_t0.50.jsonl` |
| P1/P2 agent ablation sweep | `experiments/final_p2/p1_stage8b_agent/p1_stage8b_agent_threshold_sweep.csv` |
| P1 checked rows at threshold 0.40 | `experiments/final_p2/p1_stage8b_agent/p1_stage8b_agent_checked_t0.40.jsonl` |
