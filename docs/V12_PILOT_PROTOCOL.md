# V12 Optimization Pilot Protocol

## Status and scope

This document defines a post-freeze, development-only optimization pilot for
the multimodal historical-case RAG system. It does not amend the V10/V11
technical freeze, replace the V10 primary study, or authorize any evaluation
on the V10 Test partition for model selection.

The purpose is to test whether the current development bottlenecks can be
reduced before deciding whether a separately designed study is worthwhile.
All pilot outputs are exploratory until a new confirmation protocol and a new
confirmation cohort are available.

## Frozen boundaries

- Source data: the existing OpenI case artifact and V10 cluster-disjoint split.
- Development fitting: V10 Train only.
- Development selection and diagnostics: V10 Validation only, with existing
  Validation reuse explicitly treated as post-freeze development analysis.
- V10 Calibration may be used only for calibration diagnostics already defined
  by the existing pipeline.
- V10 Test is excluded from all fitting, threshold selection, prompt selection,
  model selection, pilot stopping decisions, and case inspection.
- The target report/reference remains hidden from runtime retrieval and answer
  generation. It is used only for development metrics and diagnostics.
- Historical candidate banks must exclude the target query case.
- No human clinical correctness or safety claim is permitted.

## Pilot stages

### Stage 0: artifact and bottleneck diagnostics

Before fitting a new model, recompute or verify from saved development rows:

1. candidate relevant-case presence and recall for K=50, 100, and 200;
2. best available qrel and oracle nDCG@10 within each candidate pool;
3. current R5 nDCG@10 and its oracle-rerank gap;
4. normal, abnormal, and indeterminate subgroup summaries;
5. generation token-ceiling, raw-versus-final answer, and output-length
   sensitivity diagnostics.

Existing aggregate artifacts may be verified instead of regenerated, but the
report must label each value as `recomputed` or `verified_from_artifact`.

### Stage 1: retrieval pilot

The first new experiment compares only the following predeclared systems:

1. existing R5 candidate source and reranker;
2. existing multi-source RRF candidate generation at K=100 and K=200;
3. the same feature pipeline with one LambdaMART-compatible ranker if a
   ranking implementation is available locally;
4. multi-source candidate generation plus that ranker.

The first implementation may use the existing BM25, MedCPT and MedSigLIP
artifacts. Deterministic multi-query fusion may be added as a separate
configuration only if its query templates are frozen before evaluation.

No Qwen3 cross-encoder, visual pathology classifier, or QLoRA is included in
the first retrieval pilot. These are later extensions, not hidden tuning
degrees of freedom.

### Stage 2: generation pilot

Stage 2 is run only after Stage 1 is complete and its results are recorded.
It evaluates, on development data only:

- 96- and 128-token generation budgets;
- question-specific Findings and Impression prompts;
- section-aware historical evidence routing;
- an answer selector based only on the question and retrieved evidence;
- deterministic provenance assembly and output validation.

The selector must not use the reference answer, target report, or target
labels. It must not silently replace a malformed answer with reference-derived
text. Raw and selected outputs are both retained.

### Stage 3: optional model extensions

Only if the preceding stages show a reproducible signal may the following be
considered as separately named experiments:

- Qwen3 cross-encoder reranking over a fixed Top-50 or Top-100 shortlist;
- assertion-aware hard-negative features;
- a frozen external visual concept channel;
- MedGemma QLoRA trained only on case-disjoint development data.

Each extension requires an updated pilot record before execution. It cannot be
combined with several unrecorded changes and reported as one improvement.

## Relevance and leakage safeguards

The existing qrel is a structured report-derived proxy, not clinical gold.
Features derived from RadGraph, report labels, polarity, anatomy, or other
report structure may overlap with this proxy. Any such feature must be marked
as a potential construct-overlap feature and accompanied by label-only and
fact-only sensitivity results where applicable.

Target-report-derived information must never be used as an inference feature.
Assertion-aware hard negatives may be built from training-side report data,
but the construction rule must be fixed before Validation scoring.

## Statistics and stopping rules

The primary pilot retrieval metric is case-grouped nDCG@10. Secondary metrics
include relevant-case presence, relevant recall, MRR, subgroup performance,
and candidate-outside-pool rate. Generation metrics include Token-F1,
F1RadGraph where available, token-ceiling rate, provenance validity, and
answer length.

For the best new configuration selected using the predeclared development
rule, report paired case-level bootstrap differences against R5 with 95%
confidence intervals. A point estimate alone is not evidence of superiority.

The following are engineering continuation heuristics, not confirmatory
statistical thresholds:

- less than +0.005 nDCG@10: normally stop retrieval optimization;
- +0.005 to +0.015: retain as exploratory and require further justification;
- above +0.015 with a confidence interval supporting the direction: consider
  a new confirmation study.

Generation promotion requires improvement in both Token-F1 and at least one
independent quality or integrity measure, with no loss of provenance validity.

## Output and reporting policy

New outputs belong under `experiments/v12_optimization/`. The pilot must
retain failed configurations, environment details, artifact hashes, seeds,
and exact commands. No V10/V11 result file may be overwritten.

The pilot conclusion must be one of:

- `STOP_NO_MEANINGFUL_SIGNAL`
- `CONTINUE_RETRIEVAL_DEVELOPMENT`
- `CONTINUE_GENERATION_DEVELOPMENT`
- `DESIGN_NEW_CONFIRMATION_STUDY`

Even a strong pilot result remains a development result. The thesis primary
claims remain V10/V11 unless a later confirmation protocol is independently
designed, frozen, executed, and audited.

