# V17 Technology Reuse Audit

## Status and scope

V17 is a post-confirmation exploratory mechanism study. It does not replace,
reopen, or modify the frozen V10/V11 studies or the Final-QA confirmation
study. In particular, the Final-QA Test split and its outcomes must not be used
for V17 feature design, recipe selection, threshold selection, prompt tuning,
or stopping decisions.

The purpose of V17 is narrower than "improve the overall score":

> Test whether actual-question-conditioned selection of fact-level evidence
> from image-similar historical cases produces a relevance-specific QA benefit
> over identically processed random or mismatched historical evidence.

This audit records what can be reused, what is genuinely new, and which tempting
approaches are excluded before V17 implementation begins.

## Frozen evidence that V17 must not alter

- Final-QA confirmation used 530 Test cases and 26,747 questions.
- The selective final policy passed its prespecified Exact Accuracy improvement
  and macro-F1 non-inferiority gates.
- Related whole-report history (B6) did not exceed random whole-report history
  (B4) on Test Exact Accuracy. This observed result is descriptive input to the
  research motivation, not a target against which V17 may tune.
- Existing V10/V11 and Final-QA files, metrics, splits, prompts, outputs, and
  release claims remain historical frozen artifacts.

## Reusable repository components

### Data and split infrastructure

- The Final-QA Train split supplies the historical retrieval bank.
- The Final-QA Calibration split supplies V17 development cases.
- The Final-QA Validation split is reserved for one frozen V17 internal
  evaluation after development decisions are recorded.
- Duplicate-cluster exclusions and case-ID separation already implemented by
  the Final-QA study remain in force.
- Patient-level independence cannot be claimed because reliable patient
  identifiers are unavailable in the processed OpenI source.

### Image retrieval

- Existing MedSigLIP case embeddings and image-similarity retrieval can provide
  a fixed Top-100 shortlist.
- The image encoder remains frozen. V17 does not fine-tune MedSigLIP.
- Image-only retrieval is the V17 retrieval baseline; it is not silently
  replaced by a stronger text query before comparison.

### Question planning and evidence selection

- `v11_question_planner.py` provides deterministic intent labels such as
  presence, location, severity, uncertainty, and comparison.
- `v10_evidence.py` and `v11_evidence.py` provide sentence/fact units,
  question-conditioned ranking, compact evidence selection, and provenance.
- `v10_reranker.py` provides reusable TF-IDF sentence/fact feature logic.
- Existing RadGraph-derived fact artifacts may be reused as report-derived
  evidence units. They are not clinical annotations or physician gold labels.

### Generator and evaluator

- MedGemma remains frozen for the initial V17 pilot.
- Existing Rad-ReStruct answer parsing, exact accuracy, option-label metrics,
  case-grouped bootstrap, and question-family summaries may be reused.
- Existing compact deterministic provenance assembly should be preferred over
  asking the generator to reproduce long metadata structures.

## What is genuinely new in V17

V12 candidate generation is not equivalent to V17. Its query representation
uses indication plus a generic report question type (findings, impression, or
summary). V17 uses the actual Rad-ReStruct question text and its deterministic
intent when reranking image-retrieved candidate cases and selecting evidence.

The new contribution is therefore the controlled chain:

```text
target image + indication + actual question
    -> fixed image Top-100 shortlist
    -> question-conditioned candidate reranking
    -> identical fact selector for every history arm
    -> relevance/coverage diagnostics
    -> frozen MedGemma QA
```

The matched random and mismatched controls are equally important. A gain over
no history alone would not establish that historical-case relevance caused the
gain; extra medical text could act as a prompt prior. V17 must compare related,
random, and mismatched evidence through the same processing pipeline.

## Approaches deliberately deferred

- Final-QA Test reuse or Test-driven threshold selection.
- QLoRA or foundation-model fine-tuning before retrieval relevance is shown.
- An unrestricted learned reranker with many post-hoc features.
- Clinical-accuracy, patient-level-independence, or physician-validation claims.
- Human or blinded clinical evaluation; this remains Future Work.
- External-data validation; OpenI-only V17 is within-source evidence.
- LangChain migration. The repository's deterministic pipeline offers better
  experimental control and traceability for this study.
- An autonomous agent. V17 is a controlled multimodal RAG pipeline, not an
  agentic decision-maker.

## Risk assessment

### Primary risk: report-derived relevance proxy

Candidate relevance is inferred from structured report answers for the same
question and from report-derived facts. It is useful for controlled development
but is not a clinical-similarity gold standard. Results must be called
"report-derived answer agreement" or "proxy relevance," not clinical accuracy.

### Class imbalance

Overall Exact Accuracy can be dominated by frequent negative answers. V17 must
report positive, negative, and non-binary strata and a prespecified balanced
summary alongside overall accuracy.

### Shortlist ceiling

Question-conditioned reranking cannot recover a useful case outside the fixed
image Top-100. The target/relevant-candidate-outside-shortlist rate must be
reported rather than hidden by dropping such queries.

### Reuse of Validation

Final-QA Validation has been inspected in prior development. A frozen V17
evaluation on it is an internal out-of-development comparison, not an untouched
or independent confirmation. Only a new dataset or never-used split could
provide that stronger claim.

## Reuse decision

Reuse the mature data, embedding, evidence, provenance, generation, and
evaluation infrastructure. Add only the smallest new layer required for actual-
question-conditioned candidate ranking, report-derived proxy evaluation, and
matched controls. Proceed to generation only if the retrieval-only stage shows
that selected related evidence is measurably more relevant than both random and
mismatched evidence on Calibration.

