# V10 Publication Extension: Technology and Validity Audit

## 1. Status

This document authorizes a new V10 publication extension while preserving V9
as an immutable completed study. V10 may reuse source-neutral implementation,
frozen model snapshots, and source data, but it may not rewrite a V9 protocol,
split, checkpoint, prompt, output, metric, or result.

V10 is a prospectively specified repository study prepared after V9 outcomes
were known. It is not a blinded or externally timestamped preregistration.

## 2. Publication-level gaps carried forward from V9

V9 established that aligned image information improved report-derived
similar-case retrieval and that retrieved evidence improved lexical answer
agreement over a no-retrieval condition. Four limitations prevent stronger
claims:

1. report near-duplicates crossed the V9 Train/Test boundary;
2. relevance was report-derived rather than physician adjudicated;
3. full retrieved reports introduced question-irrelevant context;
4. 53.70% of generation rows ended with incomplete structured output.

V9 also lacked calibrated retrieval refusal, an external patient-disjoint
confirmation, and independent radiologist review.

## 3. Reusable ideas from related work

| Work | Reusable idea | V10 decision |
|---|---|---|
| FactMM-RAG | RadGraph fact supervision and factual hard negatives | Reimplement fact-level selection and hard-negative sampling locally |
| X-REM | Fine-grained image-report interaction after coarse retrieval | Retain the lightweight V9 reranker and add question/fact interaction features |
| RA-RRG | Key-phrase evidence compression and multi-view aggregation | Select evidence inside each retrieved case while preserving case provenance |
| MedProbCLIP | Uncertainty, calibration, and risk-coverage evaluation | Add a held-out calibration partition and selective retrieval policy |
| CXR-RePaiR | Transparent image-to-report retrieval baseline | Preserve frozen image-image and image-report component baselines |

External repositories are methodological references, not vendored
dependencies. Public visibility is not permission to copy unlicensed source.
The RA-RRG repository contained only a release placeholder at audit time.

## 4. V10 authorized components

V10 may add only the following publication-directed components before a
separate protocol amendment:

1. deterministic report-near-duplicate and exact-image clustering;
2. cluster-disjoint Train, Calibration, Validation, and Test partitions;
3. a case-level retriever followed by within-case sentence/fact selection;
4. question/fact features and fact-aware hard-negative pair sampling;
5. multiple deterministic reranker seeds or a development-selected ensemble;
6. calibration of a no-reliable-history decision and risk-coverage analysis;
7. compact two-stage generation with deterministic provenance assembly;
8. target-view aggregation or prespecified per-view observation fusion;
9. automated clinical metrics as complementary, non-clinical signals;
10. a blinded clinical-review package and analysis program;
11. a patient/study-level MIMIC-CXR adapter and external protocol;
12. reproducibility tests, manifests, hashes, runtime, and cost reporting.

## 5. Components explicitly rejected

- Retuning V9 after its confirmation results.
- Treating report-derived qrels as physician similarity judgments.
- Treating historical reports as proof of a target-patient finding.
- Adding LangChain, LlamaIndex, GraphRAG, or additional autonomous agents
  without a prespecified evaluated function.
- Selecting a larger generator merely because it is newer.
- Using Test to choose clustering thresholds, evidence-selection policy,
  model seed, ensemble, prompt, decoding, calibration threshold, or metric.
- Calling researcher or assistant coding independent clinical adjudication.
- Publishing MIMIC-derived text, pixels, identifiers, or model inputs.

## 6. Evidence hierarchy

The final claim hierarchy is fixed as follows:

1. cluster-disjoint OpenI retrieval confirmation;
2. aligned-versus-shuffled image dependence;
3. selective-retrieval calibration and risk-coverage;
4. end-to-end QA transfer with structurally complete outputs;
5. independently scored clinical review, if actually completed;
6. external patient-disjoint MIMIC-CXR confirmation, if authorized data are
   actually available.

Items 5 and 6 remain pending until real evidence exists. Software readiness
or an empty evaluation package is not a result.

## 7. Audit decision

The existing repository remains the implementation base. V10 should address
validity and evidence compression rather than repeat broad model replacement.
The immediate next action is to freeze `V10_DEVELOPMENT_PROTOCOL.md` and its
machine-readable configuration before generating V10 partition identities or
running V10 outcomes.

