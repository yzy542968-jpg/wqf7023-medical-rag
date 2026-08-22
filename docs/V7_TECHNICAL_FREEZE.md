# V7 Technical Freeze

## 1. Freeze status

The V7 adaptive multimodal fusion extension is technically complete and
frozen. The primary retrieval confirmation, shuffled-image alignment control,
secondary MedGemma QA transfer, and frozen semantic verification have been
executed under the committed V7 protocol and confirmation cohort.

V7 is a same-source, case-ID-disjoint, closed-set paired-report study. It is
not external validation, patient-level validation, image diagnosis, clinical
utility evaluation, or physician-adjudicated correctness evaluation.

## 2. Frozen lineage

```text
V7_DEVELOPMENT_PROTOCOL.md       2ec6dce
V7_DEVELOPMENT_DECISION_RECORD   8409d9d
V7_CONFIRMATION_PROTOCOL.md      4821f38
V7 confirmation cohort           25a39d8
V7 retrieval result record       ff629f4
V7 QA runners                    f7b8c37, 825950d, 4f3e7fd
V7 QA result record              255f040
```

The instantiated confirmation cohort contains 240 cases, 120 target cases,
120 distractor cases, and 360 report-derived questions. The cohort fingerprint
is:

```text
7ed42bfc4851350c767f631d744d0306ee9ac5a406a3b74ceb75a568ceb89c65
```

Foundation models remained frozen. The only learned component was the V7
linear query-conditional fusion model trained during development.

## 3. Final retrieval outcome

```text
BM25 text-only MRR                 = 0.590420
Global alpha*=0.52 MRR             = 0.613370
Adaptive alpha_q MRR               = 0.601897
Adaptive - global                  = -0.011473
95% case-grouped bootstrap CI      = [-0.026769, +0.003109]
```

H1, defined as adaptive superiority over the validation-selected global
fusion weight, **did not pass**. The adaptive model is therefore not promoted
to a confirmed positive methodological contribution.

The aligned adaptive system versus 100 deterministic shuffled-image controls
was:

```text
Aligned adaptive MRR               = 0.601897
Shuffled mean MRR                  = 0.595331
Shuffles >= aligned                = 1 / 100
Plus-one Monte Carlo p             = 0.019802
```

H2 **passed**. This supports the narrower conclusion that correctly aligned
visual information affected the frozen retrieval behavior. It does not imply
that adaptive fusion was better than global fusion.

## 4. Secondary QA transfer

The frozen MedGemma 1.5 generator was applied to the Top-1 report selected by
BM25, global fusion, and adaptive fusion. Verified Token-F1 was:

```text
BM25                              = 0.537279
Global alpha*=0.52                = 0.547277
Adaptive alpha_q                  = 0.546671
```

The QA transfer is descriptive and mixed: both multimodal conditions exceed
BM25 numerically, while global fusion is slightly higher than adaptive fusion.
It cannot change H1/H2 or select a different retriever.

## 5. Frozen claim

The technically supported V7 claim is:

> On a same-source, case-ID-disjoint, closed-set paired-report benchmark,
> correctly aligned image information remained useful under the frozen
> multimodal retrieval pipeline, but the learned query-conditional fusion
> policy did not establish superiority over a validation-selected global
> fusion weight.

The following claims are explicitly out of scope:

- clinical correctness or safety;
- diagnosis from an uploaded image;
- patient-level independence;
- external validity or external validation;
- deployment utility;
- physician-validated human performance;
- population-level qualitative error rates.

## 6. Post-freeze rules

After this freeze, no new model, prompt, threshold, retrieval feature,
quantization policy, dataset, confirmation case, or confirmatory experiment
may be added to V7. The local raw rows, verified rows, and summaries remain
available for audit under repository policy, but are not committed because
they contain large model outputs and source-derived content.

The remaining work is manuscript integration, final document/PDF audit,
presentation revision, and dashboard wording alignment. Those are reporting
tasks and must not alter the frozen V7 implementation or outcomes.
