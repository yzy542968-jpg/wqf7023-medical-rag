# V16 Generation Adaptation Confirmation Protocol

## Status and purpose

This protocol freezes the final V16 confirmation design before generating or
inspecting any V12/V16 outputs on the V10 Test partition. V10 and V11 remain
unchanged. V12-V16 development used V10 Train and Validation only.

The purpose is to test whether the Validation-selected V12 retrieval and V16
generation-adaptation stack transfers to the existing cluster-disjoint Test
partition. The study measures automated consistency with hidden OpenI report
sections. It does not measure physician-adjudicated diagnosis, clinical safety,
treatment utility, or external generalization.

The V10 Test partition has previously been evaluated and inspected for the
frozen V10 systems. It is therefore not a globally untouched project holdout.
However, no V12 LambdaMART ranking, V16 adapter, V16 routed policy, or V16
generation output has been evaluated on Test before this protocol. Results are
described as a held-out V16 method evaluation, not a formal preregistration or
a project-wide unseen test.

## Frozen research question

> Does a Validation-selected, section-aware MedGemma adaptation improve
> report-reference consistency when supplied with Train-bank historical cases
> selected by the frozen V12 RRF Top-200 plus LambdaMART retriever?

## Frozen data boundary

- Source: the existing processed OpenI/IU-Xray case artifact.
- Historical bank: technically eligible V10 Train cases only.
- Query frame: all technically eligible V10 Test cases.
- Split: the existing V10 duplicate-cluster-disjoint split.
- Eligibility: a case must be present in the formal OpenI adapter, have a
  successful frozen RadGraph record, have readable target image input, and
  provide non-empty Findings and Impression references.
- No case may be replaced after Test generation begins.
- A failed case remains a documented technical failure under the frozen
  denominator; it is not replaced by the next case.
- Case-ID and duplicate-cluster disjointness are verified. Identifier-based
  patient independence cannot be verified from the processed OpenI release.

The expected executable Test frame is 568 cases, matching the frozen V10 QA
frame after six previously documented unusable RadGraph records. The manifest
builder must verify the count rather than force it. A different count stops the
run and requires a documented data-integrity investigation before any model
generation.

## Frozen retrieval

The retrieval condition is the V12 development winner:

```text
BM25 + MedCPT + MedSigLIP candidate sources
    -> deterministic reciprocal-rank-fusion Top-200
    -> frozen 17-feature LambdaMART reranker
    -> Top-3 historical reports
```

The LambdaMART model was fit on V10 Train roles and selected during V12
development without Test outcomes. Its Test application uses the same Train
historical bank, feature order, RRF construction, model file, and Top-200/Top-3
budgets as development. R5 full-bank retrieval is retained in the Test ranking
artifact as the frozen historical comparator.

## Frozen generation arms

All arms use `google/medgemma-1.5-4b-it` at revision
`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`, one target image, the source
indication, a fixed question, at most 96 new tokens, at most two complete
sentences, greedy deterministic decoding, and deterministic provenance checks.

The generated matrix contains:

1. frozen base generator with no history;
2. frozen base generator with V12 retrieved history;
3. frozen base generator with deterministic random history;
4. balanced 300-case QLoRA generator with no history;
5. balanced 300-case QLoRA generator with V12 retrieved history;
6. balanced 300-case QLoRA generator with deterministic random history.

Two routed policies are derived deterministically from those saved rows without
additional model inference:

### Section route

```text
Findings   -> frozen base generator
Impression -> balanced QLoRA generator
```

### Retrieved-history impression gate

```text
Retrieved-history Impression -> balanced QLoRA generator
All other rows                -> frozen base generator
```

The retrieved-history impression gate is the primary V16 candidate. It was
chosen after the Validation asymmetry was observed and is transparently a
Validation-selected policy. It cannot be changed after Test execution.

## Controls

- `no_history` tests target-image generation without historical reports.
- `random_history` uses a domain-separated SHA-256 ordering over eligible
  Train-bank cases, excluding the target duplicate cluster.
- `retrieved_history` uses the frozen V12 Top-3.
- All three conditions retain the same target image, indication, question,
  generator configuration, and reference.

Random-history Test ordering uses the domain
`v16-confirmation-random-history`, seed `1617`, canonical case IDs, UTF-8, and
lowercase SHA-256 hexadecimal ordering.

## Outcomes and statistics

### Primary outcome

Case-averaged Token-F1 difference on retrieved-history rows:

```text
retrieved-history impression gate - frozen base
```

The paired 95% confidence interval uses 10,000 case-grouped bootstrap samples
with seed `1626`. Findings and Impression rows from the same case remain in the
same resample.

### Secondary outcomes

- balanced QLoRA minus base under each history condition;
- Findings and Impression Token-F1;
- BLEU-1, BLEU-4, ROUGE-L, METEOR, CIDEr, and BERTScore;
- CheXbert micro/macro F1 and exact-five agreement;
- RadGraph entity, entity-relation, and complete F1;
- output-token ceiling rate;
- answer-contract and evidence-provenance validity;
- latency, input/output length, and peak allocated GPU memory;
- report-indexed normal, abnormal, and indeterminate sensitivity summaries.

NLG, CheXbert, RadGraph, subgroup, and runtime outcomes are secondary. They may
not be used to choose another model or modify the route after Test evaluation.

## Decision rule

V16 supports a positive generation-adaptation conclusion only if:

1. the primary Token-F1 confidence interval is entirely above zero;
2. answer-contract and provenance validity remain 100%;
3. neither RadGraph complete F1 nor CheXbert micro-F1 shows a statistically
   supported regression, defined as its paired 95% interval lying entirely
   below zero; and
4. retrieved history remains better than random history for the primary routed
   policy in point estimate.

A positive RadGraph relation or complete-F1 interval is supporting evidence,
not a prerequisite. If the primary interval crosses zero or is negative, V16
is reported as unconfirmed or negative and V10 remains the final thesis primary
study. No Test-driven retraining, prompt revision, threshold change, routing
change, case deletion, or metric substitution is permitted.

## Standard-metric comparability boundary

BLEU, ROUGE, METEOR, CIDEr, BERTScore, CheXbert, and RadGraph are added to make
the final outputs comparable with public report-generation literature. Cross-
paper values remain contextual because IU-Xray has no single official split and
published systems differ in image views, report sections, prompts, training
data, and metric implementations. Token-F1 remains the internal primary metric
because it governed V16 development before Test.

## Frozen artifact identities

| Artifact | SHA-256 |
|---|---|
| `data/processed/openi_cases.jsonl` | `56e367190396011d4d67f43e7e733389a8346890bf8729e82fb4326d063bbd68` |
| `data/processed/v9_radgraph_modern_xl.jsonl` | `631aa3e11cc52005656ee8a66de3de1ee5d3411a2f271a3c5f8a14de39b51599` |
| `data/splits/v10/v10_cluster_disjoint_split.json` | `b4c1b091c3dbff0399d07c8350f4d8d68ce8ce52e0157dcc96f46af8c8baa7b3` |
| `data/processed/v10_medsiglip_embeddings.npz` | `f81f4629a8f6eb10dc6b35d868f719384adb40903e19d165f9fba2039fce8867` |
| `data/processed/openi_medcpt_full.npz` | `eb7cbf3b98dc2f5c7f9810abfc45f33c692697a0e511dcaaf32daa03cbaf4177` |
| `experiments/v12_optimization/retrieval/v12_qwen3_lambdamart.txt` | `8c83d6188daa66939ae6a7865c14eada827c4cf625cc0314beaa4988ec2f086c` |
| V16 balanced QLoRA `adapter_config.json` | `bfe3ca8233e7d1433644525d4aa31998bba0814a99bd452cb90ef56af60f2b22` |

The R4 and five R5 checkpoint hashes remain those recorded by the V10 freeze.
The Test query cache, Test ranking rows, Test manifest, generated rows, metric
caches, and summaries do not exist at protocol freeze and will receive hashes
only after deterministic execution.

## Claim boundary

Permitted language is limited to automated report-reference consistency on a
same-source, duplicate-cluster-disjoint OpenI Test frame. The confirmation does
not establish physician agreement, diagnostic accuracy, safety, treatment
benefit, patient-level independence verified by identifiers, external validity,
or readiness for clinical deployment. Independent blinded review and authorized
MIMIC-CXR validation remain Future Work.
