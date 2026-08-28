# V15 Stronger-Retrieval-to-QA Transfer Protocol

## Purpose

V12 established that the Train-internal-selected deeper LambdaMART improved
Validation retrieval over the original default LambdaMART. V15 asks whether
that retrieval improvement transfers to answer-reference consistency when the
generator and every other input are held fixed.

This is a Validation-only development comparison. It does not alter V10/V11,
does not load V10 Test, and cannot establish clinical correctness.

## Fixed paired design

The cohort is the existing frozen V12 48-case generation manifest:

- 24 report-indexed normal cases;
- 24 report-indexed abnormal cases;
- selected by the pre-existing SHA-256 rule before V12 generation;
- no case replacement.

Each case has Findings, Impression, and acute questions. Both conditions use:

- the same target image and indication;
- the same question planner and answer-only prompt template;
- whole-report evidence from exactly three historical cases;
- MedGemma 1.5 4B revision
  `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`;
- greedy frozen generation behavior and 96 new-token limit;
- identical answer cleaning, reference construction, and provenance checks.

Only the historical Top-3 changes:

1. `default_17`: existing V12 default LambdaMART rankings and already saved
   generation rows;
2. `deeper_17`: the pre-existing V12 deeper LambdaMART rankings with model
   SHA-256 `8c83d6188daa66939ae6a7865c14eada827c4cf625cc0314beaa4988ec2f086c`.

The existing default rows may be reused only after verifying their case IDs,
question matrix, policy, token budget, model revision, selection hash, and
ranking-row hash. The deeper condition is generated once. Existing default
outputs cannot be regenerated selectively after observing deeper results.

## Metrics

The primary analysis includes Findings and Impression only because both have
direct source-report section references. The acute question uses Impression
or Findings as a constructed proxy and is secondary.

Report for primary and all-row scopes:

- Token-F1;
- F1RadGraph entity, entity-relation, and complete;
- F1CheXbert micro F1 across 14 and five observations;
- exact five-observation set agreement;
- answer-contract validity, provenance validity, token ceilings, input tokens,
  output tokens, latency, and peak GPU memory;
- paired case-grouped bootstrap intervals for `deeper_17 - default_17`.

Subgroup findings by question type and report-indexed spectrum are descriptive
and cannot override the primary result.

## Decision rule

The deeper retrieval route demonstrates positive downstream transfer only if:

1. the primary Token-F1 or complete F1RadGraph 95% paired case-grouped interval
   has a lower bound above zero;
2. the other of those two metrics does not have an interval wholly below zero;
3. F1CheXbert micro F1-5 does not have an interval wholly below zero;
4. answer-contract and provenance validity do not decline.

If point estimates improve but intervals cross zero, report numerical but
unconfirmed transfer. If deeper retrieval worsens generation, retain the
negative result and do not tune retrieval, prompts, evidence policy, token
budget, or case selection on these outputs.

## Claim boundary

Token-F1, F1RadGraph, and F1CheXbert measure automated consistency with the
hidden same-source report reference. They are not diagnostic accuracy,
radiologist agreement, clinical utility, safety, or patient benefit. OpenI
patient-level independence remains unverifiable. Independent physician review
and external MIMIC-CXR evaluation remain Future Work.

