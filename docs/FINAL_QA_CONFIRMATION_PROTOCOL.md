# Final-QA Confirmation Protocol

## Status

This protocol is frozen before the final gate is fitted on all development
cases and before Test case identities are instantiated in a confirmation
manifest. The policy family, minimum support, minimum utility margin, generator,
retriever, adapter, conditions, endpoints and decision rules are fixed here.

This is a repository protocol with a Git timestamp, not a formal external
preregistration.

## Research question

Can a conservative, question-conditional policy use correctly paired
historical image-report evidence to improve ordinary structured chest-X-ray QA
while retaining rare-label performance relative to image-only generation?

## Frozen final policy

The final policy is fitted once on all 358 Final-QA Validation development
cases. For each Rad-ReStruct question ID, B6 replaces B3 only when:

```text
development support >= 5 question rows

and

B6 option-label macro-F1 - B3 option-label macro-F1 >= 0.05
```

Otherwise B3 is used. These values are based on the stable median nested-OOF
selection and are fixed before the Test manifest is generated.

## Confirmation frame

The withheld role contains 530 mapped cases and 26,747 questions. Test cases
are cluster-disjoint from Train, Calibration and Validation under the V10
exact/near-duplicate report clustering. Patient-level independence beyond the
source-design identity cannot be independently verified.

The Test manifest is generated only after this protocol commit. The manifest
must retain case and cluster fingerprints and prove zero overlap with all
development roles. No failed or inconvenient case may be silently replaced.

## Conditions

Three complete Test outputs are generated with the same frozen 384-step QLoRA
adapter and MedGemma revision:

1. **B3:** target image, indication, question and options; no history;
2. **B4:** B3 plus one deterministic random other-cluster Train report;
3. **B6:** B3 plus the whole report of the Top-1 eligible MedSigLIP historical
   image neighbour.

The nested final policy is assembled deterministically from B3 and B6 outputs;
it does not require a fourth generation arm. B4 remains a mandatory
non-relevant-context control and is never selectable.

## Primary endpoints

### H1: ordinary QA superiority

The final policy exceeds B3 in question-level exact answer-set accuracy. The
criterion is a paired case-grouped 95% bootstrap CI with lower bound above zero.

### H2: rare-label non-inferiority

The final policy is non-inferior to B3 in supported-label macro-F1 with margin
`-0.005`. The paired case-grouped 95% bootstrap CI lower bound must be at least
`-0.005`.

Both H1 and H2 are required for the combined positive claim. If only one
passes, the result is mixed.

## Secondary endpoints

- option micro-F1 and exact report-vector accuracy;
- final-policy macro-F1 versus B4 random history;
- B6 versus B3 exact, option micro-F1 and macro-F1;
- contract and provenance validity;
- history use, history-only recovery and image-correct-to-history-wrong counts;
- runtime, input/output tokens and peak GPU allocation.

The B4 comparison assesses relevance specificity but is secondary; no endpoint
is redefined after Test outcomes are available.

## Runtime and failure policy

The frozen generation batch size is eight with greedy decoding and 32 maximum
new tokens. Outputs are appended by unique run key, so a technical interruption
may resume only missing keys under the same configuration. OOM, process or
transient file failures permit a technical rerun without parameter changes.

A genuine case-data failure is retained as a protocol deviation and does not
trigger silent replacement. Failure of H1 or H2 is reported as a negative or
mixed confirmation and does not permit Test-driven retuning.

## Evidence boundary

This is same-source, cluster-disjoint confirmation using report-derived
structured answers. It is not external validation, clinical diagnosis accuracy,
physician adjudication, safety validation or proof of patient benefit. Human
evaluation and external patient-level validation remain Future Work.
