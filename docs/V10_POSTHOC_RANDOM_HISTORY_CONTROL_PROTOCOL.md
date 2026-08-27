# V10 Post-hoc Random-History Control Protocol

## Status and purpose

This protocol defines a post-hoc negative-control analysis of the already
frozen V10 Test cases. It does not modify the V10/V11 technical freeze or any
existing answer. It adds a new exploratory control condition to distinguish
the effect of relevant retrieved history from the generic effect of placing
arbitrary radiology-report text in the generator context.

The analysis asks:

> Does R5-selected historical evidence produce answers that are more
> consistent with the hidden target report than an equal amount of randomly
> selected Train history?

This analysis is not prospectively confirmatory because V10 outcomes were
already known when this protocol was written.

## Frozen inputs

- V10 confirmation config: `config/v10_confirmation.json`
- V10 cluster-disjoint split:
  `data/splits/v10/v10_cluster_disjoint_split.json`
- Frozen V10 retrieval rows:
  `experiments/v10_publication/v10_confirmation_retrieval_rows.jsonl`
- Frozen V10 QA rows:
  `experiments/v10_publication/v10_confirmation_qa_rows.jsonl`
- OpenI case artifact and local RadGraph artifact matching the hashes in the
  frozen confirmation config
- Target cases: the same 568 technically eligible V10 Test cases
- Question types: Findings and Impression
- Generator: `google/medgemma-1.5-4b-it`, frozen local revision
- Answer budget: 64 new tokens
- Evidence policy: three whole reports, matching G2's selected V10 policy

All source artifacts must pass the same hash and coverage checks used by the
V10 confirmation scripts. Existing V10 rows must remain byte-identical.

## Random-history construction

The candidate frame is the technically eligible V10 Train historical bank:

- report text is non-empty;
- RadGraph preprocessing status is `ok`;
- case belongs to the Train partition;
- case is not one of the R4 or R5 Top-10 cases for the target query.

Train and Test are duplicate-cluster disjoint under the V10 split. Patient-level
independence cannot be asserted because reliable patient identifiers are not
available in the processed OpenI source.

Five deterministic assignments are generated. For assignment index `a` in
`0..4`, candidates are ordered by:

```text
SHA256(
  "v10-random-history|7131|" +
  str(a) + "|" +
  canonical_case_id + "|" +
  question_type + "|" +
  candidate_case_id
)
```

The first three eligible candidates are used. Selection is without replacement
within an assignment. No assignment is selected or discarded after generation;
all five receive equal weight in the final random-control estimate.

## Controlled generation condition

For every target case, question type, and assignment:

- use the same target image and indication as G2;
- use the same question;
- use three randomly assigned whole reports;
- use the same prompt constructor, answer parser, sentence normalization,
  deterministic provenance assembly, decoding, token budget, batch size, and
  model revision as the frozen V10 QA run;
- retain raw and normalized answers and all integrity fields;
- never use the hidden target report or reference answer in the prompt,
  selection rule, generation, repair, or retry decision.

Only documented technical reruns under identical settings are permitted.
Malformed or ceiling-truncated outputs are retained and not replaced.

## Evaluation

The predefined systems are:

- G0: frozen target-image answer without history;
- G2: frozen R5-selected historical-RAG answer;
- GR: mean over the five random-history assignments.

Primary control comparison:

```text
G2 - GR
```

Secondary comparisons:

```text
GR - G0
G2 - G0
```

Metrics are:

- Token-F1;
- F1RadGraph entity, entity-relation, and complete scores;
- F1CheXbert 14-observation micro/macro F1;
- F1CheXbert five-observation micro/macro F1 and exact-set accuracy;
- answer token-ceiling rate;
- answer-contract validity;
- provenance validity.

For each case and question, the five GR metric values are averaged before the
primary paired comparison. Confidence intervals use 10,000 case-grouped
bootstrap resamples with seed `7132`. Both questions from a case remain grouped.
The five assignments are not treated as independent patients or independent
confirmation cohorts.

## Promotion and interpretation rule

- If G2 exceeds GR with a 95% interval above zero on Token-F1 and at least one
  clinical semantic metric, the result supports an alignment-specific history
  contribution in this post-hoc control.
- If G2 exceeds GR only numerically or only on lexical overlap, report a
  directional result without confirmed semantic superiority.
- If GR matches or exceeds G2, report that arbitrary report context can explain
  part or all of the observed gain; do not hide the negative result.
- GR versus G0 is diagnostic. A positive GR-minus-G0 result would indicate that
  generic radiology text or answer priors contribute to the apparent RAG gain.

No random-control result may be used to retune V10 or select a new Test-time
prompt, threshold, candidate rule, or generator.

## Claim boundary

This is an automated post-hoc negative control. It can test whether relevant
retrieval is more useful than arbitrary Train reports under the frozen
generator, but it does not establish physician-rated clinical utility,
diagnostic accuracy, safety, patient benefit, external validity, or verified
patient-level independence.

