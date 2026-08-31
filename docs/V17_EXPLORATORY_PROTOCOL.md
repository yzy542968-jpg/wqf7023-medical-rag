# V17 Post-Confirmation Exploratory Protocol

## 1. Study identity

V17 is a post-confirmation exploratory study of relevance-specific historical
evidence in multimodal medical question answering. It is not a preregistration,
does not reopen the frozen Final-QA confirmation, and must not change any V10,
V11, or Final-QA result.

The protocol is committed before V17 retrieval outcomes are generated. The
researcher has already seen prior B3/B4/B6 aggregate outcomes; consequently,
V17 is not result-blind.

## 2. Research question

For a target case whose formal report is hidden from the system, does selecting
question-relevant facts from image-similar historical image-report cases produce
more useful QA evidence than applying the identical evidence pipeline to random
or mismatched historical cases?

The study separates two necessary links:

1. Retrieval mechanism: related evidence must have higher report-derived answer
   agreement than matched controls.
2. Downstream transfer: if link 1 passes, related evidence should improve QA
   relative to matched controls without unacceptable subgroup harm.

Failure of either link is a valid negative result. V17 is not permitted to claim
that related history helps merely because its point estimate is numerically
higher.

## 3. Data roles and leakage boundary

| Final-QA split | V17 role | Permitted use |
|---|---|---|
| Train | Historical bank | Candidate cases, reports, images, structured answers |
| Calibration | Development | Feature-recipe selection, diagnostics, pilot generation |
| Validation | Frozen internal evaluation | One evaluation after a V17 decision record |
| Test | Sealed | No V17 access, tuning, evaluation, or reporting |

All roles are case-ID separated under the existing Final-QA manifests. Existing
duplicate-cluster exclusions remain active. Case-ID and cluster separation will
be asserted by code. Patient-level independence cannot be verified.

Because Validation has supported earlier project development, any later V17
Validation result is described as internal evaluation, not independent
confirmation.

## 4. Fixed retrieval task

For each Calibration question:

1. Retrieve a fixed Top-100 Train-case shortlist by frozen MedSigLIP image
   similarity.
2. Build inference-time-only question features for every shortlisted case.
3. Min-max normalize each feature within that question's Top-100 shortlist.
4. Apply each prespecified static recipe.
5. Select Top-3 cases for evidence extraction.

The actual target report and its structured answers are available only to the
evaluator. They must never be used as input features.

### 4.1 Feature set

- `image_similarity`: frozen target/candidate MedSigLIP cosine similarity.
- `question_fact_max`: maximum TF-IDF similarity between the actual question
  and candidate RadGraph/report-derived fact units.
- `question_sentence_max`: maximum TF-IDF similarity between the actual
  question and candidate report sentences.
- `indication_sentence_max`: maximum TF-IDF similarity between the target
  indication and candidate report sentences.
- `intent_preference_coverage`: deterministic compatibility between the
  question planner's intent and candidate evidence-unit sections/attributes.

No target gold answer, target report text, generation output, or Test-derived
quantity is an admissible feature.

### 4.2 Prespecified recipes

Weights are ordered as image, question-fact, question-sentence, indication-
sentence, and intent-preference coverage:

| Recipe | Weights |
|---|---|
| `image_only` | `1.00, 0.00, 0.00, 0.00, 0.00` |
| `image_fact_equal` | `0.50, 0.50, 0.00, 0.00, 0.00` |
| `image_question` | `0.40, 0.35, 0.25, 0.00, 0.00` |
| `full_query` | `0.35, 0.30, 0.20, 0.10, 0.05` |

Recipe selection uses Calibration only. The primary criterion is balanced Top-1
same-question answer agreement. A tie within 0.001 is resolved in favor of the
earlier, simpler recipe in the table. Top-3 agreement and qrel-v2 are mandatory
secondary diagnostics but do not override the deterministic selection rule.

## 5. Report-derived relevance proxy

For a target question ID, the evaluator compares the target structured answer
vector with the candidate report's vector for the same question ID. Metrics are:

- Top-1 exact answer-vector agreement.
- Top-3 any exact answer-vector agreement.
- Option-label micro-F1.
- The mean of separately computed negative, positive, and non-binary agreement
  strata when all three are estimable (`balanced_qid_agreement`).
- Candidate qrel-v2 as a secondary case-level relevance signal.

Questions for which a valid same-question candidate vector cannot be constructed
remain visible in denominators/coverage reporting and are not silently replaced.
The proxy measures report-derived answer agreement, not clinical similarity or
diagnostic correctness.

## 6. Matched control construction

Three history arms use the same Top-K, fact selector, evidence budget, prompt,
generator, and evaluator:

- `related`: Top-3 cases from the selected V17 recipe.
- `random`: three deterministic Train cases selected by SHA-256 ordering for the
  target question, excluding the target/duplicate cluster.
- `mismatched`: Top-3 selected results from another Calibration query assigned by
  a deterministic fixed-point-free permutation, filtered against the target
  cluster.

All arms pass through the same actual-question-conditioned fact selector. It is
not permissible to compare selected facts for related history with whole reports
for a control arm.

## 7. Evidence extraction

The frozen initial policy is:

- maximum 3 historical cases;
- maximum 2 evidence units per case;
- maximum 6 units total;
- maximum 1,200 evidence characters;
- actual question is the primary evidence query;
- indication is secondary context;
- every unit retains case ID, source section, and unit ID.

Coverage, selected-unit count, evidence length, and provenance validity are
reported for each arm. A relevance gate is not tuned before retrieval-only
evidence shows a useful signal.

## 8. Retrieval-only Go/No-Go rule

Generation may begin only if Calibration retrieval satisfies all conditions:

1. The selected non-baseline recipe has higher balanced Top-1 question-answer
   agreement than `image_only`.
2. Related Top-3 evidence agreement exceeds both random and mismatched controls.
3. Positive-answer agreement does not decrease relative to `image_only`.
4. No implementation or leakage audit fails.

A practical difference of +0.5 percentage points is reported as an engineering
reference, not a confirmatory significance threshold. If conditions fail, V17
stops as a retrieval negative result; QLoRA and generation are not used to hide
the failed mechanism.

## 9. Conditional generation pilot

If retrieval passes, a deterministic case-stratified Calibration subset is
created before generation. The intended size is approximately 2,000-3,000
questions while retaining complete cases and representation of negative,
positive, and non-binary question strata.

The frozen generator is evaluated in four conditions:

- target image + question + indication, no history;
- identical inputs + related fact evidence;
- identical inputs + random fact evidence;
- identical inputs + mismatched fact evidence.

Primary generation metric: Exact Accuracy over all pilot questions.

Mandatory metrics:

- positive, negative, and non-binary Exact Accuracy;
- balanced stratum accuracy;
- option-label micro-F1;
- coarse question-family macro accuracy;
- related-minus-random and related-minus-mismatched paired differences;
- negative-transfer rate relative to no history;
- evidence coverage and abstention/empty-evidence rate;
- case-grouped bootstrap confidence intervals.

The pilot is successful only if related evidence exceeds both matched controls on
balanced accuracy and does not achieve the result solely by improving frequent
negative answers. Overall Exact Accuracy remains reported even when it is not the
most discriminating metric.

## 10. Validation and stopping

After the Calibration retrieval and conditional generation work is complete, a
`V17_DEVELOPMENT_DECISION_RECORD.md` must freeze:

- selected ranking recipe;
- evidence policy;
- generator revision, quantization, prompt, decoding, and parser;
- pilot sampling rule;
- all metrics, bootstrap settings, and failure handling.

Only then may the chosen method be evaluated once on Final-QA Validation. No
Validation-specific retuning is allowed. Final-QA Test remains sealed regardless
of the Validation result.

## 11. Interpretation boundaries

Permitted claims are limited to report-derived proxy relevance, automatic
question-answer agreement, internal QA metrics, provenance, and computational
cost. V17 cannot establish clinical safety, diagnostic accuracy, patient-level
generalization, physician usefulness, or external validity.

Clinical/blinded human assessment and external patient-level validation remain
Future Work. Negative or mixed V17 results must be retained and must not be
converted into a positive claim by changing the metric family after inspection.

