# Final-QA v2 Selective Paired-History Gate Protocol

## Status and prior knowledge

This development protocol is locally frozen before the selective-gate outcomes
are computed. It is not a formal preregistration. The Final-QA v1 Validation
results and the v2 oracle-payload feasibility result are already known. In
particular, correctly paired history improved supported-label macro-F1, fixed
50/50 fusion reduced ordinary question accuracy, and image-only and historical
answers showed complementary errors. The entire 358-case Validation role is
therefore development data from this point onward. Final-QA Test remains
uninstantiated and inaccessible.

## Objective

Determine whether features available at inference can identify questions for
which a correctly paired historical case should replace an image-only answer,
without the common-answer and over-reliance damage caused by unconditional
fusion.

This is a bounded feasibility question. A positive result permits development
of a deployable report-to-fact historical payload. It does not permit a Test run
and does not establish clinical correctness.

## Data boundaries

- Historical bank: mapped Final-QA Train cases only.
- Development frame: the previously inspected Final-QA Validation cases.
- The deterministic v2 feasibility split remains unchanged: 170 selection
  cases and 188 development-holdout cases.
- The 170 selection cases are divided again at case level by SHA-256 into gate
  training and gate calibration partitions.
- No case, question, threshold, model, feature, or outcome from Final-QA Test is
  available during this experiment.
- Duplicate-cluster exclusions inherited from the Final-QA manifest remain in
  force.

## Historical paired unit

Each candidate is indivisible:

```text
historical image
+ that image's own report embedding
+ that report's structured answer payload
+ case and pair provenance
```

The structured answer payload is an oracle development substitute for facts
that must later be extracted from report text. It may not be presented as a
deployable input.

## Stage A: multimodal cooperative candidate selection

The image-only Top-20 historical shortlist is reused to bound cost. Each
target-question and historical candidate pair receives the following features:

1. target-image to historical-image cosine similarity;
2. target-image to historical-report cosine similarity;
3. question to historical-report cosine similarity;
4. historical-image to its own-report cosine compatibility;
5. reciprocal image rank;
6. margin from the highest image similarity.

Questions are encoded once with the pinned MedSigLIP text tower on CUDA. The
candidate selector compares the existing image Top-1 baseline with a logistic
MCR-lite model. Candidate supervision is whether the candidate report's answer
to the same structured question equals the target reference answer. The target
reference is used only as a development label, never as a feature.

## Stage B: question-level source gate

For each question, the image-only answer and selected historical answer are
compared.

- If they agree, the common answer is retained.
- If they disagree, the gate selects image-only or paired history.
- When the estimated historical advantage is insufficient, image-only is the
  deterministic default.

Candidate reliability, retrieval margins, Top-3 answer agreement, answer-set
overlap, question ID and answer type are gate features. The target answer is
used only to construct development labels.

Three gate families are compared:

1. one-dimensional image-similarity threshold;
2. regularized logistic regression;
3. a two-layer MLP trained with early stopping.

Model and threshold selection use only gate calibration. A difference smaller
than `0.0005` in exact answer-set accuracy selects the simpler model. The MLP is
eligible only if it provides a material calibration improvement.

## Metrics and advancement

The primary metric is question-level exact answer-set accuracy. Key secondary
metrics are option micro-F1 and supported-label macro-F1. The experiment also
reports source disagreement, history-use rate, image-correct-to-history-wrong
over-reliance, history-only recovery, and case-grouped confidence intervals.

The selected method advances only if, on the fixed 188-case development
holdout:

1. exact answer-set accuracy exceeds image-only;
2. option micro-F1 difference is at least `-0.001`;
3. supported-label macro-F1 difference is at least `-0.005`;
4. at least one disagreement is resolved in favour of history;
5. the method exceeds its fixed-point-free shuffled-pair counterpart.

Passing this gate does not unlock Test. It unlocks only the next development
stage: replacing oracle historical answers with deterministic report-derived
facts and repeating the development controls.

## Negative and pairing controls

The study retains image-only, history-only, fixed fusion, deterministic random
history and 20 fixed-point-free shuffled report-payload assignments. Shuffling
preserves the target images, historical image set and retrieval scores while
breaking only report ownership.

## Stopping rule

No new MedGemma generation is allowed in this stage. If the selected gate fails
the advancement rule, the selective paired-history extension stops or is
redesigned using Train/development data only. Frozen V10/V11/V12/V16 and
Final-QA v1 artifacts remain unchanged.
