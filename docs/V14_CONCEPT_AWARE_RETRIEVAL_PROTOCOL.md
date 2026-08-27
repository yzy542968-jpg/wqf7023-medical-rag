# V14 Concept-Aware Retrieval Development Protocol

## Status and purpose

V14 is a development-only experiment. It does not alter the frozen V10/V11
study, the V12 retrieval pilot, or the V13 target-concept and QA results. V10
Test is prohibited throughout V14.

V13 established two distinct findings: a linear head over frozen target-image
MedSigLIP embeddings predicted automated report-derived CheXbert observations
better than a prevalence baseline, while verbalizing thresholded concepts in
the MedGemma prompt reduced answer-reference consistency. V14 therefore tests
the narrower question:

> Can continuous target-image concept probabilities improve historical-case
> reranking when they are used as non-verbal retrieval-state features rather
> than asserted as diagnostic text to the generator?

## Data boundary

- Historical bank and ranker fitting: V10 Train only.
- Ranker early stopping: the frozen V10 internal-early-stop Train role only.
- Feature-route selection: V10 Calibration only.
- One development evaluation after route selection: V10 Validation only.
- V10 Test cannot be loaded, ranked, scored, inspected, or used for a V14
  decision.
- Candidate generation remains the V12 multi-source RRF Top-200 frame.
- Query cases are excluded from their own historical candidate lists.
- Existing duplicate-cluster assignments define every out-of-fold split.

Only case-ID and duplicate-cluster disjointness are claimed. Reliable patient
identifiers are unavailable in the processed OpenI artifact, so patient-level
independence cannot be verified.

## Leakage-controlled concept predictions

Candidate concepts are automated CheXbert binary labels computed from each
historical Train report. These labels are available historical-case metadata;
they are never computed from a hidden Calibration or Validation report for use
as an input.

Target-image concept probabilities use the V13 linear architecture with
`C=1.0` over frozen 1,152-dimensional MedSigLIP image embeddings.

For Train queries, probabilities must be out-of-fold:

1. assign each V10 duplicate cluster to one of five folds by
   `SHA256("v14-oof|7145|" + canonical_cluster_id) mod 5`;
2. fit 14 one-versus-rest linear heads on the other four folds using the V13
   fitting specification;
3. predict probabilities for the held-out fold;
4. concatenate the five held-out predictions in canonical Train case order.

Thus no Train query's report-derived label can supervise its own image
probability vector. Calibration and Validation probabilities use the already
frozen V13 selected checkpoint. Thresholded concept words are not used.

## Predeclared concept-agreement features

For target probabilities `p` and candidate binary report labels `y`, append
exactly six features to the existing 17-dimensional V12/R5 feature vector:

1. `soft_agreement = 1 - mean(abs(p - y))`;
2. `soft_positive_recall = sum(p * y) / max(sum(p), epsilon)`;
3. `soft_candidate_precision = sum(p * y) / max(sum(y), epsilon)`;
4. `soft_cosine = dot(p, y) / max(norm(p) * norm(y), epsilon)`;
5. `no_finding_agreement = p_nf` when the candidate has `No Finding`, else
   `1 - p_nf`;
6. `query_concept_confidence = mean(abs(p - 0.5) * 2)`.

`epsilon=1e-8`. All features must be finite and bounded to `[0, 1]`. The sixth
feature is constant within a query group but may let the ranker condition its
use of candidate-level concept agreement on image-concept confidence.

No hidden target report label, RadGraph target fact, reference answer, case ID,
filename, or QA outcome is permitted as a model feature.

## Ranker comparison

Two LightGBM LambdaMART rankers are fit on identical query groups, RRF Top-200
candidates, qrel-v2 training labels, seed, and hyperparameters:

1. `base_17`: the existing 17 V12/R5 features;
2. `concept_23`: the same 17 features plus the six frozen concept features.

Both use the V12 configuration: ten-level monotone qrel quantization,
`n_estimators=300`, `learning_rate=0.05`, `num_leaves=15`,
`min_child_samples=40`, `reg_lambda=1.0`, `random_state=2026`, and early
stopping patience 25 on the internal-early-stop role. No hyperparameter search
is allowed.

## Metrics and selection

Primary ranking metric is case-grouped mean nDCG@10 under qrel-v2. Report:

- overall, report-indexed normal, abnormal, and indeterminate nDCG@10;
- label-only and fact-only nDCG@10 sensitivity analyses;
- Hit@1, Hit@5, relevant-case presence in RRF Top-200, and relevant-item
  recall at the frozen `qrel-v2 >= 0.5` threshold;
- paired case-grouped bootstrap intervals for `concept_23 - base_17`;
- latency, peak memory when available, and model size.

The concept route is promoted from Calibration to one Validation evaluation
only when all conditions hold:

1. Calibration qrel-v2 nDCG@10 improves by at least `0.005`;
2. Calibration fact-only nDCG@10 does not decrease by more than `0.005`;
3. all input hashes, OOF coverage, finite-feature, and no-Test assertions pass.

If promoted, Validation is evaluated exactly once without changing folds,
features, model settings, candidate depth, qrel, or metric definitions.

A positive Validation development result requires the case-grouped 95%
bootstrap interval for qrel-v2 nDCG@10 to have a lower bound above zero and the
fact-only point difference to remain no worse than `-0.005`. Otherwise V14 is
reported as negative or mixed development evidence and is not promoted to a
future confirmation study.

## Interpretation boundary

The target and candidate concepts are automated report-derived proxies. A
gain may partly reflect compatibility with the report-derived relevance
construct. Label-only and fact-only sensitivity analyses are therefore
mandatory. V14 cannot establish diagnosis, clinical similarity, safety,
patient benefit, physician utility, external generalization, or patient-level
independence. Independent radiologist review and external data remain Future
Work.

## Protocol correction before implementation

The initial protocol commit named `target-outside-RRF-Top-200 rate`. That
metric is invalid for the final similar-case task because the target case is
intentionally absent from the Train historical bank. Before implementation or
V14 outcome inspection, it was replaced by relevant-case presence and
relevant-item recall at the existing `qrel-v2 >= 0.5` development threshold.
This correction changes no cohort, feature, model, or observed result.
