# Final-QA Real-Output Source-Gate Protocol

## Status

This is a post-hoc development protocol frozen before the new out-of-fold
source-gate outcomes are computed. The complete Final-QA Validation B3/B4/B6/P1
results and the v2 oracle-payload gate results are already known. Validation is
therefore development data. Five-fold out-of-fold evaluation prevents the same
case from fitting and evaluating a fold-specific gate, but it does not recreate
an untouched confirmation set.

Final-QA Test remains uninstantiated and inaccessible.

## Motivation

The actual report-text experiment established a trade-off:

- B6 Top-1 paired history improved ordinary exact QA over B3 image-only;
- B6 reduced supported-label macro-F1;
- random history was competitive, so generic context effects remained
  plausible;
- the oracle-payload study showed that correct report ownership nevertheless
  contains answer-relevant signal.

A global history policy is therefore too coarse. This experiment asks whether
the source should be selected by question type and observable disagreement,
using the already generated, deployable B3 and B6 answers.

## Inputs available to the gate

The gate may use only information available at inference:

1. the structured question identity and answer type;
2. the B3 and B6 predicted option sets;
3. their option overlap and answer counts;
4. the frozen Top-1 target-image to historical-image cosine similarity;
5. the number of available answer options.

Gold indices and target report content are labels for development metrics only.
They may not enter a feature. B4 random-history outputs are an evaluation
control and may not enter the gate.

## Five-fold case-level OOF design

Each of the 358 Validation cases is assigned to one of five folds by:

```text
SHA-256("final-qa-real-output-gate|7053|" + case_id) modulo 5
```

For each fold, policy statistics and model parameters are fitted on the other
four folds and applied once to the held-out fold. All questions from a case
remain together.

## Candidate policies

### Question-ID exact-utility policy

For every structured question ID, compare B3 and B6 exact answer-set accuracy
on the four training folds. Use B6 for that question ID only when its accuracy
is higher; ties retain B3.

### Question-ID macro-utility policy

For every question ID, compare the supported option-label macro-F1 of B3 and
B6 on the training folds. Use B6 only when its macro-F1 is higher; ties retain
B3.

### Logistic disagreement gate

Common B3/B6 answers are retained. On disagreements, regularized logistic
regression estimates whether B6 is exactly correct while B3 is not. Numeric
features are standardized and question ID/answer type are one-hot encoded.
The decision threshold is selected inside each training fold from the fixed
grid in the JSON protocol. It maximizes training-fold macro-F1 subject to
question exact and option micro-F1 remaining within `0.001` of B3. Exact ties
prefer a higher threshold and therefore less historical reliance.

## Metrics and advancement

The primary development metric is five-fold OOF supported-label macro-F1.
Question exact answer-set accuracy, option micro-F1, complete report-vector
accuracy, history use, negative transfer and history-only recovery are reported
jointly.

The selected OOF policy advances only if it:

1. exceeds B3 macro-F1;
2. keeps exact accuracy within `-0.001` of B3;
3. keeps option micro-F1 within `-0.001` of B3;
4. exceeds B4 random-history macro-F1;
5. uses B6 for at least one genuine disagreement.

A GO outcome still does not unlock Test automatically. It justifies fitting the
selected gate on all development cases and writing a separate confirmation
protocol before deciding whether a long B3/B6 Test generation run is worth the
cost.

## Stopping rule

No new MedGemma generation, QLoRA update, retriever tuning or Test access is
allowed in this stage. A failed OOF gate closes this branch. Frozen historical
studies and their results remain unchanged.
