# V14 Architecture Sensitivity Note

## Reason for the additional run

The predeclared V14 comparison reproduced the original V12 default
LambdaMART hyperparameters. After that run completed, a comparator audit
confirmed that the value presented as the strongest V12 development result
(`0.62023` Validation qrel-v2 nDCG@10) came from a pre-existing `deeper`
configuration selected in the V12 extended pilot, not from the default model
(`0.59827`). The two default V14 models were internally fair, but neither was
the strongest existing comparator.

To answer whether concept features add value under the already selected V12
architecture, one explicit post-protocol sensitivity run is allowed:

```text
n_estimators = 300
learning_rate = 0.03
num_leaves = 31
min_child_samples = 40
reg_lambda = 1.0
random_state = 2026
```

The same deeper configuration is applied to `base_17` and `concept_23`.
Candidate generation, Train fit/internal roles, OOF concept predictions,
Calibration gate, Validation evaluation, qrels, metrics, and all other inputs
remain unchanged. The default artifacts are retained and not overwritten.

## Interpretation

This is a transparent architecture sensitivity analysis motivated by a
baseline-identity audit after the default result was observed. It is not part
of the original V14 protocol and cannot be described as prospectively
predeclared. It is nevertheless a fair same-architecture comparison because
the deeper V12 configuration existed before V14 and is applied symmetrically.
All outcomes must be retained, including a failure to exceed the stronger V12
baseline. V10 Test remains prohibited.

