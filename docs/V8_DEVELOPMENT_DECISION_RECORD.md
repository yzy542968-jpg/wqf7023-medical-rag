# V8 Development Decision Record

## 1. Decision

V8 candidate-level multimodal reranking is recorded as a **development no-go**.
The candidate-level scorer did not exceed the validation-selected global fusion
comparator on the frozen V8 development matrix. Therefore:

- the global fusion baseline is retained as the V8 development comparator;
- no V8 confirmation case IDs were generated;
- no V8 confirmation retrieval, shuffled-image control, or QA transfer was run;
- V6 and V7 frozen artifacts, results, and claims remain unchanged.

This is a development decision, not a confirmation outcome. It is not evidence
that candidate-level reranking cannot work on another dataset or under another
pre-specified study design.

## 2. Reproducibility state

The run used the V8 protocol committed at `b2ab5e0`, the tracked V7
development-only retrieval matrix, and the existing V7 development manifest.
The V8 confirmation frame was audited but its final case IDs were not
instantiated.

| Artifact | Value |
|---|---|
| V8 protocol commit | `b2ab5e0` |
| Development retrieval rows | `experiments/post_submission_v7/development_retrieval_rows.jsonl` |
| Development rows SHA-256 | `c37729122f4a562727a6663da8d81d83452ea748c36f95836b72698313612f19` |
| V7 development manifest SHA-256 | `e4b2025b5ee6770be30878faedb2d040fbdad130f6ea4a9c8250e2eba739bd3d` |
| V8 configuration SHA-256 | `b7562d45228d5c830c147d99fe0f485c10cf2d5b2a3f0281fdb2547a2466a135` |
| Confirmation IDs instantiated | `false` |

The local summary and diagnostic checkpoint are intentionally excluded from
the public repository under the repository's local-artifact policy. Their
paths and hashes are recorded in the local
`experiments/post_submission_v8/v8_development_summary.json` output.

## 3. Development matrix

The development data were case-disjoint Train A, Train B, and Validation blocks
from the V7 development manifest. The V8 runner used:

- 720 development cases in total;
- 240 cases in Train A plus Train B for model fitting;
- 120 validation target cases / 360 question rows;
- a deterministic 42-case internal early-stopping holdout;
- 4,648 internal pairwise training pairs;
- 5,640 final training pairs;
- 15 question rows whose target was outside the BM25 Top-100 shortlist.

Target-outside-shortlist rows were excluded from pairwise gradient training
because they have no valid in-shortlist positive candidate, but they were
retained in evaluation as retrieval failures. This policy was fixed in the
protocol and was not changed after observing the validation result.

## 4. Comparator and candidate models

The comparator was selected on the same V8 development validation matrix using
the pre-specified alpha grid `{0.00, 0.01, ..., 1.00}`. The selected global
fusion weight was:

```text
S_global = 0.52 * text_score + 0.48 * image_score
Validation case-grouped MRR = 0.6407744656
```

The candidate-level feature schema contained the seven retrieval features and
three question-type indicators specified in the V8 protocol. Linear and MLP
candidate scorers were trained with the frozen pairwise logistic objective.

The best candidate by the protocol's development selection rule was an MLP
with learning rate `0.001`, zero weight decay, and 15 final epochs:

```text
Candidate validation case-grouped MRR = 0.5411455603
Global validation case-grouped MRR     = 0.6407744656
Candidate minus comparator             = -0.0996289053
```

Because the candidate did not first exceed the comparator point estimate, it
failed the development gate. The stronger confirmation requirement, namely a
strictly positive lower bound of the 95% confirmation bootstrap interval, was
therefore never reached.

## 5. Diagnostic interpretation

The result is consistent with the candidate scorer being too unconstrained for
the available development sample. The global fusion score already provides a
strong, calibrated within-shortlist ordering. Replacing it with an unconstrained
candidate score discards that ordering and overfits the pairwise examples.

Additional read-only diagnostics were performed using only development rows:

- baseline-anchored residual scorers were more stable than the unconstrained
  scorer but did not produce a validated positive gain;
- listwise residual training did not produce a validated positive gain;
- tree-based candidate classifiers and a baseline-failure-only gate did not
  produce a development improvement after selecting the correction strength on
  the internal development holdout;
- no V8 confirmation case, confirmation metric, shuffled-image outcome, or V7
  outcome was used in these diagnostics.

These diagnostics support retaining the no-go decision. They do not justify
adding an unplanned model, threshold, or feature after the fact.

## 6. Claim boundary

The valid conclusion is:

> Under the frozen V8 development protocol and the available case-disjoint
> development matrix, the tested candidate-level pairwise reranker did not
> improve on the validation-selected global text/image fusion baseline.

The following claims are not made:

- candidate-level reranking is impossible in general;
- the V8 confirmation cohort was tested;
- the learned model is clinically correct or clinically safe;
- patient-level independence was established;
- V8 provides external validation;
- V6 or V7 findings were invalidated.

The V7 result remains the latest completed adaptive-fusion experiment, with
correctly aligned image features showing a statistically significant alignment
signal while the adaptive query-level fusion hypothesis itself did not pass.
That negative result and this V8 no-go are both retained as part of the study's
honest model-development record.

## 7. Stopping rule

No V8 confirmation cohort is to be generated unless a new, separately approved
protocol or protocol addendum defines a new development design before any
confirmation IDs are instantiated. Such a future study would need to justify
its additional data, model class, loss, and selection rule in advance.

For the current V8 protocol, technical work stops at this development decision.
