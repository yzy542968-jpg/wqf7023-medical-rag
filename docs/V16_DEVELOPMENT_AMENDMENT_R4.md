# V16 Development Amendment R4: Exploratory Question-Type Routing

**Status:** completed exploratory development amendment
**Scope:** Validation only; no Test evaluation
**Date:** 2026-08-29

## Trigger

The full 376-case Validation comparison showed an asymmetric effect of the
balanced QLoRA adapter. Relative to the frozen base generator, the adapter
improved the retrieved-history condition overall, but the question-type
breakdown showed lower findings Token-F1 and higher impression Token-F1. A
findings-only QLoRA adapter was therefore trained under the already documented
R3 exploratory extension, using the same 300-case development source,
question-type filter, qv target profile, learning rate, and 256-step budget.

## Findings-only screening decision

On the 12-case screening set, the findings-only adapter produced findings
Token-F1 of `0.283623` in the retrieved-history condition, compared with
`0.332819` for the frozen base. The adapter was not promoted and was not
expanded to the 376-case Validation set.

This screening result is development evidence only. It does not establish a
general conclusion about section-specialized fine-tuning.

## Exploratory routes

To test whether the observed asymmetry could be handled by a transparent
question-type route, a deterministic row-level route was assembled from
already generated outputs:

```text
findings   -> frozen base generator
impression -> balanced qv QLoRA adapter
```

The route is selected only by the known question type. It does not inspect the
reference answer, metric value, or target-case outcome. The route was applied
to the same 12-case screening matrix and the complete 376-case Validation
matrix, including no-history, retrieved-history, and random-history controls.

No new model was trained for the route and no V10, V11, or V15 artifact was
modified.

A second, deployment-oriented gate was also assembled from the same rows:

```text
retrieved_history + impression -> balanced qv QLoRA adapter
all other conditions/types   -> frozen base generator
```

This gate leaves the two controls unchanged and has the same retrieved-history
result as the question-type route. It is recorded as an engineering candidate,
not as a new confirmatory experiment.

## Interpretation boundary

This route is a post-hoc exploratory Validation candidate. It must not be
described as a prespecified confirmatory system or as evidence of clinical
diagnostic accuracy. All reported scores are automated report-reference
consistency measures. The route does not provide physician-validated
correctness, clinical safety, or external-validation evidence.

The route should be retained as a candidate for later engineering comparison,
but it cannot replace the frozen V16 base-versus-QLoRA comparison in the
primary thesis evidence without a newly frozen protocol and a fresh untouched
confirmation cohort.
