# V16 Development Amendment R3

## Scope

This amendment records an exploratory section-specialization candidate after
the 376-case Validation comparison showed a clear asymmetry between findings
and impression generation. It does not change V10, V11, or V15 frozen
artifacts and does not authorize Test evaluation.

The balanced 300-case QLoRA candidate improved the retrieved-history
condition overall, but its full Validation Token-F1 was higher for impression
questions than for findings questions. The section-specialization pilot tests
whether this is caused by a shared adapter having to model two distinct report
sections rather than by retrieval itself.

## Predeclared pilot

The pilot may train a `qv` QLoRA adapter on the same stable 300-case Train
subset, the same three history conditions, learning rate `5e-5`, and 256
forward steps, while retaining only one question type (`findings`). The
existing balanced adapter remains the predeclared impression route. At
inference, the route is selected only from the known question type:

```text
findings question   -> findings-specialized adapter
impression question -> balanced 300-case adapter
```

No target reference, generated answer, or Validation outcome is used to route
an individual case. A 12-case screening run will be used to decide whether
the section-routed candidate merits expanded Validation generation. If it is
expanded, the same 376-case manifest, V12 rankings, prompts, decoding, and
paired metrics will be used.

This is a post-hoc development extension motivated by the observed section
asymmetry, not a blinded or preregistered claim. It remains subject to the
same automated report-reference consistency boundary and cannot establish
clinical diagnostic accuracy, safety, or physician agreement.
