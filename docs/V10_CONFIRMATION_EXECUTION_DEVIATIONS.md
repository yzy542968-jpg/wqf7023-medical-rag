# V10 Confirmation Execution Deviations

This log records technical execution deviations without changing the frozen
V10 models, data, metrics, thresholds, prompts, or decision rules.

## Retrieval process timeout

The first retrieval process was terminated by an external 120-second command
limit after aligned scoring and 9 of 100 shuffled assignments. Formal artifacts
are written only after all assignments finish, so no partial result was retained
or used. The identical command and frozen configuration were rerun with a longer
process allowance and completed all 100 assignments.

## QA retrieval-system identifier

The first QA command stopped before model loading or generation because its
preflight matrix check requested internal retrieval identifier `r4_original`,
whereas the frozen retrieval output uses `r4_nine_feature`. No QA output or
outcome existed. The lookup string was corrected to the already-prespecified R4
comparator; no scientific condition or setting changed.
