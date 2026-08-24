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

## QA process segmentation and offline loading

The 4,544-row QA matrix exceeded the external 30-minute command allowance.
The frozen runner resumed by exact case/question/system key and completed in
four process segments without overwriting completed rows. Hugging Face startup
also attempted network metadata checks despite locally cached weights. Later
segments set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; this affected only
artifact lookup and did not change the cached model revision, weights, prompts,
decoding, batch size, or outputs already completed.
