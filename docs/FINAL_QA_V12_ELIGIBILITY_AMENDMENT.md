# Final-QA V12 Eligibility Amendment

The V12 Calibration ranking artifact covers 376 V10 Calibration cases and
three generic query views. Six of the 358 mapped Final-QA Calibration targets
are absent: `CXR16`, `CXR614`, `CXR1536`, `CXR1761`, `CXR1778`, and `CXR2115`.
Their OpenI report text is empty and their RadGraph status is `empty_report`, so
the original V12 ranking/evaluation runner excluded them before qrel creation.

An implementation audit confirmed that V12 inference features use the target
image, available indication, generic question text, and historical-bank
features. Target report/RadGraph content is used to construct development qrels
but not to rank candidates at inference. The ranking is therefore compatible
with the report-withheld task for covered targets.

No Final-QA question is removed. For a target absent from the V12 artifact, the
V12-history candidate deterministically falls back to the already frozen
MedSigLIP image-only Top-3 ranking. The same question-conditioned fact selector
then runs unchanged. The number and SHA-256 fingerprint of fallback target IDs
are recorded in the aggregate summary. This amendment was committed after the
coverage failure was observed but before any V12-history model output was
generated; it is a transparent technical missing-input policy, not an
outcome-driven replacement.
