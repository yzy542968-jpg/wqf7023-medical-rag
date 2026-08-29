# Final-QA Post-Pilot Development Amendment

## Purpose

The 192-step QLoRA pilot passed its prespecified base-model promotion rule, but
the relevant-history effect over the adapted no-history condition was not
statistically supported. This amendment is committed before further adapter
training and before generating V12 rankings for the Final-QA Calibration role.
It governs Calibration development only. Validation and Test remain prohibited.

## Adapter-duration decision

Candidate A is the completed 192-forward-step q/v QLoRA adapter. Candidate B is
trained from the same base revision, same 1,800 Train examples, same ordering
rule, same optimizer, and same case-level internal split for 384 forward steps.
Both candidates are evaluated on the same 256 Calibration questions under B3
no history and the current P1 history condition.

Candidate B advances only if its mean option micro-F1 across B3 and P1 exceeds
Candidate A by at least 0.010, neither condition decreases by more than 0.005,
and contract validity does not decrease by more than 0.010. If Candidate B
advances, one final 576-step candidate may be trained under the same rules. The
576-step candidate must satisfy the same rule relative to Candidate B. The
first failed step increase stops duration development. Exact ties select the
shorter adapter.

## Retrieval-policy decision

The completed P1 pilot used MedSigLIP image-only Top-3 retrieval followed by
question-conditioned hierarchical fact selection. The new candidate replaces
only the case ranking with the frozen V12 LambdaMART retrieval stack and retains
the same Top-3, fact selector, evidence budget, generator, adapter, parser, and
Calibration questions.

V12 rankings are regenerated for the mapped V10 Calibration cases against the
V10 Train historical bank. Only the generic `findings` query view is used for
the primary structured-QA candidate because Rad-ReStruct questions instantiate
findings-oriented report elements. The specific QA question still controls
within-case fact selection. Target report text, target RadGraph facts, and QA
answers are unavailable to retrieval and generation.

The V12 policy advances over image-only P1 only if option micro-F1 is higher,
contract validity decreases by no more than 0.010, and negative transfer from
B3 does not increase. A positive point estimate is a development selection
criterion, not confirmatory evidence.

## Audit and stopping boundaries

- All parameter fitting uses Train only.
- All decisions in this amendment use Calibration only.
- Duplicate-cluster exclusion and deterministic provenance remain mandatory.
- The frozen V10-V16 results and the completed 192-step pilot are not altered.
- Validation is accessed only after a separate Validation protocol commits the
  selected adapter candidates and retrieval/evidence grid.
- Test remains untouched until a development decision record and confirmation
  protocol are committed.
- Failure to improve is retained as a negative development result.
