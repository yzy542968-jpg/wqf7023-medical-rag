# V17 Development Decision Record

## Final decision

V17 stops after Calibration development and does not advance to Final-QA
Validation or Test. Final-QA Test was never accessed by V17.

The stopping decision is based on the complete evidence chain rather than a
single favorable metric:

1. Actual-question-conditioned retrieval passed all retrieval-only Go checks.
2. Base MedGemma compact-fact generation showed a relevance-specific gain but
   used the wrong generator arm for comparison with Final-QA; the deviation and
   result were retained.
3. The corrected QLoRA-384 compact-fact run improved positive-answer accuracy
   but reduced primary Exact Accuracy and increased negative transfer.
4. A cross-fitted inference-time relevance threshold selected no history in all
   five outer folds and was stopped.
5. Restoring the QLoRA-familiar Top-1 whole-report format produced the highest
   Related Exact Accuracy and a numerical Related-over-Random difference, but
   the confidence interval crossed zero and Mismatched was numerically higher.
6. A pre-existing frozen question-ID gate placed Related above both matched
   controls numerically, but both relevance-specific confidence intervals
   crossed zero.

## Selected findings

- Retrieval finding: supported on report-derived proxy relevance.
- History-versus-no-history QA finding: supported for the whole-report
  extension on this Calibration pilot.
- Related-versus-random QA superiority: numerical only, not confirmed.
- Related-versus-mismatched QA superiority: not confirmed.
- Compact fact evidence as a replacement for whole reports: not promoted.
- TF-IDF relevance threshold gate: not promoted.
- Pre-existing question-ID gate: retained only as sensitivity analysis.

## Frozen implementation choices

- Retrieval recipe: `full_query` for V17 exploratory analysis.
- Image shortlist: frozen MedSigLIP Top-100.
- Retrieval output: Top-3 for compact evidence; Top-1 for whole-report
  extension.
- Generator: previously selected Final-QA QLoRA-384 adapter for comparable
  results.
- Primary metric: Exact Accuracy.
- Statistical unit: case ID; 10,000 bootstrap replicates.
- Raw per-question rows remain local; compact summaries and hashes are public.

## Why no Validation run follows

The protocol allowed Validation only after a coherent development method was
frozen. V17 did not establish the required relevance-specific downstream QA
advantage. Running Validation and then selecting whichever policy looked best
would weaken the study. The correct outcome is a mixed/negative development
extension with a clear mechanistic diagnosis.

## Future method, not current result

A future study may train evidence-use behavior explicitly with relevant,
conflicting, random, and no-history contexts while preserving a new untouched
case/cluster split. That is a new training study, not a permitted post-hoc fix to
V17. Physician/blinded evaluation and external patient-level validation remain
Future Work.

