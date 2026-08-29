# Deployable Structured-Head Development

The zero-shot MedGemma Calibration pilot showed that parser repair alone was
insufficient: repaired option micro-F1 remained low. Before spending several
hours on complete zero-shot generation, this addendum freezes a trainable,
low-cost structured reconstruction experiment using existing frozen multimodal
representations.

One MLP receives the target MedSigLIP image embedding, a Train-fitted indication
representation, a retrieved historical report embedding, retrieval similarity
and a history-presence flag. History is dropped in 50% of Train presentations.
The identical checkpoint can therefore be evaluated with history absent and
with the Top-1 paired historical report present. This isolates the effect of
historical report information more cleanly than comparing separately trained
models.

The historical payload is the frozen MedSigLIP embedding of the real historical
report text. It is not the case's gold Rad-ReStruct answer vector. Retrieval is
based on target-image similarity against mapped V10 Train cases and excludes the
target's complete duplicate cluster. Calibration controls early stopping and
the prespecified threshold grid; Validation is reported once; Test is
prohibited.

This experiment belongs to the protocol's secondary hierarchy-consistent
report-vector reconstruction mode. It can show whether paired report text adds
structured answer signal, but it does not replace the final independent-QA arm,
produce human-readable fact citations, or establish clinical diagnostic
accuracy.
