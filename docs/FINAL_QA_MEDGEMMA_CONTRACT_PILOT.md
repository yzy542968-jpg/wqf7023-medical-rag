# MedGemma Compact-Answer Contract Pilot

This Calibration-only pilot is frozen before generation. It checks whether the
local frozen MedGemma model can reliably answer Rad-ReStruct questions using a
compact option-index contract before the project spends GPU time on complete
development roles.

Two conditions use the same 256 deterministically selected questions: B1 sees
only the question and answer options; B3 additionally sees the target chest
radiograph and available pre-report indication. Neither condition receives the
target report, gold prior-answer history, historical evidence, or the complete
case-specific question list.

Sampling is SHA-256 based and stratified into 176 single-choice, 64 multi-choice
and 16 fixed-choice rows. The model must return only a JSON array of zero-based
option indices. Invalid outputs are retained as empty predictions. The pilot
reports answer-set accuracy, option micro-F1, answer-type results, contract
validity, token counts, runtime and peak VRAM.

This is a format and signal preflight, not model selection and not confirmation.
It cannot support a Test claim. Prompt changes after this run must be documented
and evaluated on Calibration before any complete Validation run.
