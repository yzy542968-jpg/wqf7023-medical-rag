# Paired Report-Text RAG Calibration Pilot

This pilot tests real historical report text rather than gold historical answer
vectors or one compressed report embedding. It is frozen before RAG generation
and uses the same 256 Calibration questions selected for the compact-contract
pilot.

Every condition receives the same target chest radiograph, available indication,
question and answer options. B3 receives no history. B4 receives one
deterministic random other-cluster Train report. B6 receives the whole Findings
and Impression of the Top-1 MedSigLIP image neighbor. P1 retrieves the Top-3
image neighbors and uses the frozen V11 planner and within-case selector to
provide at most two question-relevant sentences or RadGraph facts per case and
six units in total.

Historical cases remain separate and every evidence unit retains case, section,
unit, position and source-hash provenance. The target report, gold answer
history and historical Rad-ReStruct vectors are prohibited. MedGemma uses
greedy decoding, a 32-token ceiling, `<end_of_turn>` stopping and the previously
frozen bounded wrapper normalizer.

This is a Calibration mechanism pilot. It determines whether whole-report or
question-conditioned real history deserves a complete development run. It is
not Validation, Test, physician-rated accuracy or external validation.
