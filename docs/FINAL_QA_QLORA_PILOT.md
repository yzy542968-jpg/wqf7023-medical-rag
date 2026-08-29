# MedGemma QLoRA Balanced-History Pilot

Zero-shot MedGemma remained strongly biased toward option 0 after bounded parser
repair. This pilot tests whether parameter-efficient Train-only adaptation can
learn the structured answer contract, correct class imbalance and resist
irrelevant history before any complete Validation run.

Six hundred base questions are selected deterministically from mapped V10 Train
cases: 160 binary yes, 160 binary no, 70 non-binary single-choice, 180
multi-choice and all 30 fixed-choice rows. Each question produces three
supervised examples with the same target answer: no history, one deterministic
random other-cluster report, and question-conditioned evidence from the Top-3
image-neighbor reports. The prompt never contains the target report, gold prior
answers or historical gold vectors.

The fixed MedGemma revision is loaded in NF4 and only q/v LoRA parameters are
trained. The pilot is capped at 192 forward steps with gradient accumulation 16
and uses a deterministic 10% case-level internal split. The existing 256-row
Calibration pilot then compares base and adapter under no-history and relevant-
history inputs.

The adapter is promoted only if option micro-F1 improves in both conditions
without a material contract-validity regression. A negative result is retained.
This is Train/Calibration model development, not Validation, Test or clinical
accuracy evidence.
