# V16 Development Amendment R1

## Scope

This amendment records additional V16 development candidates after the first
100-case, 64-forward-step QLoRA pilot. It does not change V10/V11/V15 frozen
artifacts, does not authorize Test evaluation, and does not constitute a
confirmation protocol.

The first pilot used only four optimizer updates and showed no reliable
Token-F1 or RadGraph improvement over the frozen MedGemma base. Its reduced
token-ceiling rate is retained as an engineering observation, not a quality
claim.

## Candidate Changes

The next development candidates may vary only the following predeclared
training factors:

1. LoRA target profile: `qv` (`q_proj` and `v_proj`) versus the initial `all`
   language-projection profile.
2. Learning rate: `5e-5` for the lower-update-drift candidate.
3. Training conditions: retrieved-history-only versus the balanced set of
   no-history, retrieved-history, and random-history examples.
4. Case subset: a stable SHA-256 case ordering for pilot-size development
   subsets, rather than lexical case-ID order.
5. Forward-step budget: 256 steps, with a fixed case-level internal split and
   no use of confirmation/Test outcomes.

The next balanced candidate uses all three predeclared training conditions
(`no_history`, `retrieved_history`, and `random_history`) with the `qv`
profile, learning rate `5e-5`, a stable 300-case development subset, and 256
forward steps. This candidate is intended to test whether the retrieval gain
survives without the no-history regression observed in the retrieved-only
candidate; it is not selected from Validation outcomes.

A longer-training candidate is also predeclared before its run: the same
three-condition `qv` recipe and learning rate on a stable 1,000-case Train
subset for 1,024 forward steps (64 optimizer updates at accumulation 16).
This tests whether the promising 256-step result is under-trained. It remains
a Validation development candidate only; no checkpoint or output may be
selected using Test outcomes.

The candidate runs are exploratory and are not selected by inspecting Test
results. A candidate may be retained only if it improves the predefined
Validation comparison without a material regression in report-level pathology
labels, RadGraph complete score, output contract, or provenance validity.

## Interpretation Boundary

All results remain automated consistency measures against same-source report
references. They are not clinical diagnosis accuracy, safety, physician
agreement, or external validation. A higher Token-F1 alone is insufficient for
promotion.
