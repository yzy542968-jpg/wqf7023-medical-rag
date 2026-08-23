# V9 Qualitative Error Analysis

## Current status

The deterministic case pack and assistant-proposed coding are complete. The
24 cases remain pending researcher review; therefore this document records
only provisional, tool-assisted observations and does not present a final
researcher-reviewed qualitative result.

The selection contains six largest G3 gains, six largest losses, six agent
retry/recovery cases, and six historical-evidence abstention cases. Selection
and coding rules were committed before systematic extraction.

## Provisional observations

1. Retrieval-conditioned answers can improve strongly over image-only answers,
   but the effect is heterogeneous. Selected case-level mean gains reach
   `+0.6052`, while selected losses reach `-0.2333`.
2. Multimodal retrieval improvement does not guarantee a reference-consistent
   final answer. Generator interpretation and output-format failure remain
   separate downstream error stages.
3. The learned R4 route sometimes produces a historical-support statement
   that the frozen NLI checker cannot substantiate from cited reports.
4. A backup R1 image-image route occasionally recovers support, but most
   failed checks end in historical-evidence abstention.
5. Removing unsubstantiated historical support preserves the target-image
   answer by design. It improves traceability, not verified diagnostic
   accuracy.

## Required researcher action

The student must inspect every row in the local review pack and either confirm,
refine, or exclude the proposals. Particular attention is needed for whether
the target-image answer is visibly aligned with the radiograph and frozen
reference, because the automated historical-report checker cannot adjudicate
image findings.

Until that review is complete, Chapter 4/5 may report the quantitative agent
results and describe this qualitative analysis as pending. It must not report
the assistant label counts as human findings.

