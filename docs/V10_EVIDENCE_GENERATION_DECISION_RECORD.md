# V10 Evidence and Compact Generation Decision Record

Status: evidence policy and generation interface frozen; V10 Test not run.

## Execution record

The prompt-only two-stage JSON attempt failed technically and was terminated
after 144 rows. Revision 1 generated all 2,256 planned Validation rows using an
answer-first interface and deterministic provenance/schema assembly. The
previous V9 structured-output failure mode was removed: all three evidence
conditions achieved 100% assembled-schema validity, 100% provenance integrity,
and 100% nonempty answer-stage validity.

The raw Revision 1 rows have SHA-256
`e1ac34c2c816fbac6cc0840c7754eabc99552bf6ad5f296e7d3ee8685c4e5dc4`.
The prespecified two-sentence finalizer produced authoritative rows with
SHA-256
`9c6e2783bed180d3dd60b384b50642c2faf9219567e579806dd84ae9c108039e`.

## Validation policy result

| Evidence policy | Token-F1 | Mean evidence characters | Assembled valid | Provenance valid |
|---|---:|---:|---:|---:|
| E0 whole findings + impression | **0.204190** | 748.0 | 1.000 | 1.000 |
| E1 Top-3 sentences per case | 0.185220 | 415.5 | 1.000 | 1.000 |
| E2 Top-2 sentences + Top-5 facts per case | 0.178818 | 1038.9 | 1.000 | 1.000 |

Whole-report E0 exceeded E1 by 0.018970 and E2 by 0.025372, both greater than
the frozen 0.005 material-difference threshold. E0 is selected for G1/G2/G3
generation input. The compactness tie-break was not invoked.

The negative E1/E2 result is retained. Fact-aware R5 improved case retrieval,
but supplying finer sentence/fact units to MedGemma did not improve automated
report-reference consistency. This separates the value of facts as reranking
features from their value as generator context.

## Locked generation interface

MedGemma generates answer text only. Python retains at most two complete
sentences, assigns conservative deterministic uncertainty, attaches at most one
traceable historical evidence unit per retrieved case, and assembles the final
schema. Historical citations are labelled retrieved evidence, not clinically
adjudicated support.

The raw answer token-ceiling rate remained substantial (61.4% for E0), so V10
does not claim that free-text continuation was eliminated. It claims the more
specific engineering result that continuation no longer corrupts the final
structured record and that bounded answer extraction is deterministic.

No later stage may change the evidence policy, prompt, answer ceiling, stop
token, normalization, provenance rule, or selection criterion in response to
Calibration or Test outcomes.
