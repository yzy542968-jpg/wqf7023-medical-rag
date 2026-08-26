# V11 Technology Reuse Audit
## Scope

This audit defines what the V11 development extension reuses from the frozen V10/V9 implementation and what it adds. It is an implementation boundary, not a claim that V11 has already been confirmed on an untouched test set.

## Reused components

| Existing asset | V11 use | Boundary |
|---|---|---|
| `v10_evidence.EvidenceUnit` | sentence/fact provenance records | V10 behavior remains unchanged |
| V10 sentence and RadGraph fact extraction | input units for hierarchical selection | V11 adds a new selector |
| V10 R4/R5 retrieval runtime | reference architecture and future integration point | no V10 checkpoint or result is modified |
| MedSigLIP and MedGemma interfaces | planned runtime integration | no new model result is claimed by this audit |
| deterministic V10 output assembly | provenance design pattern | V11 uses a separate compact contract |
| V10 cluster-disjoint split | development/validation bookkeeping | V11 does not instantiate confirmation IDs |

## New V11 components

1. `v11_qrel.py` defines a structured report-derived relevance proxy with explicit lesion, anatomy, severity, polarity, uncertainty, indication and report-label components.
2. `v11_question_planner.py` maps a natural question to an inspectable evidence preference plan.
3. `v11_evidence.py` retains case-level retrieval, then selects sentence/fact units within each retrieved case while preserving case and section provenance.
4. `v11_selective.py` computes retrieval confidence and a validation-fitted history-use gate. The gate is a research proxy, not a clinical safety threshold.
5. `v11_output_contract.py` constrains the answer schema and assembles provenance deterministically after generation.

## Explicit non-goals

- No V10 model, threshold, prompt, result, confirmation split, or freeze artifact is edited.
- No physician annotation is fabricated.
- No external MIMIC-CXR or other dataset result is claimed.
- No report-derived qrel is treated as clinical correctness.
- No V11 confirmation cohort is generated before a later, separate confirmation protocol.

## Technical risk register

| Risk | Current handling | Remaining requirement |
|---|---|---|
| Weak report-derived relevance labels | qrel-v2 components and sensitivity modes are exposed | physician-reviewed relevance or an authorized external benchmark |
| Cross-case contamination | evidence units retain `case_id`; only within-case units are selected | automated integration test in the eventual runtime |
| Long-context truncation | compact fact/sentence budgets and two-stage output contract | run MedGemma generation on development/validation |
| Wrong historical evidence | selective gate and explicit abstention path | human safety/utility review |
| Planner overfitting | deterministic rules are transparent and separately scored | natural clinical question benchmark and reviewer audit |
| Confirmation leakage | development-only protocol and no case instantiation | separate confirmation protocol and manifest |
