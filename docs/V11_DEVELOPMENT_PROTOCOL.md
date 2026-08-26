# V11 Development Protocol

## Status and purpose

V11 is a development extension after the V10 technical freeze. It addresses the largest remaining engineering risks: report-derived relevance bias, whole-report context inflation, fixed question roles, malformed/overlong generation, and forced use of weak historical evidence.

V11 is **not** a new confirmation result, formal preregistration, clinical validation, or external validation. The V10 quantitative results remain frozen and are not recomputed with V11 definitions.

## Research question

Can a transparent hierarchical retrieval-and-grounding pipeline improve the efficiency and auditability of multimodal similar-case RAG by retaining case-level retrieval while selecting question-relevant facts within each retrieved case?

The primary engineering comparison is:

```text
case-level whole report
vs
case-level -> sentence selection
vs
case-level -> fact/sentence selection
vs
case-level -> fact/sentence selection + selective history gate
```

## Data and split discipline

- Source: the existing processed OpenI/IU-Xray case records.
- Development source: V10 `train` partition.
- Validation source: V10 `validation` partition.
- V10 `test` partition is not used for V11 development.
- No final V11 confirmation case IDs are instantiated in this phase.
- Case IDs, not individual questions, are the unit of split isolation.
- Patient-level independence remains unverifiable in the processed source because reliable patient identifiers are not available.

## qrel-v2 proxy

The new qrel-v2 score is a transparent report-derived proxy. It combines:

| Component | Weight | Intended signal |
|---|---:|---|
| lesion type | 0.25 | shared observation concepts |
| anatomy/location | 0.20 | shared anatomic concepts and relation targets |
| severity | 0.10 | shared severity modifiers |
| polarity | 0.15 | present/absent/uncertain status |
| uncertainty | 0.10 | qualified versus definite statements |
| indication | 0.10 | lexical indication overlap |
| report label | 0.10 | active report-index labels |

Empty component sets receive zero similarity; they do not receive automatic full credit. Component availability is reported explicitly. The conservative primary score keeps unavailable components at zero, while an availability-normalized sensitivity score excludes unavailable component weights from its denominator; the latter must never be described as a clinical score. Relevance must be computed against the complete candidate bank before Top-k diagnostics, so the ideal list is not defined by the shortlist itself. Results must include full-bank nDCG, qrel>=0.5 relevant-item recall@100, availability coverage and full/fact-only/label-only sensitivity modes. The current OpenI `problems` field is described as `report-indexed normal`, `report-indexed abnormal` or `report-index indeterminate`, never as independent clinical adjudication.

## Retrieval and fact selection

The system first retrieves cases from the development corpus. A future learned reranker may use a fixed Top-100 shortlist, but a target/relevant case outside that shortlist is retained as a retrieval failure and is not manufactured into a pairwise positive. The selected Top-3 cases are then processed independently:

```text
target image + indication + natural question
    -> case-level candidate retrieval
    -> Top-3 case IDs
    -> sentence/RadGraph fact ranking within each case
    -> compact evidence with case_id, section and source hash
    -> target-image answer generation
```

The selector has fixed development budgets: at most two units per case, six units overall and 1,200 characters. These are engineering operating points and must be selected on development/validation, not confirmation outcomes. The current audit keeps case-level retrieval as the first stage and reports the large residual candidate-retrieval gap separately from evidence compression.

## Natural question planner

The initial planner is deterministic and inspectable. It covers presence, location, severity, comparison/change, device, uncertainty and summary intents. It changes evidence preference and answer style; it does not diagnose the image or substitute for a radiologist. A future language-model planner is optional and must beat the deterministic planner on a predefined question benchmark before adoption.

## Generation contract

The preferred V11 generator interface is answer-only generation followed by deterministic metadata assembly:

```text
target image + compact historical evidence
    -> MedGemma answer (at most two complete sentences)
    -> deterministic abstention and case/section/fact provenance attachment
```

The compact JSON interface remains a diagnostic alternative and returns:

```json
{"answer":"...","uncertainty":"low|medium|high","abstain":false,"evidence":["case:section:type:index"]}
```

The answer is bounded to two complete sentences. In JSON mode, evidence IDs are checked against the selected units; invalid or unknown IDs invalidate the structured output rather than being silently accepted. In answer-only mode, no model-generated evidence IDs are accepted: provenance fields are assembled deterministically by code. Historical reports are explicitly labeled as other-patient analogies and never as target-patient proof.

## Selective historical-evidence gate

The gate uses top score, Top-1/Top-2 margin, component agreement, evidence coverage and ensemble dispersion. Its threshold is fitted on the development/validation workflow only using report-derived proxy relevance. We will report a risk-coverage curve. Without physician labels, this is retrieval selectivity, not clinical risk calibration or a safety guarantee.

## Planned development matrix

1. Whole-report evidence.
2. Sentence-only evidence.
3. Hierarchical case-to-fact evidence.
4. Hierarchical evidence plus selective history suppression.
5. For each condition: report qrel-v2 nDCG/MRR/Hit@k proxy, character count, unit count, provenance completeness, and abstention rate.
6. If MedGemma is run, report structured validity, token-ceiling rate, answer token-F1 against the existing report reference, and evidence provenance validity separately.

No V11 model is promoted on a single metric. Any apparent improvement with a confidence interval crossing zero is described as numerical, not confirmed superiority.

## Stopping and promotion

V11 development stops before any confirmation run when:

- qrel-v2 definitions and budgets are frozen;
- the development ablation is reproducible;
- V10 remains unchanged;
- the output contract and negative/abstention path pass tests;
- model-dependent generation results are clearly separated from deterministic evidence results.

Promotion to a confirmation study requires a new confirmation protocol, a new case-disjoint cohort manifest, and a new frozen configuration. Human review and authorized external validation remain future work.
