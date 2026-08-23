# V9 QA and Bounded-Agent Confirmation Protocol

## Status

This protocol was written after the V9 retrieval outcome was frozen at commit
`7efcefc` and before V9 Test answers were generated or inspected. It does not
alter the frozen retrieval systems, checkpoint, Test rankings, or retrieval
results. It is a prospective protocol for the downstream QA and bounded-agent
stage, not a formal preregistration.

## Evaluation frame

The QA frame contains all 685 V9 Test cases with nonempty findings and
impression references. Each case contributes two fixed questions:

1. `findings`: What are the main radiographic findings?
2. `impression`: What is the most likely radiographic impression?

The resulting matrix contains 1,370 questions. The `acute` retrieval query is
not used for generative scoring because OpenI does not provide a
physician-adjudicated acute-abnormality binary reference. A heuristic label
would create a questionable gold standard.

The target findings and impression are evaluation-only references. They are
never included in the generator prompt, retrieval features, or agent state.

## Query and image policy

Every generation condition receives the same target clinical indication,
question, and one target chest radiograph. The image is selected
deterministically: prefer a frontal projection and then use lexicographic
filename order. Historical images remain retrieval inputs only and are not
passed to MedGemma. This isolates the effect of retrieved report evidence
while keeping the target image available in the final multimodal QA system.

## Systems

The four fixed generation conditions are:

```text
G0 No retrieval
   target image + indication + question

G1 Text RAG
   G0 + Top-3 reports from frozen R0 BM25

G2 Fixed multimodal RAG
   G0 + Top-3 reports from frozen R3 fixed fusion

G3 Learned multimodal RAG
   G0 + Top-3 reports from frozen R4 learned MLP
```

`G4` is a bounded evidence-control layer applied to `G3`. It checks only
claims presented as historical support. If those claims are unsupported by
the cited G3 reports, it performs one deterministic retry against the frozen
R1 image-image Top-3 reports. If support remains insufficient, the agent
removes the historical-support statement and citations, records an evidence
abstention, and preserves the target-image answer as explicitly unverified by
the historical-evidence checker.

The agent cannot browse, change a model, modify a threshold, access the target
report, or loop more than once. Each route, citation, support decision, retry,
revision, and reason is recorded.

## Generator and prompt

All fixed conditions use `google/medgemma-1.5-4b-it` revision
`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` under local 4-bit NF4 inference,
deterministic decoding, and `max_new_tokens=192`. The prompt states that
historical reports are analogies rather than proof about the target patient.

The requested JSON fields are:

```text
answer
target_image_findings
supporting_case_ids
historical_support
uncertainty
abstain
```

Parsing failures are retained and reported. They cannot be silently replaced
or regenerated under a changed prompt.

## Outcomes and statistics

The primary QA metric is case-grouped, equal-question Token-F1. The primary
comparison is `G3 - G0`, with a 10,000-iteration case bootstrap and 95%
confidence interval. `G1`, `G2`, per-question results, exact match, structured
output validity, latency, tokens, and GPU memory are secondary outcomes.

The agent primary outcome is the change in automated unsupported historical
support from `G3` to `G4`. Its QA-preservation rule uses a prespecified
Token-F1 noninferiority margin of `-0.01`. The checker is
`cnut1648/biolinkbert-mednli` with the frozen thresholds in the JSON config.

Automated Token-F1, NLI support, and RadGraph-style overlap are not physician
adjudication or clinical correctness. Independent clinical human evaluation
remains Future Work. No QA or agent result may trigger retrieval retuning,
prompt revision, threshold revision, case replacement, or selective rerun.

