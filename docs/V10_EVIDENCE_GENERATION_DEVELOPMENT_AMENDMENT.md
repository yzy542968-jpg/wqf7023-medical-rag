# V10 Evidence and Compact Generation Development Amendment

Status: frozen before V10 compact-generation Validation outcomes.

This amendment operationalizes Sections 8, 9, and 12 of the V10 development
protocol. V10 Test is not read.

## Validation frame

The frame contains every technically eligible V10 Validation case and the
`findings` and `impression` questions. Retrieval is the frozen R5-attention
ensemble and supplies three historical cases per question. Each question is
run under all three frozen evidence policies:

- E0: all findings and impression sentences from each retrieved case;
- E1: three question-relevant sentences from each retrieved case;
- E2: two question-relevant sentences plus five question-relevant RadGraph
  facts from each retrieved case.

Evidence is selected independently inside each case and always retains case,
section, unit, and source-hash provenance.

## Generator and decoding

Both stages use frozen local `google/medgemma-1.5-4b-it` revision
`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` with NF4 double quantization,
bfloat16 compute, deterministic greedy decoding, and batch size 8.

1. The image-conditioned answer stage has a 64-token ceiling and returns only
   `{"a":"answer","u":"low|medium|high"}`.
2. The text-only historical-support stage has a 96-token ceiling and returns
   only `{"s":[{"p":"provenance_id","t":"short support"}]}`.
3. Python filters provenance IDs against selected evidence and assembles the
   final schema. Invalid model fields are never invented or silently repaired.

The runner is resumable by `(case_id, question_type, evidence_policy)` and
retains raw generations locally. Public summaries contain only aggregate
metrics, configuration hashes, artifact hashes, and outcome boundaries.

## Selection rule

Primary metric is equal-question Token-F1 against the same-source report
reference, with case-averaged Token-F1 as a required diagnostic. The assembled
schema-valid and citation-valid rates must both be at least 0.99. Among eligible
policies, select the highest equal-question Token-F1. If the best policy exceeds
another by less than 0.005, apply the frozen compactness preference E2, then E1,
then E0. Mean input characters and selected evidence-unit counts are reported.

This is automated report-reference consistency, not physician-adjudicated
clinical correctness.

