# Post-Hoc MedGemma Wrapper-Repair Audit

The strict r1 contract pilot completed 512 Calibration generations but produced
zero contract-valid rows. Inspection showed that many answers contained a
syntactically complete option array wrapped by MedGemma's terminal
`<end_of_turn>` token and, especially in the text-only condition, a Markdown
code fence. Twenty-five rows had no complete array, usually because the
16-token ceiling truncated a multi-option response.

This audit is explicitly post-hoc parser development. It does not overwrite the
strict 0% contract result committed at `960d4fd`. Before repaired answer metrics
are computed, the deterministic rules are frozen in
`config/final_qa_medgemma_parser_audit.json`: strip one terminal model token,
strip one outer code fence, then require the complete remaining payload to be a
valid JSON array. The audit cannot extract an array from prose, complete a
truncated array, use gold answers, regenerate selected rows or delete failures.

If the repaired contract is technically adequate, complete development runs
will stop generation on `<end_of_turn>`, use 32 output tokens, and retain the
same normalizer. That future configuration must be committed before complete
Validation generation. This Calibration audit is not confirmation evidence.
