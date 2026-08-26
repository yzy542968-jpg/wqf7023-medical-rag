# V11 MedGemma Generation Preflight

## Scope

This is a one-case, 12-row development preflight for the V11 generation path.
It uses three fixed questions and four evidence policies on the V10 Validation
partition. It is not a final V11 validation estimate, confirmation result,
clinical evaluation or external validation.

Command:

```powershell
.\.venv\Scripts\python.exe scripts\run_v11_medgemma_development.py --max-cases 1 --batch-size 1
```

The resumable full development command is the same script without
`--max-cases`; its output is intentionally local because it contains model
generations and evidence text.

## Why two output modes are recorded

The initial compact-JSON prompt required the 4B multimodal model to generate
the answer, uncertainty, abstention flag and long evidence IDs in one output.
On the first 24-row smoke run, raw JSON validity was 0% and the output hit the
96-token ceiling on nearly every policy. The model usually emitted a readable
key-per-line format instead of JSON.

The development default therefore uses the more defensible two-stage contract:

```text
MedGemma generates only a bounded answer
        -> deterministic code attaches abstention and provenance
```

The JSON parser remains available as a diagnostic. A deterministic YAML-like
repair is marked `parser_repaired` and is never counted as raw JSON success.

## One-case preflight result

| Policy | Answer-contract valid | Raw JSON valid | Token-ceiling rate | Mean input tokens | Mean output tokens | Non-proxy Token-F1 |
|---|---:|---:|---:|---:|---:|---:|
| whole report | 100% | 0% | 33.3% | 720.7 | 55.0 | 0.0917 |
| sentence only | 100% | 0% | 33.3% | 618.3 | 64.7 | 0.0566 |
| case-to-fact | 100% | 0% | 0% | 548.7 | 26.3 | 0.0943 |
| case-to-fact + selective gate | 100% | 0% | 0% | 548.7 | 26.3 | 0.0943 |

All four policies had 100% deterministic evidence-provenance validity in this
preflight. These numbers are diagnostic only; one case cannot support a model
comparison or a clinical conclusion.

## Engineering decision

The preferred output path is **answer-only generation plus deterministic
provenance assembly**. Whole-report context is more expensive and more prone
to token-ceiling pressure. Case-to-fact evidence is substantially shorter in
the preflight while preserving the same provenance checks. The result does
not show that case-to-fact improves medical answer accuracy; a larger
development-only run would be needed before any future confirmation protocol.
