# V11 MedGemma Generation Development Results

## Scope and status

This document records a clean, deterministic MedGemma development diagnostic
run after the V10 technical freeze. It uses only the V10 Train and Validation
partitions, does not instantiate a V11 confirmation cohort, and does not alter
any V10 model, parameter, prompt, result, or Test output.

The run compares three evidence policies under the same target image,
indication, question roles, BM25 Top-3 case shortlist, MedGemma revision and
decoding settings:

1. `whole_report`
2. `sentence_only`
3. `case_to_fact`

The earlier interrupted four-policy file is retained as a development trace
only. It is not used for the results below.

## Reproducible run

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
& ".\\.venv\\Scripts\\python.exe" scripts\\run_v11_medgemma_development.py `
  --max-cases 48 `
  --stratify-spectrum `
  --policies whole_report sentence_only case_to_fact `
  --batch-size 4 `
  --rows-output experiments\\v11_development\\v11_medgemma_generation_48_clean_rows.jsonl `
  --summary-output data\\splits\\v11\\v11_medgemma_generation_48_clean_summary.json
```

The deterministic case selector produced 48 Validation cases: 24
report-indexed normal and 24 report-indexed abnormal. The source case IDs are
identified by the SHA-256 recorded in the machine-readable summary. There are
432 rows: 48 cases x 3 question roles x 3 policies. The model revision is the
pinned `google/medgemma-1.5-4b-it` revision already used by the repository.

## Results

| Policy | Token-F1, all rows | Token-F1, non-proxy rows | Mean input tokens | Mean evidence characters | Token-ceiling rate |
|---|---:|---:|---:|---:|---:|
| Whole report | 0.1312 | 0.1336 | 798.2 | 672.3 | 11.81% |
| Sentence only | 0.1451 | 0.1279 | 604.1 | 351.9 | 9.72% |
| Case-to-fact | 0.1531 | 0.1304 | 539.3 | 245.9 | 11.81% |

The case-to-fact policy reduced mean evidence characters by approximately
63.4% relative to whole-report evidence and reduced mean input tokens by
approximately 32.4%. It had the highest all-row Token-F1 in this diagnostic,
but the non-proxy comparison is close to the sentence-only policy and the run
does not provide an independent clinical reference. The results therefore
support an engineering efficiency and auditability direction, not a confirmed
answer-quality improvement.

## Automated semantic metrics and paired uncertainty

The frozen 432 rows were subsequently evaluated with
`modern-radgraph-xl`. Rows with an empty source-report reference were retained
and assigned zero overlap rather than being removed after outcome inspection.
The case-grouped bootstrap used 10,000 deterministic resamples over the same
48 cases.

| Policy | Token-F1 (95% CI) | Complete F1RadGraph (95% CI) |
|---|---:|---:|
| Whole report | 0.1312 [0.1033, 0.1612] | 0.0669 [0.0470, 0.0883] |
| Sentence only | 0.1451 [0.1165, 0.1757] | 0.0845 [0.0597, 0.1121] |
| Case-to-fact | 0.1531 [0.1260, 0.1814] | 0.0809 [0.0601, 0.1029] |

The prespecified `case_to_fact - whole_report` Token-F1 difference was
`+0.02195`, 95% CI `[-0.00026, +0.04302]`. The corresponding complete
F1RadGraph difference was `+0.01395`, 95% CI
`[-0.00691, +0.03442]`. Both intervals crossed zero. The clean run therefore
supports substantial context compression, lower input-token use and complete
provenance, but it does not confirm superior answer quality.

## Output-contract findings

- Answer-only contract validity: `100%` for all three policies.
- Deterministic evidence provenance validity: `100%` for all three policies.
- Raw compact-JSON validity: `0%` by design, because this run uses the
  answer-only generation path and attaches provenance deterministically.
- Structured-output validity and parser-repair rates are therefore not the
  primary success criteria for this run.
- Peak allocated GPU memory was approximately `4,261.6 MiB` with batch size 4.

The answer-only contract was selected to separate language generation from
machine-verifiable metadata assembly. This avoids treating a malformed model
JSON object as a provenance failure, while still retaining compact-JSON
parsing as a separate diagnostic path.

## Interpretation boundary

This is a development-only, same-source, report-reference diagnostic. The
findings do not establish physician-adjudicated correctness, clinical safety,
clinical usefulness, external generalization, or a superiority claim for
MedGemma. The acute question role uses a report-field proxy reference and is
not an independently adjudicated acute-answer label. Human blind review and
external patient-disjoint validation remain Future Work.

Machine-readable outputs:

- `data/splits/v11/v11_medgemma_generation_48_clean_summary.json`
- `data/splits/v11/v11_medgemma_generation_48_statistical_summary.json`
- `experiments/v11_development/v11_medgemma_generation_48_clean_rows.jsonl`

The development result does not authorize a V11 confirmation run. Any future
confirmation study would require a new protocol, explicit candidate-pool and
generation policies, a case-disjoint cohort manifest, and a new frozen
configuration before Test outcomes are inspected.
