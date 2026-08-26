# V10 Post-hoc Qrel and Spectrum Sensitivity Audit
Status: complete, exploratory, and separate from the frozen V10 confirmation.

This audit reads the frozen V10 retrieval rows and recomputes evaluation under
three relevance constructions. It does not rerun a model, change a checkpoint,
modify a ranking, select a threshold, or replace any frozen V10 result.

## Scope and inputs

The audit uses the same 2,510-case Train historical bank, 574-case cluster-
disjoint Test partition, and three frozen question roles. Ranking rows exist
for 568 technically eligible Test cases. The six cases without retrieval rows
are retained as explicit exclusions and are not silently replaced:

```text
CXR894  CXR1293  CXR1297  CXR1615  CXR2601  CXR2765
```

The evaluated rows contain five systems: R0 BM25, R1 image-image, R2
image-report, R4 nine-feature, and R5 fact-attention. R3 fixed multimodal is
listed in the V10 development protocol but is absent from the frozen V10
confirmation rows. This is recorded as a scope difference, not reconstructed
after the Test result.

The executable audit is:

```text
scripts/audit_v10_qrel_sensitivity.py
```

The machine-readable output is:

```text
data/splits/v10/v10_qrel_sensitivity_summary.json
```

Input SHA-256 values are stored in that output. The audit uses 10,000
case-bootstrap iterations with seed 7051.

## Relevance variants

The following variants are evaluated on the identical frozen rankings:

| Variant | Active-label weight | RadGraph-fact weight |
|---|---:|---:|
| Combined | 0.60 | 0.40 |
| Label-only | 1.00 | 0.00 |
| Fact-only | 0.00 | 1.00 |

These variants are sensitivity analyses, not new confirmatory endpoints. None
is physician-adjudicated clinical similarity. The combined result remains the
original V10 primary construct.

## Main results

| Qrel | R4 nDCG@10 | R5 nDCG@10 | R5 minus R4 | 95% case-bootstrap CI | Interpretation |
|---|---:|---:|---:|---:|---|
| Combined | 0.349049 | 0.360074 | +0.011025 | [+0.007698, +0.014455] | Positive under frozen construct |
| Label-only | 0.337255 | 0.342424 | +0.005169 | [+0.000964, +0.009511] | Positive overall, but spectrum-dependent |
| Fact-only | 0.310757 | 0.331594 | +0.020837 | [+0.017495, +0.024225] | Positive under fact-only construct |

The combined result reproduces the frozen V10 primary retrieval result within
bootstrap sampling precision. The fact-only result is not evidence that R5 is
clinically correct: R5 also uses RadGraph-derived features, so this sensitivity
can expose shared representation dependence rather than remove it.

## Report-indexed spectrum analysis

The evaluated Test cases contain 195 report-indexed normal, 359
report-indexed abnormal, and 14 report-index indeterminate cases. These labels
come from the OpenI `problems` field and are not independent clinical
adjudication.

| Qrel | Subgroup | n | R5 minus R4 | 95% CI | Interpretation |
|---|---|---:|---:|---:|---|
| Combined | Normal | 195 | +0.026304 | [+0.019815, +0.032972] | Confirmed under this construct |
| Combined | Abnormal | 359 | +0.002153 | [-0.001285, +0.005595] | Numerical only |
| Combined | Indeterminate | 14 | +0.025726 | [-0.002307, +0.057431] | Numerical only |
| Label-only | Normal | 195 | +0.026528 | [+0.017663, +0.035633] | Confirmed under this construct |
| Label-only | Abnormal | 359 | -0.007331 | [-0.010916, -0.003808] | R5 lower than R4 |
| Label-only | Indeterminate | 14 | +0.028202 | [-0.010241, +0.073430] | Unresolved |
| Fact-only | Normal | 195 | +0.020764 | [+0.015145, +0.026455] | Confirmed under this construct |
| Fact-only | Abnormal | 359 | +0.021039 | [+0.016860, +0.025409] | Confirmed under this construct |
| Fact-only | Indeterminate | 14 | +0.016694 | [+0.004929, +0.028889] | Exploratory; small subgroup |

## Construct audit

All 195 evaluated normal cases and all 14 indeterminate cases have empty
active-label sets under the current `problems`-field normalization. The current
similarity function assigns an empty-versus-empty active-label similarity of
1.0. Consequently, each normal or indeterminate query has an average of 968
Train candidates with combined qrel at least 0.50, whereas abnormal queries
have an average of 9.36 such candidates.

This does not prove that the V10 ranking is wrong. It does show that the
combined qrel has a very different interpretation across the spectrum:

```text
normal/indeterminate: broad agreement with "no active indexed label"
abnormal:             specific active-label and fact overlap
```

The primary combined result must therefore be described as performance under a
report-derived qrel, not as a universal clinical-similarity result.

## Required interpretation changes

The following wording is permitted:

> R5 improved overall nDCG@10 relative to R4 under the frozen report-derived
> relevance construct. The improvement was robust to a fact-only sensitivity
> analysis, while label-only results showed spectrum-dependent behavior.

The following wording is not supported:

> R5 improved retrieval for all patient types.

> R5 was clinically more accurate than R4.

> RadGraph fact features independently caused the improvement.

The abnormal label-only decrease and the combined abnormal confidence interval
crossing zero should be reported as limitations or mixed findings, not hidden
behind the overall mean.

## Future work boundary

This audit does not replace clinical review. The prepared blinded package is
retained as Future Work because no independent reviewer result exists. A future
study should define relevance before confirmation using explicit normal-case
criteria and pooled expert judgments, then test fact and attention components in
a factorial design. External patient-disjoint validation remains a separate
future study and is not claimed here.
