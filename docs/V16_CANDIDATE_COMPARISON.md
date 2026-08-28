# V16 Candidate Comparison

**Scope:** exploratory generation adaptation on the V16 Validation partition
**No Test evaluation:** all results in this document are Validation-only
**Primary metric:** Token-F1 against the frozen report reference
**Metric boundary:** Token-F1, CheXbert, and RadGraph are automated
report-reference consistency measures; they are not diagnostic accuracy,
clinical correctness, clinical safety, physician utility, or external
validation.

## 1. Evaluation frame

The main comparison uses the deterministic 376-case Validation manifest. It
contains 752 rows per generation arm across three conditions and two question
types: 376 findings rows and 376 impression rows. The base and balanced QLoRA
arms use the same cases, questions, retrieved evidence, random-history
controls, decoding policy, and frozen ranking rows.

The balanced QLoRA candidate was trained on a separate 300-case development
source. The complete Validation comparison was run only after generation
policy and the Validation manifest had been fixed. The resulting route was
constructed from these already generated Validation rows and is explicitly
post-hoc exploratory.

## 2. Candidate screening

| Candidate | Validation evidence | Decision |
| --- | --- | --- |
| Frozen base, 96-token/2-sentence policy | Stable reference arm; retrieved Token-F1 `0.202410` on 376 cases | Retain as baseline |
| QLoRA qv, 300 cases, retrieved-only training | Positive signal on its separate development/screening source, but not directly comparable with the 376-case result | Do not use as primary comparison |
| QLoRA qv, balanced 300 cases, 256 steps | Retrieved Token-F1 `0.244502` on 376 cases; improves over base by `+0.042092`, 95% CI `[+0.028199, +0.056376]` | Retain as global adaptation candidate |
| QLoRA qv, balanced 1000 cases, 1024 steps | 12-case retrieved Token-F1 `0.220084`, below the 300-case candidate `0.336516` on the same screening matrix | Do not promote |
| Base 160-token/4-sentence policy | No stable retrieved improvement and a higher output-ceiling rate | Do not promote |
| Findings-only qv adapter, 300 cases, 256 steps | 12-case retrieved findings Token-F1 `0.283623` vs base `0.332819` | Do not promote |
| Base findings + balanced QLoRA impression route | 376-case retrieved Token-F1 `0.259585`; post-hoc exploratory route | Retain only as exploratory candidate |
| Retrieved-history impression gate | Same `+0.057174` retrieved Token-F1 difference while all controls fall back to base | Retain as deployment-oriented exploratory candidate |
| Extractive top-1 report-copy diagnostic | Token-F1 `0.401548` on 86 valid rows; not a generative system and uses target-reference copying | Diagnostic upper-bound-style comparator only |

The extractive diagnostic must not be presented as a fair end-to-end model
result. It copies the selected target-associated report section when that
reference is available and therefore answers a different task from generating
an answer from historical evidence.

## 3. Complete 376-case comparison

### 3.1 Token-F1 and output ceiling

| Generator | Condition | Token-F1 | Token-ceiling rate |
| --- | --- | ---: | ---: |
| Base | no history | `0.162143` | `39.10%` |
| Balanced QLoRA | no history | `0.148228` | `3.72%` |
| Base | retrieved history | `0.202410` | `37.10%` |
| Balanced QLoRA | retrieved history | `0.244502` | `2.26%` |
| Base | random history | `0.186629` | `33.38%` |
| Balanced QLoRA | random history | `0.206647` | `3.59%` |
| Routed | no history | `0.157809` | `23.94%` |
| Routed | retrieved history | `0.259585` | `22.47%` |
| Routed | random history | `0.220667` | `18.75%` |

The balanced adapter has a clear retrieved-history Token-F1 gain and a large
reduction in output-ceiling hits. It also has a no-history regression. The
random-history result is not a deployment target and must remain a negative
control; an improvement under random history does not establish useful
evidence use.

### 3.2 Question-type asymmetry

For the 376-case comparison:

| Generator | Condition | Findings Token-F1 | Impression Token-F1 |
| --- | --- | ---: | ---: |
| Base | retrieved history | `0.265273` | `0.139548` |
| Balanced QLoRA | retrieved history | `0.235108` | `0.253896` |
| Routed | retrieved history | `0.265273` | `0.253896` |

The route therefore preserves the base findings score while retaining the
balanced adapter's impression score. This explains its higher aggregate
retrieved-history score, but the choice was made after observing the Validation
breakdown and is consequently exploratory.

## 4. Paired differences

The balanced QLoRA retrieved-history arm versus the base retrieved-history arm
showed:

| Metric | Difference | 95% bootstrap CI | Interpretation |
| --- | ---: | --- | --- |
| Token-F1 | `+0.042092` | `[+0.028199, +0.056376]` | positive automated overlap change |
| RadGraph entity F1 | `+0.011654` | `[-0.002086, +0.025838]` | not statistically resolved |
| RadGraph relation F1 | `+0.014450` | `[+0.001676, +0.027630]` | positive automated relation overlap |
| RadGraph complete F1 | `+0.020706` | `[+0.007999, +0.033855]` | positive automated complete overlap |
| CheXbert micro-F1 | `+0.009391` | CI crosses zero | not statistically resolved |
| CheXbert macro-F1 | `+0.031369` | CI crosses zero | not statistically resolved |
| CheXbert exact-5 | `+0.022606` | `[+0.007979, +0.038564]` | positive automated label agreement |

The post-hoc routed retrieved-history arm versus the base arm showed Token-F1
difference `+0.057174`, 95% CI `[+0.043791, +0.071026]`. Its no-history
difference was `-0.004333`, 95% CI `[-0.007699, -0.001058]`, and its
random-history difference was `+0.034038`, 95% CI `[+0.022524, +0.045957]`.
These control results are why the route is not promoted to a globally superior
generator. Its strongest evidence is conditional on retrieved history and
should be described as a retrieval-conditioned exploratory candidate.

The retrieved-history impression gate uses the base rows for no-history and
random-history controls. Its full Validation Token-F1 values are therefore
`0.162143` (no history), `0.259585` (retrieved history), and `0.186629`
(random history). Relative to the base, the paired differences are `0.000000`,
`+0.057174` (95% CI `[+0.043791, +0.071026]`), and `0.000000`, respectively.
This is a transparent conditional deployment policy, but it was selected after
observing Validation asymmetry and therefore remains post-hoc.

Compared with the global balanced QLoRA arm, the gate is `+0.015082` Token-F1
in retrieved history, with 95% CI `[+0.009294, +0.020907]`. This difference
comes from retaining the stronger base findings output; it is not an
independent test of a newly trained model.

## 5. Development decisions

1. Retain the balanced 300-case qv adapter as the principal V16 adaptation
   candidate for retrieval-conditioned engineering analysis.
2. Do not promote the 1000-case/1024-step candidate, the longer base output
   policy, or the findings-only adapter.
3. Retain the question-type route as a transparent exploratory candidate only:
   frozen base for findings and balanced QLoRA for impression.
4. Retain the retrieved-history impression gate as the safer engineering form
   of that route for future work, while keeping it outside the confirmatory
   claim set.
5. Do not claim that QLoRA globally improves generation. The full comparison
   shows a positive retrieved-history effect but a no-history regression and
   control-condition complications.
6. Do not evaluate any of these candidates on the frozen Test partition in
   this exploratory phase.

## 6. Reproducibility artifacts

The comparison is supported by the following local artifacts:

- `data/splits/v16/v16_qv_balanced_300_256_r1_376_metrics.json`
- `data/splits/v16/v16_qv_balanced_300_256_r1_376_evaluation.json`
- `data/splits/v16/v16_qv_findings_only_300_256_r1_12_evaluation.json`
- `data/splits/v16/v16_qv_impression_route_vs_base_376.json`
- `data/splits/v16/v16_qv_impression_retrieved_gate_vs_base_376.json`
- `data/splits/v16/v16_qv_impression_retrieved_gate_vs_qv_balanced_376.json`
- `scripts/merge_v16_question_type_route.py`
- `scripts/evaluate_v16_paired_rows.py`

Large generation rows, adapter weights, image pixels, and local source data
remain subject to the repository's local-artifact policy and are not treated
as public raw-data releases.
