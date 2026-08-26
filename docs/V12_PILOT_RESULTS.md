# V12 Optimization Pilot Results

## Status

V12 is an isolated, Validation-only development pilot on branch
`v12-optimization-pilot`. It does not amend the V10/V11 freeze, reopen Test,
or change any frozen model, prompt, split, result, manuscript, or dashboard
claim. The pilot uses report-derived relevance proxies; it is not clinical
validation.

The pilot was run on 376 technically eligible V10 Validation cases and a
2,506-case Train historical bank. The 48-case generation subset was selected
before generation by a deterministic SHA-256 rule within report-indexed normal
and abnormal strata. No case was replaced after output inspection.

The earlier Stage 0 candidate audit retained the broader 2,510-Train / 384-
Validation executable frame, whereas the learned-ranker pilot required
RadGraph-complete rows and therefore used 2,506 / 376. This integrity-based
frame difference is recorded explicitly; the two summaries are not treated as
identical-condition head-to-head evidence.

## Stage 0: bottleneck diagnostics

The saved candidate-generation audit showed that RRF was useful for candidate
coverage but not uniformly for all metrics:

| Candidate condition | nDCG@10 | Relevant presence | Relevant recall |
|---|---:|---:|---:|
| RRF Top-50 | 0.58668 | 48.18% | 7.41% |
| RRF Top-100 | 0.58668 | 57.99% | 11.82% |
| RRF Top-200 | 0.58668 | 65.71% | 19.11% |

The Top-200 pool had an oracle nDCG@10 of 0.86530 under qrel-v2, compared
with observed RRF nDCG@10 of 0.58668. This leaves ranking headroom, but the
oracle gap is based on the same report-derived proxy and should not be read as
clinical headroom.

The frozen V10 generation rows also showed a raw-versus-final Token-F1 gap,
which motivated the answer-only generation check. That diagnostic does not
prove that all raw content is correct or safely recoverable.

## Stage 1: learned retrieval pilot

The pilot trained a LightGBM 4.7.0 LambdaMART ranker on V10 Train role-based
fit/internal groups. It used the existing 17-dimensional R5 feature pipeline
over an RRF Top-200 candidate pool. Foundation encoders and the V10 R5 models
remained frozen. Evaluation used V10 Validation only.

| System | qrel-v2 nDCG@10 | Difference vs R5 | 95% CI |
|---|---:|---:|---:|
| R5 full bank | 0.55493 | reference | - |
| RRF candidate only | 0.54225 | -0.01268 | [-0.01979, -0.00553] |
| RRF + frozen R5 rerank | 0.55566 | +0.00073 | [+0.00003, +0.00143] |
| RRF + LambdaMART | **0.62023** | **+0.06531** | **[+0.05550, +0.07494]** |
| Full-bank LambdaMART | 0.54654 | -0.00838 | [-0.01957, +0.00278] |

The learned ranker therefore produced a strong exploratory improvement under
the training-aligned qrel-v2 proxy. It is not yet a confirmed new research
result because the training objective and primary evaluation construct share
report-derived components. The predeclared sensitivity results are essential:

| System | Label-only difference vs R5 | 95% CI | Fact-only difference vs R5 | 95% CI |
|---|---:|---:|---:|---:|
| RRF candidate only | -0.05206 | [-0.06804, -0.03613] | -0.03794 | [-0.04671, -0.02918] |
| RRF + frozen R5 rerank | +0.00132 | [-0.00027, +0.00299] | +0.00111 | [-0.00008, +0.00227] |
| RRF + LambdaMART | **+0.03017** | **[+0.01384, +0.04696]** | **+0.03139** | **[+0.01985, +0.04295]** |

These sensitivity gains are themselves report-derived constructs, and the
ranker uses RadGraph-derived features. The full-bank
application was worse than R5, indicating that the improvement is tied to the
RRF Top-200 candidate frame rather than a safe full-bank replacement.

The defensible conclusion is:

> LambdaMART is a promising retrieval-development direction with a large
> proxy-metric signal, but its superiority is not established independently of
> the report-derived feature/evaluation construct.

### Proxy-specific ranker sensitivity

To separate target-construct effects from model capacity, two additional
rankers were trained using the same Train fit/internal split and the same
preselected RRF Top-200 candidates, but with proxy-specific objectives. The
label-only ranker reached Validation nDCG@10 `0.38317` against the R5
label-only baseline `0.33442`, a difference of `+0.04876` (95% CI
`[+0.03287, +0.06528]`). The fact-only ranker reached `0.37386` against
`0.33278`, a difference of `+0.04108` (95% CI `[+0.02916, +0.05313]`).
These results strengthen the claim that the learned ranking signal is not
exclusive to qrel-v2, but they do not create independent clinical labels:
all three objectives are derived from the same source reports and RadGraph
annotations.

RRF alone is not promoted. The next retrieval step, if pursued, must use a
newly frozen independent relevance/confirmation design rather than simply
selecting the most favorable qrel result.

## Stage 2: generation pilot

The generation pilot used the same 48-case V12 manifest, three question roles,
the saved LambdaMART Top-3, MedGemma 1.5 4B, and two evidence policies. It
compared 96 and 128 new-token budgets. Each budget produced 288 rows.

| Policy | Token-F1 | Non-proxy Token-F1 | Input tokens | Evidence chars | Contract valid | Provenance valid | Ceiling rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Whole report | 0.19303 | 0.19160 | 747.6 | 593.1 | 100% | 100% | 1.39% |
| Case-to-fact | 0.18235 | 0.14511 | 532.4 | 235.4 | 100% | 100% | 1.39% |

Case-to-fact reduced mean input tokens by 28.8% and evidence characters by
60.3%, while preserving deterministic provenance validity. It did not improve
answer overlap:

```text
case-to-fact - whole-report Token-F1 = -0.01068
95% case-grouped bootstrap CI = [-0.03119, +0.00977]
```

The 96- and 128-token runs produced identical Token-F1 and paired differences.
The 96-token budget is retained only as a lower-latency engineering tie-break
for this pilot; it does not replace the frozen V10/V11 generation setting.

The generation conclusion is therefore narrower than “fact selection improves
QA”:

> Case-to-fact selection improves context compactness and auditability in this
> pilot, but answer-quality superiority is not demonstrated.

### Question-type evidence router

A deterministic router using only the question type selected whole-report
evidence for Findings and Impression questions and case-to-fact evidence for
the acute question. On the same 48-case Validation subset, this mixed policy
obtained Token-F1 `0.21334`, compared with `0.19303` for fixed whole-report
evidence and `0.18235` for fixed case-to-fact evidence. The fixed-policy
comparison is now reported across all three question types; earlier zero-row
subgroup summaries were a presentation bug and have been corrected.

This result is exploratory and post-hoc: the router rule was derived after
observing the pilot rows. The acute reference is a source-derived proxy, so
the non-proxy router score is `0.19160` and the apparent gain cannot be
presented as validated clinical QA improvement. The rule is useful as a
candidate policy for a future protocol, not as confirmation evidence.

## Decision

The combined pilot decision is:

```text
CONTINUE_RETRIEVAL_DEVELOPMENT
STOP_GENERATION_OPTIMIZATION_AT_CURRENT_SCOPE
```

Retrieval has enough exploratory signal to justify one separately frozen
follow-up study, but not enough independent evidence to promote LambdaMART to
the thesis primary system. The best current development stack is therefore
RRF Top-200 candidate generation followed by a Train-selected LambdaMART
reranker, with proxy-specific variants retained only for sensitivity analysis.
Generation does not justify more broad prompt or budget search under the
current 48-case automated pilot; the question-type router may be carried into
a separately frozen follow-up because it improved the development score, but
it must be selected before a new confirmation evaluation.

No Test evaluation was performed. No V10/V11 output was overwritten. No
clinical, physician-adjudicated, external-validity, or deployment-safety claim
is supported by V12.

## Reproducibility artifacts

- Protocol: `docs/V12_PILOT_PROTOCOL.md`
- Diagnostics: `experiments/v12_optimization/diagnostics/`
- Retrieval summary: `experiments/v12_optimization/retrieval/v12_retrieval_pilot.json`
- Retrieval sensitivity: `experiments/v12_optimization/retrieval/v12_validation_rankings.json`
- Qwen3 and weighted-RRF audit: `experiments/v12_optimization/retrieval/v12_qwen3_weighted_rrf.json`
- Proxy-specific validation audits: `experiments/v12_optimization/retrieval/v12_label_only_validation_rankings.json`, `v12_fact_only_validation_rankings.json`
- Generation summaries: `experiments/v12_optimization/generation/v12_generation_96_summary.json`, `v12_generation_128_summary.json`
- Generation budget comparison: `experiments/v12_optimization/generation/v12_generation_budget_comparison.json`
- Question router audit: `experiments/v12_optimization/generation/v12_question_router_analysis.json`
- Case manifest: `experiments/v12_optimization/generation/v12_generation_manifest.json`

Large row files and local model/data artifacts remain subject to the repository
release policy and are not required to be published as public raw data.
