# V12 Extended Pilot Notes

## Purpose

These notes record post-protocol exploratory extensions executed after the
initial V12 pilot. They do not amend `V12_PILOT_PROTOCOL.md`, the V10/V11
freeze, or any Test result. They are retained so that later reconstruction of
the project does not silently turn exploratory choices into confirmation
choices.

## Qwen3 dense-channel audit

The Qwen3-Embedding-0.6B document/query channel was evaluated with a fixed
instruction, 512-token truncation, last-non-padding-token pooling, L2
normalization, and signature-checked local caches. Eight weighted-RRF recipes
were selected on the Train internal role and evaluated on Validation. The
predeclared three-channel BM25 + MedCPT + MedSigLIP recipe was selected; adding
Qwen3 did not improve the Validation candidate ranking. Qwen3-only and
Qwen-heavy recipes were consistently worse. Qwen3 is therefore retained as a
negative result and is not part of the recommended stack.

## Learned ranking extensions

The primary exploratory ranker uses a Train fit/internal split, the existing
17-dimensional R5 feature representation, RRF Top-200 candidates, and
LightGBM LambdaMART. The deepest configuration was selected using only Train
internal qrel-v2 nDCG@10. On Validation it reached qrel-v2 nDCG@10 `0.62023`
versus R5 full-bank `0.55493` (paired case bootstrap difference `+0.06531`,
95% CI `[+0.05550, +0.07494]`).

Three safeguards limit the interpretation:

1. qrel-v2 is derived from the same report/fact universe used by several
   ranking features;
2. label-only and fact-only are sensitivity proxies, not clinical labels;
3. scoring the full Train bank with the Top-200-trained ranker fell to
   `0.54654`, so the improvement should be described as candidate-frame
   reranking rather than unrestricted full-bank retrieval.

Proxy-specific rankers were also fitted without changing the split or feature
pipeline. The label-only model reached `0.38317` on label-only nDCG@10 versus
R5 `0.33442`, with difference `+0.04876` and 95% CI
`[+0.03287, +0.06528]`. The fact-only model reached `0.37386` versus R5
`0.33278`, with difference `+0.04108` and 95% CI `[+0.02916, +0.05313]`.
The agreement across proxies is encouraging, but it remains same-source
automated evidence rather than independent validation.

## Generation policy extension

The 48-case Validation generation subset was unchanged. The router used only
question type:

```text
findings   -> whole_report
impression -> whole_report
acute      -> case_to_fact
```

The mixed policy achieved Token-F1 `0.21334` across all rows, versus `0.19303`
for fixed whole-report evidence and `0.18235` for fixed case-to-fact evidence.
The acute reference is proxy-derived, so the non-proxy mixed score is
`0.19160`; this is not evidence of clinical answer accuracy. The routing rule
was derived after observing the pilot and must be frozen before a future
confirmation evaluation.

## Implementation audit note

Two state builders are used in the pilot. The weighted-RRF training helper
stores a compact `feature_case_ids` view, while the original Validation
helper stores the complete R5 feature matrix in `runtime.candidate_ids` order.
The evaluator therefore uses runtime candidate indices for the original
Validation state. This distinction was checked after an attempted temporary
indexing change produced a KeyError; the temporary change was removed and the
correct state semantics were documented in code.

## Reproduction boundaries

- All runs use V10 Train/Validation only.
- No Test case, Test metric, or Test output was read by the new scripts.
- Cached embeddings are local artifacts with explicit SHA-256 signatures.
- Large row files, model files, and embedding caches remain local according
  to repository policy.
- A future final study must define a new cohort, freeze the selected stack,
  and run a new confirmation evaluation before promoting any V12 result.
