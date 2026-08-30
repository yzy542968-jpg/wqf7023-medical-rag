# Final-QA v2 Selective Paired-History Gate Result

## Decision

**STOP under the frozen advancement rule.**

The selected similarity gate improved question-level exact answer-set accuracy
and supported-label macro-F1, and it exceeded every shuffled-pair replicate.
However, its option micro-F1 change (`-0.001477`) crossed the frozen
non-inferiority margin (`-0.001`). The method therefore did not advance to
Test or to a deployable claim.

This is a useful mixed result rather than a system failure. Correct pairing
contains measurable information, but a similarity-only rule does not separate
helpful history from harmful history reliably enough across all answer options.

## Boundaries

- Final-QA Validation was reused as development data; it was not an untouched
  confirmation set.
- Final-QA Test was not generated, inspected or evaluated.
- No new MedGemma generation was run.
- Historical answer payloads were oracle development substitutes derived from
  paired historical reports. They are not available as direct labels in a
  deployable system.
- The experiment did not modify V10/V11 or Final-QA v1 artifacts.
- Outcomes are automated report-reference consistency measurements, not
  clinician-adjudicated diagnostic accuracy.

## Reproducible inputs

The pinned MedSigLIP text tower encoded 202 unique question texts on CUDA and
mapped them back to 17,864 Validation question instances. Encoding took about
58.1 seconds on the local RTX 5070 Laptop GPU.

```text
model: google/medsiglip-448
revision: 9cea28a1a1195f665105faa6e8544c112fd960a4
embedding dimension: 1152
question-text SHA-256:
6ff15e0ec9eb3bb89a5148100c41dbd8466d48042321db2af32a71195517add8
```

Of 2,351 Final-QA Train cases, 2,348 had both the pinned image and report
embedding required by the paired-history candidate frame. Three missing-pair
cases were excluded before candidate construction.

## Fixed development partitions

```text
Validation development frame        358 cases
Selection partition                 170 cases
  gate training                      84 cases
  gate calibration                   86 cases
Development holdout                 188 cases

MLP internal training                46 cases
MLP internal early stopping          38 cases
```

All partitions were deterministic and case-level. The 188-case holdout had
already been used by the preceding v2 feasibility study and must therefore be
described as a development holdout, not as unseen confirmation.

## Candidate selection

The MCR-lite candidate selector used six frozen features over each image-based
Top-20 shortlist. Its candidate training frame contained 80,680 rows, of which
31,391 were exact answer matches for the corresponding question.

On gate calibration, image Top-1 was materially better than the learned
logistic candidate selector:

| Candidate policy | Question exact | Option micro-F1 | Macro-F1 |
|---|---:|---:|---:|
| Image Top-1 | 0.87152 | 0.89767 | 0.50401 |
| Logistic MCR-lite | 0.85144 | 0.87660 | 0.47065 |

The fixed selector rule therefore retained image Top-1. This negative result
is informative: the tested linear combination of cross-modal similarities did
not improve answer-relevant case selection. In particular, high report-answer
agreement in this structured dataset can be driven by common negative answers,
so candidate labels require stronger rarity or clinical-concept weighting in a
future method.

## Gate calibration

The three source-gate families produced the following calibration results:

| Gate | Frozen threshold | Question exact | Option micro-F1 | Macro-F1 |
|---|---:|---:|---:|---:|
| Image-similarity threshold | 0.85282 | 0.87322 | 0.89952 | **0.51394** |
| Logistic regression | 0.44 | **0.87346** | **0.90217** | 0.49588 |
| Two-layer MLP | 0.29 | 0.87297 | 0.89991 | 0.50954 |

Logistic regression was only `0.000242` higher in exact accuracy than the
similarity threshold. This is below the protocol's `0.0005` tie tolerance, so
the simpler similarity gate had to be selected. Selecting logistic regression
after seeing the holdout would violate the frozen rule.

## Development-holdout result

The fixed 188-case holdout contained 9,368 structured questions.

| Metric | Image only | Selected gate | Difference |
|---|---:|---:|---:|
| Question exact answer-set accuracy | 0.85344 | 0.85397 | **+0.00053** |
| Option micro-F1 | 0.88578 | 0.88430 | **-0.00148** |
| Supported-label macro-F1 | 0.36823 | 0.38784 | **+0.01961** |
| Structured micro-F1 | 0.97555 | 0.97642 | +0.00088 |
| Exact report-vector accuracy | 0.38298 | 0.26064 | -0.12234 |

The gate used history for 356 disagreements (`3.80%` of holdout questions).
Before hierarchy-consistency side effects, this recovered 166 image-wrong,
history-correct questions but replaced 161 image-correct answers with an
incorrect historical answer. The resulting net was only five questions at
that local decision level.

The question-weighted exact improvement was positive, but the case-averaged
bootstrap estimate was negative (`-0.00421`) with a 95% interval of
`[-0.00948, +0.00140]`. Consequently, the experiment does not support a claim
of case-level superiority.

The lower exact report-vector accuracy is also important. Small per-question
changes can trigger hierarchy cleaning and make it harder to reproduce an
entire 2,470-label vector exactly, even while rare-label macro-F1 improves. No
single metric should therefore be used as a complete definition of success.

## Pairing control

The selected aligned system exceeded all 20 fixed-point-free report-pair
shuffles:

```text
aligned exact accuracy             0.85397
shuffled exact mean                0.84215
shuffled exact range       0.83711–0.84639
aligned minus shuffled mean        0.01182
plus-one Monte Carlo p             0.04762
shuffled macro-F1 mean             0.36658
```

This supports the narrow statement that correct image-report ownership matters
for the tested historical signal. It does not show that the selected gate is
better than image-only under every metric.

## Frozen advancement checks

| Check | Result |
|---|---|
| Exact accuracy exceeds image-only | Pass |
| Option micro-F1 delta >= -0.001 | **Fail** |
| Macro-F1 delta >= -0.005 | Pass |
| History used on at least one disagreement | Pass |
| Aligned exceeds every shuffled counterpart | Pass |

Because one required condition failed, the overall decision is STOP.

## Research interpretation

The experiment establishes three useful development findings:

1. Correctly paired historical reports carry answer-relevant signal beyond
   arbitrary report ownership.
2. Unconditional fusion is unnecessary, but a one-dimensional similarity gate
   remains too coarse: its rare-label gains came with a small common-option
   penalty.
3. The simple logistic MCR-lite candidate selector was worse than image Top-1,
   so the next candidate model should not merely add more cosine features. It
   should use question-specific clinical facts, rarity-aware supervision or a
   pairwise objective that distinguishes informative positives from common
   normal-answer matches.

Any follow-up is post-hoc development informed by these outcomes. A sensible
next pilot would derive concise facts from the retrieved report, preserve the
image-only answer by default, and allow history to contribute only a
question-relevant supported fact. Such a pilot must remain on Train/development
data and must not unlock Final-QA Test without a newly frozen deployable method
and confirmation protocol.

## Reproduction

```powershell
python scripts/cache_final_qa_v2_question_embeddings.py `
  --radrestruct-root <RADRESTRUCT_ROOT> --local-files-only

python scripts/develop_final_qa_v2_selective_gate.py `
  --radrestruct-root <RADRESTRUCT_ROOT>
```

Primary machine-readable outputs:

- `experiments/final_qa_development/final_qa_v2_question_embeddings.json`
- `experiments/final_qa_development/final_qa_v2_selective_gate.json`

The compressed question-embedding cache is local and reproducible from the
pinned model, revision and question-text fingerprint.
