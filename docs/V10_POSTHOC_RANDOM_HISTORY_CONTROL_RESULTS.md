# V10 Post-hoc Random-History Control Results

## Status

This exploratory negative control was executed after the V10 freeze under the
separately committed protocol at `183c5e8`. It did not alter any frozen V10
retrieval row, answer, configuration, or primary metric.

The analysis generated five deterministic random-history assignments for each
of 568 technically eligible Test cases and two question types. The resulting
5,680 rows used the same target image, indication, question, MedGemma revision,
three-report context size, 64-token answer budget, parsing, and provenance
assembly as G2. Random reports came only from the Train bank and excluded the
R4 and R5 Top-10 cases for that query.

## Generation integrity

| Item | Result |
|---|---:|
| Test cases | 568 |
| Questions per case | 2 |
| Random assignments | 5 |
| Generated rows | 5,680 |
| Train candidate bank | 2,506 cases |
| Answer-contract validity | 100% |
| Citation validity | 100% |
| Mean random-history input tokens | 783.63 |
| Mean random-history output tokens | 62.57 |
| Random-history token-ceiling rate | 84.37% |

All assignments were retained. Their Token-F1 values ranged from `0.17886` to
`0.18446`; no assignment was selected after outcome inspection.

## Main comparison

The primary comparison was frozen selected history (G2) minus the mean of the
five random-history assignments (GR). Assignment values were averaged within
each case/question before 10,000 case-grouped bootstrap resamples.

| Metric | G0 no history | GR random history | G2 selected history | G2 - GR | 95% CI |
|---|---:|---:|---:|---:|---:|
| Token-F1 | 0.14942 | 0.18278 | 0.20919 | **+0.02641** | **[+0.01920, +0.03388]** |
| F1RadGraph entity | 0.11336 | 0.12159 | 0.14969 | **+0.02810** | **[+0.01971, +0.03688]** |
| F1RadGraph entity-relation | 0.10282 | 0.10862 | 0.13392 | **+0.02531** | **[+0.01730, +0.03337]** |
| F1RadGraph complete | 0.08265 | 0.08747 | 0.11053 | **+0.02306** | **[+0.01569, +0.03040]** |
| F1CheXbert micro F1-14 | 0.54068 | 0.40744 | 0.54071 | **+0.13327** | **[+0.10955, +0.15709]** |
| F1CheXbert macro F1-14 | 0.24974 | 0.18640 | 0.27118 | **+0.08478** | **[+0.05311, +0.11364]** |
| F1CheXbert micro F1-5 | 0.33798 | 0.24195 | 0.35955 | **+0.11760** | **[+0.06513, +0.17073]** |
| F1CheXbert exact set-5 | 0.74384 | 0.68662 | 0.76056 | **+0.07394** | **[+0.05088, +0.09613]** |

The G2-minus-GR interval was above zero for the lexical metric, all three
RadGraph metrics, and every reported aggregate F1CheXbert metric. The same
direction held separately for Findings and Impression questions. For example,
complete F1RadGraph improved by `+0.02138` for Findings and `+0.02474` for
Impression, with both intervals above zero.

## What random history explains

Random history increased Token-F1 over no history by `+0.03337` (95% CI
`[+0.02747, +0.03900]`). Its complete F1RadGraph difference was only
`+0.00482` and the interval crossed zero (`[-0.00090, +0.01039]`). More
importantly, random history reduced F1CheXbert micro F1-14 by `-0.13324`
(`[-0.15670, -0.10934]`) and micro F1-5 by `-0.09603`
(`[-0.14267, -0.05109]`) relative to G0.

This separates two effects:

1. Generic radiology-report context can increase lexical overlap and may
   provide answer-style priors.
2. The selected historical cases provide an additional alignment-specific
   contribution that random reports do not reproduce, particularly on graph
   and pathology-label consistency.

G2 did not establish a global F1CheXbert advantage over G0. Its micro F1-14
difference was effectively zero (`+0.00002`, 95% CI
`[-0.02523, +0.02560]`). The defensible interpretation is therefore not that
historical retrieval improves every pathology metric over image-only
generation. Rather, relevant retrieval avoids the large pathology-consistency
degradation caused by arbitrary history while improving lexical and RadGraph
consistency over both controls.

## Claim boundary

These are automated, same-source, post-hoc measurements. F1CheXbert uses
report-derived labels and F1RadGraph uses automated graph overlap. The result
supports an alignment-specific contribution from selected historical context
under the frozen generator, but does not establish diagnostic accuracy,
physician-rated relevance, clinical utility, safety, patient benefit, external
validity, or patient-level independence.

## Reproducibility

- Random row SHA-256:
  `d2590dfd51a3a983da1d8ee9f71ff2c66a6417368922415bc3f357d202da6631`
- Selection fingerprint:
  `96b490b5a9038f2e2e4b9360cec7db0b099caecaa2e57d8b6ec53b166ac4fd9c`
- Random RadGraph row SHA-256:
  `1615a29ab073233241ff694fea75ede9d009c9cdee5611b509e3a071223ad247`
- F1CheXbert checkpoint SHA-256:
  `6550703c92d640e1e04d8105a7a185d76ece0f25fcbf033d292785bf22c0fde1`
- Machine-readable summary:
  `data/splits/v10/v10_random_history_control_summary.json`

