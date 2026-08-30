# Final-QA Full Validation Decision Record

## Decision status

The prespecified Final-QA Validation advancement rule **did not pass**. The
Final-QA Test role remains uninstantiated and unaccessed. No Test confirmation
protocol or Test generation config is created from this result.

This record closes the full Validation run defined in
`FINAL_QA_VALIDATION_PROTOCOL.md`. It does not modify any V10/V11 frozen model,
split, output or result. It also does not constitute clinical validation.

## Evaluation frame

- Role: V10-mapped Rad-ReStruct Validation only.
- Cases: 358.
- Questions: 17,864 per condition.
- Conditions: four, giving 71,456 generated rows.
- Structured answer space: 2,470 hierarchy-cleaned labels; 837 labels had
  positive Validation support.
- Generator: MedGemma 1.5 4B with the independently trained 384-forward-step
  q/v QLoRA adapter.
- Decoding: frozen greedy 32-token contract.
- Historical bank: eligible V10 Train cases, with the target duplicate cluster
  excluded.
- Bootstrap: 10,000 paired case-level replicates, seed 7023.

The complete rows contained 71,456 unique run keys and no duplicate key. Each
condition contained exactly 17,864 rows and all 358 Validation cases. Contract
invalidity was retained rather than filtered: B3 0 rows, B4 3, B6 2 and P1 7.

## Conditions

| ID | Condition | Selection role |
| --- | --- | --- |
| B0 | Train-label majority vector | Non-generative baseline |
| B3 | Target image, indication, question and options; no history | Primary no-history comparator |
| B4 | B3 plus one deterministic random other-cluster Train report | Mandatory non-relevant-context control; never selectable |
| B6 | B3 plus the Top-1 MedSigLIP image-neighbour report | Meaningful-history candidate |
| P1 | B3 plus question-conditioned evidence from Top-3 image neighbours | Meaningful-history candidate |

## Primary and secondary results

| System | Supported-label macro-F1 | Question option micro-F1 | Exact answer set | Official-compatible F1 | Root-question F1 | Exact report vector |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 majority | 0.286094 | n/a | n/a | 0.289107 | 0.603291 | 0.382682 |
| B3 no history | **0.309839** | 0.848367 | 0.849698 | **0.312726** | **0.622725** | 0.379888 |
| B4 random history | 0.305652 | 0.876241 | 0.878359 | 0.308565 | 0.617025 | **0.382682** |
| B6 Top-1 image history | 0.293340 | **0.877200** | **0.878974** | 0.296305 | 0.610036 | 0.374302 |
| P1 question-conditioned history | 0.292260 | 0.860918 | 0.863245 | 0.295215 | 0.604149 | 0.346369 |

The apparently high structured micro-F1 values (B3 0.975231; B6 0.975645;
P1 0.972978) and element accuracies above 0.994 are not used as the primary
claim because the 2,470-dimensional report space is sparse and dominated by
negative elements. Supported-label macro-F1 gives rare supported labels equal
weight and was frozen as the primary metric before generation.

Question-level accuracy and report-level reconstruction answer different
questions. B6 achieved 87.90% exact answer-set accuracy and 90.14% single-choice
accuracy, showing that the fine-tuned QA generator is technically effective.
However, its errors were distributed across labels such that complete
case-vector macro-F1 was lower than B3. High row-level accuracy therefore does
not establish a beneficial historical-retrieval effect.

## Paired uncertainty and negative transfer

| Comparison | Macro-F1 difference vs B3 | 95% paired case-bootstrap interval | Probability difference > 0 | Negative transfer among B3-correct rows |
| --- | ---: | ---: | ---: | ---: |
| B4 - B3 | -0.004187 | [-0.009053, +0.002223] | 0.1199 | 1.11% (169/15,179) |
| B6 - B3 | -0.016498 | **[-0.020490, -0.002348]** | 0.0030 | 1.31% (199/15,179) |
| P1 - B3 | -0.017579 | **[-0.021722, -0.003052]** | 0.0022 | 2.65% (403/15,179) |

B6 and P1 both had intervals fully below zero on the prespecified primary
metric. This is evidence of Validation-set degradation relative to B3 under the
frozen implementation, not merely absence of evidence for improvement.

## Frozen selection decision

B6 had the higher supported-label macro-F1 of the two meaningful-history
candidates and is therefore the mechanically selected candidate under the
B6-versus-P1 tie/selection rule. It nevertheless fails the advancement rule:

1. B6 supported-label macro-F1 (0.293340) is below B3 (0.309839).
2. B6 option micro-F1 (0.877200) is above B3 (0.848367).
3. B6 contract validity (0.999888) is within 0.010 of B3 (1.000000).

All three criteria were required. The first criterion failed. Consequently B6
is **not** promoted to a Final-QA Test system.

The random-history result further limits interpretation. B4 exceeded B6 on the
primary macro-F1 (0.305652 versus 0.293340), while B6 only marginally exceeded
B4 on question option micro-F1. Relevant-history conditions had shown greater
official-question concordance during the earlier audit, yet that relevance did
not translate into superior full Validation QA. Generic context or formatting
effects remain plausible, and a relevance-specific QA benefit is not
established.

## Runtime and provenance

- One uninterrupted four-condition invocation: 36,951.24 seconds (10.26 h).
- Aggregate generation time: 0.5171 seconds per question-condition row.
- Peak allocated VRAM: 4,380.52 MiB.
- Mean input tokens: B3 412.14, B4 483.46, B6 485.43, P1 551.71.
- Mean evidence units: B4 1.88, B6 1.90, P1 5.98.
- Provenance completeness: 100% in all generated conditions.

Condition-specific latency was not prospectively instrumented and is therefore
not reconstructed after the fact. The reported timing covers the complete
uninterrupted invocation only.

## Interpretation boundary

### Supported conclusions

- The 384-step QLoRA model achieved strong structured question answering on
  full Validation: 84.97-87.90% exact answer-set accuracy across conditions and
  87.12-90.14% single-choice accuracy.
- Adding context changed QA output, but the effect was not relevance-specific:
  deterministic random history was competitive with or better than meaningful
  history on the primary report-level metric.
- Under the frozen retrievers and prompt, image-neighbour and
  question-conditioned historical evidence degraded supported-label macro-F1
  relative to image-only B3.
- Better question-level averages do not guarantee better reconstruction of the
  complete structured report vector.

### Unsupported conclusions

- Historical RAG improves Final-QA report reconstruction.
- Relevant history is better than random history for this generator.
- B6 or P1 should be confirmed on Test.
- The reported automated metrics establish diagnostic accuracy, clinical
  usefulness or safety.

## Consequence for the thesis

This Final-QA extension should be reported as a full, protocol-governed
Validation study with a strong no-history fine-tuned QA result and a negative
historical-RAG result. It should not replace the frozen V10 primary study and it
should not be presented as a successful RAG confirmation. The negative result
is methodologically useful: retrieved history must be shown to improve the
target task beyond generic-context effects, and that condition was not met.

The Test role remains a protected holdout for a future, genuinely new method
developed without further reuse of these Validation outcomes. Independent
radiologist review and external patient-level validation remain Future Work.

## Audit trail

- Protocol commit before Validation generation: `5516d79`.
- Runtime-config correction before Validation generation: `b693fd3`.
- Complete rows SHA-256:
  `bc040f296d4e77e6cec7b52919ccb63d56f5ea36d8684959feb9b8a2f16addf2`.
- Validation config SHA-256:
  `af081dc0ce1118c43856156dcbdf0b5b8c0ec3b079ea7eddf19277d2264ab0ec`.
- Compact generation summary SHA-256:
  `904ba5efdf1cdb11a7105480fbc57f8114725c43bb12ac05df1d47d61cd01066`.
- Full Validation evaluation SHA-256:
  `f12eb12f51c8fdbc5d6ecf43d62f2cca8fd81bf17c2dcfef984091c3e5e1eaf4`.
- Evaluator implementation SHA-256:
  `4640de03f922b5ed02e08f8d7af0c0ea95fcd55c8c3acbf745416f7130f43762`.
- The large 39 MB per-question rows remain local under repository policy.
- The compact generation summary and evaluation JSON are committed with this
  record.

The generation summary's legacy Calibration status label was corrected to the
Validation-specific status after completion. This metadata-only correction did
not alter rows, predictions, metrics, split membership or runtime values. The
runner was also corrected so future full-Validation summaries emit the proper
status automatically.
