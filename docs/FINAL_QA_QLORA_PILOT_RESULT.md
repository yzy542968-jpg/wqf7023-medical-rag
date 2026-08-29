# Final-QA QLoRA Pilot Result

## Status and boundary

This record reports a Train/Calibration development pilot. The 477 training
cases and 358 Calibration cases were case-ID disjoint and duplicate-cluster
disjoint. Validation and Test were not accessed. The results measure agreement
with report-derived Rad-ReStruct answer annotations, not physician-adjudicated
diagnostic accuracy.

The dataset builder produced 600 deterministically selected Train questions
and three supervised conditions per question: no history, deterministic random
history, and relevant question-conditioned historical facts. The target report,
gold prior-answer history, and historical gold answer vectors were not exposed
in prompts. The resulting 1,800-row file passed the leakage assertions in the
committed dataset summary.

## Training audit

MedGemma 1.5 4B remained quantized and frozen. Only 2,228,224 q/v LoRA
parameters were trainable (0.0894% of 2,492,451,184 total parameters). The
pilot completed 192 forward steps and 12 optimizer steps, with no sequence
skips. Peak allocated GPU memory was 7,859 MiB. Mean loss over the final 100
training rows was 0.2116; mean loss over 24 examples from a case-disjoint
internal monitoring partition was 0.3172. Training loss alone is not treated as
evidence of downstream improvement.

## Paired Calibration results

The same 256 Calibration rows, target images, prompts, parser, and historical
evidence were used for the frozen base model and the adapter.

| Condition | Model | Option micro-F1 | Exact answer-set accuracy | Contract validity |
| --- | --- | ---: | ---: | ---: |
| Target image + indication, no history | Base | 0.1254 | 0.0938 | 0.9961 |
| Target image + indication, no history | QLoRA | **0.7090** | **0.7031** | **1.0000** |
| Top-3 image-neighbour question-conditioned facts | Base | 0.1570 | 0.1094 | 0.9102 |
| Top-3 image-neighbour question-conditioned facts | QLoRA | **0.7290** | **0.7344** | **0.9961** |

The adapter-minus-base option micro-F1 difference was +0.5835 without history
(10,000-case-bootstrap 95% CI [+0.5084, +0.6571]) and +0.5720 with relevant
history (95% CI [+0.4978, +0.6457]). The prespecified pilot promotion rule was
therefore met.

## Interpretation audit

The large adapter effect demonstrates that the zero-shot generator and strict
answer contract were major bottlenecks. It does not establish the RAG claim by
itself. Within the adapted model, relevant-history minus no-history was only
+0.0200 option micro-F1 and +0.0313 exact accuracy. The case-bootstrap 95% CI
for the micro-F1 difference was [-0.0157, +0.0558], and the exact-accuracy CI
was [-0.0039, +0.0664]. Historical evidence therefore produced a positive
point estimate but no statistically supported benefit in this pilot.

Eight of the 180 no-history-correct rows became incorrect with relevant
history, giving a 4.44% negative-transfer rate among that denominator.
Provenance completeness remained 100% because citations were assembled
deterministically from supplied evidence units.

The adapter is promoted as a development candidate. It is not yet the final
model. Subsequent work must determine whether additional Train-only adaptation
improves held-out performance and whether a better history-retrieval/filtering
policy can improve QA over the adapted no-history baseline. Test remains
prohibited until a separate confirmation protocol and exact configuration are
committed.
