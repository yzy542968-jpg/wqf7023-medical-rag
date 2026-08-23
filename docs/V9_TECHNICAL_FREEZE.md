# V9 Technical Freeze

## Freeze status

V9 model development, retrieval confirmation, multimodal QA confirmation,
bounded-agent evaluation, statistics, deterministic qualitative extraction,
and the interactive demonstration are technically complete. V5-V8 remain
unchanged. No V9 outcome triggered model, prompt, threshold, case, or metric
retuning.

The only unfinished research action is the student's review of the 24-case
qualitative pack. Independent clinical human evaluation remains Future Work.

## Final task

V9 models a new patient whose formal report is unavailable to the system. It
uses one target chest radiograph, pre-report clinical indication, and medical
question to retrieve similar image-report cases from a fixed other-patient
historical bank, then uses those reports as explicitly labeled analogies for
multimodal question answering.

The target report is hidden at inference and used only for offline relevance
and QA scoring. This is distinct from earlier closed-set work that attempted
to retrieve the target patient's own report.

## Data and models

```text
OpenI source cases                         3,851
Primary stratifiable cases                3,759
Train / Validation / Test        2,631 / 376 / 752
Report-bearing historical bank            2,608
Complete-reference QA Test cases             685
QA questions                               1,370
QA generation rows                         5,480
```

Frozen foundation components are MedSigLIP-448, MedGemma 1.5 4B, modern
RadGraph XL, and BioLinkBERT-MedNLI. The project-trained V9 component is an
865-parameter `9 -> 32 -> 16 -> 1` MLP reranker trained with 307,176 weighted
pairwise examples. Foundation-model parameters were not updated.

## Retrieval result

| System | nDCG@10 | MRR |
|---|---:|---:|
| BM25 | 0.134156 | 0.083542 |
| Image-image | 0.315561 | 0.328270 |
| Image-report | 0.274069 | 0.256032 |
| Fixed multimodal | 0.246935 | 0.211322 |
| **Learned MLP** | **0.327942** | **0.331968** |

The primary learned-minus-image nDCG@10 difference was `+0.012381`, 95% CI
`[+0.009226,+0.015584]`. Aligned learned retrieval exceeded all 100 shuffled
image controls (`p=0.009901`).

## QA and agent result

| System | Token-F1 |
|---|---:|
| G0 target image, no retrieval | 0.145559 |
| G1 BM25 RAG | 0.147947 |
| G2 fixed multimodal RAG | 0.179090 |
| **G3 learned multimodal RAG** | **0.184803** |

G3 minus G0 was `+0.039244`, 95% case-bootstrap CI
`[+0.032572,+0.045745]`. G3 minus G2 was only `+0.005713`, with a CI crossing
zero, so downstream superiority over fixed multimodal RAG is not claimed.

The G4 agent reduced automated unsupported historical-support rows from
`16.42%` to `0%` by one backup route or evidence-field abstention. The paired
difference CI was `[-18.47,-14.45]` percentage points. Target-answer Token-F1
was unchanged by design. This is claim control, not image-answer verification.

## Reproducibility hashes

```text
Retrieval rows:
baa56924928b144c9b877b8e2218e04d17df6b77a6f794ed3830f7ccf3e449fd

QA Top-3 ranking pack:
28639821abc5fba8189c7c0149822ed0e3935325d0136578803155cc5a4ebd9b

QA raw rows:
89c69c9a27e393c93c85e572587b330f908598e835cb8162a8678cd15ba512b4

Agent rows:
9cc8b4513f2ef12f7e849d7b5853a79ef07495b022699c5a84785d1d94624fc1

MLP checkpoint:
8afa68a48de9d6c9128d190f1368d0d45d41a958e5eb12787d7e725e7eb09efa
```

Large source-derived rows, generations, report text, image pixels, vectors,
and checkpoints remain local under repository policy. Aggregate summaries,
protocols, scripts, hashes, tests, and the lightweight qualitative index are
public.

## Final supported claim

> In a retrospective same-source OpenI study, correctly aligned chest-image
> information substantially improved report-derived similar-case retrieval;
> a small learned reranker added a reproducible retrieval gain over the
> strongest frozen image component; and retrieved multimodal historical
> evidence improved MedGemma report-reference consistency over the same model
> without retrieval. A bounded agent suppressed unsupported historical
> evidence statements, but did not verify target-image diagnoses.

V9 does not establish physician-adjudicated similarity, clinical diagnostic
accuracy, patient safety, external generalization, treatment utility, or
deployment readiness. OpenI's source design states one study per patient, but
released patient identifiers are unavailable for independent patient-level
verification.

## Post-freeze rule

No further V9 model development, prompt tuning, threshold tuning, data
reallocation, confirmation experiment, or outcome-driven case replacement is
permitted. Remaining work is researcher qualitative review and reporting:
manuscript, PDF, defence deck, README, and dashboard verification.
