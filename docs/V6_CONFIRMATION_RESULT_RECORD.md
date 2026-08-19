# V6 Model-Modernized Confirmation Result Record

## 1. Status and purpose

The V6 model-modernized confirmation study is complete. Its purpose was to test whether the central V5 finding, namely an alignment-specific contribution from correctly paired radiology images, remained reproducible with newer models, a broader within-source case spectrum, and a newly instantiated confirmation cohort.

This record summarizes frozen automated outcomes. It does not constitute physician adjudication, external validation, a clinical safety assessment, or evidence of deployment utility.

Frozen lineage:

- Development protocol: `b85db42`
- Development decisions: `ec01bee`
- Confirmation protocol: `eee7405`
- Instantiated cohort: `43fe1a0`
- Retrieval implementation: `268111a`
- Retrieval outcomes: `c6442c9`
- QA implementation: `8ce4db0`
- Raw QA outcomes: `5aa8a6b`
- Verifier implementation: `c1d4c6c`
- Verified QA outcomes: `3ae127f`
- Statistical implementation: `39b09bb`
- Statistical outcomes: `9258e2f`

## 2. Frozen confirmation design

The confirmation candidate pool contained 240 case IDs from the IU-Xray/OpenI source. It contained 172 report-indexed normal cases and 68 report-indexed abnormal cases. The target and distractor roles were balanced, with 86 normal and 34 abnormal cases in each role. The 120 target cases produced 360 deterministic report-derived questions.

The evaluation was case-ID disjoint from the enumerated development set. Reliable patient identifiers were unavailable in the processed source, so patient-level independence could not be verified.

The primary retrieval contrast was:

```text
BM25 indication + question retrieval
vs
BM25 shortlist + correctly paired MedSigLIP image reranking
```

The downstream QA factorial was:

```text
BM25 retrieval      -> Qwen2.5-1.5B-Instruct
MedSigLIP reranking -> Qwen2.5-1.5B-Instruct
BM25 retrieval      -> MedGemma 1.5 4B
MedSigLIP reranking -> MedGemma 1.5 4B
```

Image pixels affected report retrieval only. Both generators received the same semantic prompt containing the clinical indication, question, and the Top-1 selected report findings and impression. No image pixels were passed to either generator in the primary factorial.

## 3. Retrieval outcomes

| System | Hit@1 | Hit@5 | Hit@10 | MRR | Extractive proxy Token-F1 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.5417 | 0.7056 | 0.7583 | 0.6168 | 0.6550 |
| Qwen3-Embedding-0.6B | 0.4778 | 0.6722 | 0.7500 | 0.5723 | 0.6600 |
| BM25 + BioViL-T | 0.5500 | 0.7139 | 0.7722 | 0.6277 | 0.6985 |
| BM25 + MedSigLIP | **0.5750** | **0.7222** | **0.7833** | **0.6474** | **0.7036** |

The primary MedSigLIP-minus-BM25 MRR difference was:

```text
+0.03069
95% case-grouped bootstrap CI: [0.00902, 0.05368]
```

The lower confidence bound exceeded zero, so the prospectively frozen retrieval-over-text success criterion passed.

The modern general-purpose dense text retriever did not outperform BM25 on this confirmation cohort. This result supports retaining BM25 as an interpretable text baseline and shows that model recency alone did not guarantee better radiology report retrieval.

## 4. Alignment-specific image control

The correctly aligned MedSigLIP condition was compared with 100 deterministic, unique, fixed-point-free shuffled-image assignments.

```text
Correctly aligned MedSigLIP MRR: 0.64744
Mean shuffled-image MRR:         0.59133
Shuffled range:                  [0.56679, 0.62023]
Shuffled assignments at least as high as correct: 0/100
Plus-one Monte Carlo p:          0.00990
```

The correctly aligned condition exceeded every shuffled control and passed the frozen alignment-specificity criterion. This is evidence that the retrieval improvement depended on correct image-report alignment, rather than merely adding arbitrary image embeddings.

This remains a closed-set paired-report retrieval finding. It is not evidence that the image encoder independently diagnosed the target image or that an uploaded image can be linked to its real patient outside the indexed benchmark.

## 5. Downstream QA outcomes

### 5.1 Raw generation

| Generator | Retrieval | Raw Token-F1 | Exact match | Top-1 retrieval accuracy |
|---|---|---:|---:|---:|
| Qwen2.5 | BM25 | 0.1593 | 0.0722 | 0.5417 |
| Qwen2.5 | MedSigLIP | **0.1711** | **0.0750** | **0.5750** |
| MedGemma 1.5 | BM25 | 0.4923 | 0.0000 | 0.5417 |
| MedGemma 1.5 | MedSigLIP | **0.5303** | 0.0000 | **0.5750** |

Raw Token-F1 increased under MedSigLIP retrieval for both generators:

- Qwen2.5: `+0.01183`, 95% CI `[0.00064, 0.02410]`
- MedGemma 1.5: `+0.03802`, 95% CI `[0.02126, 0.05616]`

MedGemma generated longer free-text answers, for which exact string match was zero despite substantially higher token overlap. Exact match is therefore retained as a secondary diagnostic rather than interpreted as the main quality measure.

### 5.2 Frozen semantic verification

The unchanged V5 verifier used `cnut1648/biolinkbert-mednli` with lexical weight `0.2`, support threshold `0.6`, entailment threshold `0.75`, and contradiction threshold `0.5`.

| Generator | Retrieval | Verified Token-F1 | Support rate | Abstention rate | Revision rate |
|---|---|---:|---:|---:|---:|
| Qwen2.5 | BM25 | 0.1669 | 0.3056 | 0.6944 | 0.6944 |
| Qwen2.5 | MedSigLIP | **0.1789** | 0.2889 | 0.7111 | 0.7111 |
| MedGemma 1.5 | BM25 | 0.4840 | 0.9417 | 0.0528 | 0.2889 |
| MedGemma 1.5 | MedSigLIP | **0.5226** | **0.9694** | **0.0250** | **0.2389** |

The primary verified Token-F1 contrasts were:

| Generator | MedSigLIP minus BM25 | 95% case-grouped bootstrap CI | Frozen point-estimate criterion |
|---|---:|---:|---|
| Qwen2.5 | +0.01206 | [0.00165, 0.02370] | Passed |
| MedGemma 1.5 | +0.03857 | [0.02198, 0.05642] | Passed |

Both generator-specific point differences were positive, so the frozen generator-robustness criterion passed. Both confidence intervals also excluded zero, although CI exclusion was not required by the predeclared pass/fail rule.

The secondary difference-in-differences estimate was:

```text
(MedSigLIP - BM25 under MedGemma)
-
(MedSigLIP - BM25 under Qwen2.5)
= +0.02651
95% CI: [0.01202, 0.04229]
```

This suggests that the downstream benefit from improved retrieval was larger under MedGemma than under Qwen2.5. No separate confirmatory threshold was assigned to this secondary estimate.

## 6. Verifier interaction

The frozen verifier behaved very differently across generators. It abstained on approximately 69-71% of Qwen2.5 answers, compared with 2.5-5.3% of MedGemma answers. The Qwen support-rate difference was small and negative (`-0.01667`, 95% CI `[-0.04167, 0.00833]`), while the MedGemma support-rate difference was positive (`+0.02778`, 95% CI `[0.01389, 0.04444]`).

These results should not be interpreted as proof that MedGemma is clinically correct or that Qwen2.5 is clinically unsafe. They show that the frozen automated verifier was much more compatible with MedGemma's answer form and content. Generator-verifier interaction is therefore a material measurement limitation, and verified Token-F1 must be reported alongside raw Token-F1, support, abstention, and revision rates.

No verifier threshold was changed after observing these outcomes.

## 7. Predefined spectrum sensitivity

| Subgroup | Cases | Retrieval MRR difference | 95% CI |
|---|---:|---:|---:|
| Report-indexed normal | 86 | -0.00272 | [-0.02043, 0.01651] |
| Report-indexed abnormal | 34 | +0.11520 | [0.06118, 0.17339] |

The aggregate retrieval improvement was concentrated in report-indexed abnormal cases. For report-indexed normal cases, the retrieval MRR difference was close to zero. This is consistent with images contributing more discriminative information when reports contain abnormal findings, while normal reports remain highly lexically and visually similar.

The corresponding verified Token-F1 differences were:

| Subgroup | Qwen2.5 difference | Qwen 95% CI | MedGemma difference | MedGemma 95% CI |
|---|---:|---:|---:|---:|
| Report-indexed normal | +0.01550 | [0.00471, 0.02845] | +0.02777 | [0.00905, 0.04879] |
| Report-indexed abnormal | +0.00336 | [-0.01931, 0.03044] | +0.06590 | [0.03123, 0.10276] |

These subgroup analyses were predefined secondary sensitivity analyses. They were not assigned independent confirmatory hypotheses and should not be used to claim population-level clinical subgroup effectiveness.

## 8. Computational profile

All primary runs used an NVIDIA GeForce RTX 5070 Laptop GPU.

| Component | Runtime | Peak allocated GPU memory |
|---|---:|---:|
| Qwen3 text embeddings | 3.69 s | 1,345 MiB |
| MedSigLIP embeddings | 27.24 s | 1,817 MiB |
| BioViL-T embeddings | 8.13 s | 822 MiB |
| Qwen2.5, 720 generations | 192.72 s | 2,980 MiB |
| MedGemma 1.5, 720 generations | 1,346.05 s | 3,191 MiB |
| Frozen verifier, 1,440 rows | 44.86 s | not separately recorded |

MedGemma was approximately seven times slower than Qwen2.5 for this factorial, but produced substantially higher raw and verified Token-F1. This is an explicit performance-cost trade-off rather than an unqualified model win.

## 9. Protocol execution note

The formal retrieval and QA generation matrices completed without outcome-driven reruns or case replacement. The first verifier launch stopped before model loading and before writing any verified row because the process pointed to the wrong local Hugging Face cache. The verifier was restarted with `HF_HOME` directed to the already existing repository cache. Model identity, code, thresholds, input rows, and all frozen settings were unchanged. This was a documented technical rerun under the protocol's identical-configuration allowance.

## 10. Main interpretation

The complete V6 evidence chain supports four bounded conclusions:

1. Correctly paired MedSigLIP reranking improved closed-set report retrieval over the same BM25 text baseline on the untouched, broader-spectrum within-source cohort.
2. The improvement was alignment-specific because the correctly paired condition exceeded all 100 shuffled-image controls.
3. The retrieval improvement transferred to downstream report-grounded QA under both an older general generator and a newer medical generator.
4. The magnitude and measured support of that transfer depended on the generator and on the frozen verifier, so retrieval quality, generation quality, and automated grounding measurement remain distinct components.

The strongest research claim is not that newer models are universally better. It is that the alignment-specific image contribution identified in the preliminary study replicated under a modernized encoder/generator stack, while the final effect varied across case spectrum and generator choice.

## 11. Claim limits

- Same-source confirmation is not external validation.
- Case-ID disjointness is not verified patient-level independence.
- Report-derived questions are not physician-authored clinical questions.
- Closed-set paired-report retrieval is not radiographic diagnosis.
- Reference consistency and automated evidence support are not physician-adjudicated correctness.
- No deployment safety, clinical utility, or real-patient report-linkage claim is supported.
- Report-indexed normal/abnormal labels come from the dataset `problems` field and are not new clinical adjudications.

## 12. Frozen artifacts

Tracked summaries and analysis:

- `experiments/post_submission_v6/confirmation_retrieval_summary.json`
- `experiments/post_submission_v6/confirmation_qa_factorial_summary.json`
- `experiments/post_submission_v6/confirmation_qa_factorial_verified_summary.json`
- `experiments/post_submission_v6/confirmation_statistical_analysis.json`

Large per-question rows and embedding caches remain local under repository policy. Their SHA-256 fingerprints are recorded in the tracked summaries and statistical analysis.
