# V6 Model-Modernized Confirmation Study: Development Decision Record

## 1. Status and scope

This record closes the V6 development stage specified in
`docs/V6_DEVELOPMENT_PROTOCOL.md`. It records deterministic decisions made from
the previously frozen V5 development split. It is a locally committed decision
record, not a formal preregistration. V5 artifacts and conclusions remain
unchanged.

No V6 confirmation case IDs were generated, previewed, or inspected during this
stage. The development source contained 120 target cases, 360 report-derived
questions, and the frozen V5 240-case candidate pool. Development cases were
machine-verified to have zero case-ID overlap with both the V6
confirmation-eligible and stratifiable frames. Patient-level independence could
not be verified because reliable patient identifiers are unavailable.

## 2. Development provenance

The development protocol and separation audit were committed as `b85db42`.
Text-retriever development was committed as `d63a9b8`, multimodal retrieval as
`72ee2d2`, and the generator factorial and frozen-verifier application as
`91a9449`.

| Artifact | SHA-256 or identifier |
|---|---|
| Source cases JSONL | `56e367190396011d4d67f43e7e733389a8346890bf8729e82fb4326d063bbd68` |
| Frozen V5 cohort | `f480f534291ff3081e06bf5c587596fdaa81e266534c2e9fb61215cd4b469e8b` |
| V6 development case IDs | `a7f381262f4f9ae29a4a68f5bdca884686d97b1de2f95e3e22b3491be9aebfa5` |
| V6 eligible case IDs | `cd7b59c7d890846055ad43b8b108cec28b475ecf810f14966a538a3c6c7e98eb` |
| V6 stratifiable case IDs | `e0b03681591d48d4d83babca2d72b80872d2cb64dd01a170911d78d1b3ee5186` |
| Text retrieval rows | `e09faae4dcf2c33359df3557e7a68d994ac95ae2de26330939495668329a5208` |
| Multimodal retrieval rows | `f477e58d76bdf7c040c328093de12bf700e4eacc34429f12bdbb8623f00c6776` |
| QA factorial rows | `ce0b5852a384a30e61d7015fd322aa767139b02d9360b143eeda73478c605087` |
| Verified QA rows | `9071598aa3633cb20a11c4c1bd67525b49988adce49d8d65771bcd8542725505` |

Large per-question rows, prompts, generated answers, report text, model caches,
and image pixels remain local under repository policy. Tracked summaries retain
counts, metrics, model revisions, implementation hashes, and local-row hashes.

## 3. Frozen development decisions

| Decision | Frozen value | Rule and basis |
|---|---|---|
| Primary text retriever `T*` | BM25, `k1=1.5`, `b=0.75` | Qwen3 had to exceed BM25 MRR by at least 0.005; it did not. |
| Primary text query | Clinical indication plus question | Specified before development outcomes. |
| Text shortlist | Top 100 from `T*` | Inherited fixed V5 policy. |
| MedSigLIP report chunks | Sentence-aware findings/impression chunks, at most 64 tokenizer tokens | No overlap, truncation, or dropped non-empty segment. |
| Report aggregation | Maximum image-to-chunk cosine | Max MRR exceeded mean MRR by 0.02523, above the 0.005 threshold. |
| Multi-view aggregation | Normalize views, mean by case, normalize again | Same policy for MedSigLIP and BioViL-T. |
| Fusion | Independent shortlist min-max; 0.5 text plus 0.5 image | No V6 weight sweep permitted. |
| Primary image-text encoder | MedSigLIP-448 | Modern encoder under the selected standardized chunk policy. |
| Historical encoder comparator | BioViL-T | Same chunk boundaries and selected max policy. |
| Historical generator | Qwen2.5-1.5B-Instruct, FP16 | Retained for the primary 2 x 2 robustness design. |
| Modern generator | MedGemma 1.5 4B IT, 4-bit NF4 with BF16 compute and double quantization | Technical 8 GB GPU preflight; selection did not use answer quality. |
| Primary generator input | Indication, question, and Top-1 report findings/impression; no image pixels | Identical semantic prompt across generators. |
| Decoding | Greedy, no sampling, maximum 256 new tokens | Model-specific official chat template is the only wrapper difference. |
| Primary verifier | Frozen V5 BioLinkBERT-MedNLI semantic verifier | No model or threshold change. |

## 4. Text-retriever decision

Qwen3-Embedding used model revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`, 1,024-dimensional normalized
embeddings, and the frozen radiology retrieval instruction. BM25 and Qwen3 used
identical indication-plus-question queries, reports, questions, and candidates.

| Retriever | Hit@1 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.59722 | 0.76389 | 0.83611 | 0.66874 |
| Qwen3-Embedding-0.6B | 0.48333 | 0.68889 | 0.75278 | 0.57810 |

The Qwen3-minus-BM25 MRR difference was `-0.09065`. The frozen rule therefore
selected BM25 as `T*`. Qwen3 remains a secondary modern dense baseline in
confirmation; its newer release date is not treated as evidence of domain
superiority.

The question-only diagnostic produced MRR 0.02282 for BM25 and 0.02615 for
Qwen3, compared with 0.66874 and 0.57810 when clinical indication was included.
This confirms that the indication is a strong retrieval cue in this constructed
task and must remain visible in interpretation.

## 5. Multimodal retrieval decision

MedSigLIP used revision
`9cea28a1a1195f665105faa6e8544c112fd960a4`, 448-pixel model processing,
1,152-dimensional embeddings, FP16 model execution, and a strict 64-token text
limit. The 240 candidate reports produced 582 chunks and 454 image views; the
maximum observed chunk length was exactly 64 tokens. No report section was
silently truncated.

BioViL-T used text revision `692f09e`, image-weight MD5
`a83080e2f23aa584a4f2b24c39b1bb64`, and 128-dimensional embeddings. Both
encoders used the exact same MedSigLIP-tokenizer-derived chunk texts and case-view
aggregation policy.

| Retrieval system | Hit@1 | Hit@5 | Hit@10 | MRR | Extractive proxy Token-F1 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.59722 | 0.76389 | 0.83611 | 0.66874 | 0.66052 |
| MedSigLIP mean chunk | 0.64444 | 0.80556 | 0.86111 | 0.72239 | 0.74748 |
| MedSigLIP max chunk | 0.67778 | 0.83333 | 0.88611 | 0.74762 | 0.76294 |
| BioViL-T mean chunk | 0.61667 | 0.80556 | 0.88333 | 0.70611 | 0.72617 |
| BioViL-T max chunk | 0.61111 | 0.81667 | 0.88056 | 0.70300 | 0.72253 |

Maximum image-to-chunk cosine exceeded normalized mean aggregation by 0.02523
MRR, so the frozen selection rule chose maximum cosine. Under that common policy,
MedSigLIP exceeded standardized BioViL-T by 0.04462 MRR and BM25 by 0.07887 MRR
on development. These are development observations, not confirmation claims.

Encoding the complete development retrieval state took 26.89 seconds for
MedSigLIP and 8.22 seconds for BioViL-T on the reference GPU. Peak allocated GPU
memory was 1,816 MiB and 857 MiB, respectively. Cache hits are excluded from
these build-time measurements.

## 6. Generator and verifier decisions

Qwen2.5 used revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` in FP16. MedGemma used revision
`91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b` with NF4 weight quantization,
double quantization, and BF16 compute. The MedGemma choice was made from hardware
headroom: a successful single-example preflight used about 3.18 GiB peak
allocated GPU memory, leaving materially more headroom than unquantized BF16 on
an 8,151 MiB GPU. No answer score was used to choose precision.

The full development matrix contained 1,440 rows: 360 questions under each of
four retrieval-generator conditions. Every condition had the identical qid set,
120 target cases, no duplicate `(system, qid)` key, and no image pixels in the
generator input.

| System | Raw Token-F1 | Verified Token-F1 | Support rate | Abstention rate |
|---|---:|---:|---:|---:|
| BM25 to Qwen2.5 | 0.12614 | 0.13730 | 0.23333 | 0.76667 |
| MedSigLIP to Qwen2.5 | 0.15164 | 0.16242 | 0.25764 | 0.74167 |
| BM25 to MedGemma | 0.47753 | 0.47126 | 0.97647 | 0.01111 |
| MedSigLIP to MedGemma | 0.56135 | 0.55466 | 0.97223 | 0.01111 |

The multimodal-minus-text verified Token-F1 difference was positive for Qwen2.5
(`+0.02512`) and MedGemma (`+0.08340`). The corresponding raw differences were
`+0.02550` and `+0.08382`. These observations support retaining the planned 2 x 2
confirmation matrix but do not establish confirmation performance.

Qwen2.5 produced a high rate of explicit insufficient-evidence responses under
the common V6 prompt. This generator-specific behavior was not used to rewrite
the prompt, remove Qwen2.5, or retune the verifier. Preserving the common semantic
prompt is necessary for an interpretable robustness comparison. Absolute scores
across generators must be interpreted with their markedly different answer
length and abstention behavior.

The frozen verifier used `cnut1648/biolinkbert-mednli`, lexical weight 0.2,
support threshold 0.6, entailment threshold 0.75, and contradiction threshold
0.5. Its configuration SHA-256 was
`302e8ce368351af087259e53f63e134b4514fa4b9e1fd3a209e5e041a101fe9f`.
Automated support and revision results are not human clinical adjudication.

Qwen2.5 generated 720 development answers at 2.77 records/second and about 2,987
MiB peak allocated GPU memory. MedGemma generated 720 at 0.36 records/second and
about 3,204 MiB. These are single-machine, warm-cache measurements and will be
reported as engineering cost rather than universal model speed.

## 7. Decisions not authorized by development

Development did not authorize any of the following:

- removing BM25, Qwen3, BioViL-T, Qwen2.5, or MedGemma from the prespecified
  confirmation comparisons;
- changing the common prompt because one generator abstained more often;
- changing fusion weights, shortlist size, verifier thresholds, decoding, or
  quantization in response to answer outcomes;
- treating report-derived reference consistency as physician-adjudicated
  correctness;
- claiming patient-level independence, external validation, diagnosis, clinical
  utility, or deployment safety;
- instantiating or previewing the deterministic confirmation cohort before the
  confirmation protocol and config are committed.

## 8. Development closure

The following state is now ready to enter the confirmation protocol:

```text
T*                         = BM25(k1=1.5, b=0.75)
text query                 = indication + question
shortlist                  = 100
fusion                     = 0.5 text + 0.5 image after shortlist min-max
modern image encoder       = MedSigLIP-448 at pinned revision
report aggregation         = maximum image-to-chunk cosine
historical image comparator= standardized BioViL-T
generators                 = Qwen2.5 FP16 and MedGemma 1.5 NF4/BF16
primary generator pixels   = none
decoding                   = greedy, max_new_tokens=256
verifier                   = frozen V5 semantic verifier
statistics                 = case-grouped, 5,000 bootstrap resamples, seed 7026
```

After this record is committed, the next allowed actions are to write and commit
`V6_CONFIRMATION_PROTOCOL.md` and its frozen machine-readable config. Actual
confirmation case IDs may be generated only after that separate protocol commit.
