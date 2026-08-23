# V9 Supplemental Validity and Robustness Results

## Status and scope

This document reports the post-hoc exploratory analyses prespecified in
`docs/V9_SUPPLEMENTAL_VALIDITY_PROTOCOL.md`. The protocol was committed after
the V9 technical freeze but before these supplemental outcomes were generated.
None of the analyses changed a V9 model, checkpoint, prompt, threshold, split,
question, primary metric, or frozen quantitative result.

These results strengthen or qualify the interpretation of V9. They are not a
second confirmation study, a formal preregistration, an external validation,
or a clinical adjudication.

## 1. Cross-split duplicate and near-duplicate audit

Normalized findings-plus-impression text was compared between each Validation
or Test case and all 2,631 Train cases. Exact equality used SHA-256 after NFKC,
lowercasing, and whitespace normalization. Near-duplicate text used character
3-5 gram TF-IDF cosine similarity. Images were compared with 64-bit dHash over
all view pairs. The image diagnostic is deliberately described as a perceptual
collision measure; it cannot establish patient identity.

| Split | Cases | Exact Train report duplicate | Report cosine >=0.90 | >=0.95 | >=0.99 | dHash distance 0 | <=4 | <=8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 376 | 84 | 115 | 101 | 89 | 3 | 222 | 362 |
| Test | 752 | 162 | 214 | 187 | 170 | 11 | 447 | 731 |

The high image dHash counts reflect the limited visual diversity and shared
layout of chest radiographs as well as possible duplicates; they must not be
interpreted as 731 repeated patients. The report audit is more directly
relevant to the report-derived qrel. At the prespecified cosine threshold of
0.95, 187 Test cases were excluded and 565 remained.

| Frozen system | Full Test nDCG@10 | nDCG@10 after >=0.95 report-similarity exclusion |
|---|---:|---:|
| R0 BM25 | 0.134156 | 0.139179 |
| R1 image-image | 0.315561 | 0.264642 |
| R2 image-report | 0.274069 | 0.247601 |
| R3 fixed multimodal | 0.246935 | 0.226451 |
| R4 learned MLP | **0.327942** | **0.279730** |

R4 remained first after the exclusion and exceeded R1 by 0.015088 nDCG@10.
This preserves the direction of the primary retrieval finding, although the
substantial near-duplicate prevalence is an important same-source validity
threat and should be stated prominently.

## 2. Relevance-construct sensitivity

The five frozen rankings were re-evaluated under three prespecified qrels:
active-label similarity alone, RadGraph-fact similarity alone, and the frozen
0.60/0.40 combination. No ranking was retrained or selected.

| Qrel | BM25 | Image-image | Image-report | Fixed fusion | Learned MLP | R4 minus R1 |
|---|---:|---:|---:|---:|---:|---:|
| Active labels only | 0.081369 | 0.318698 | 0.288005 | 0.222156 | **0.333863** | +0.015165 |
| RadGraph facts only | 0.207467 | 0.289271 | 0.225719 | 0.266962 | **0.292220** | +0.002950 |
| Frozen 0.60/0.40 | 0.134156 | 0.315561 | 0.274069 | 0.246935 | **0.327942** | +0.012381 |

R4 ranked first under all three constructs. The advantage was smallest for
RadGraph facts alone, so the defensible conclusion is ordering robustness with
construct-dependent effect size, not uniform large superiority.

## 3. Modern dense text baseline and wording robustness

`Qwen/Qwen3-Embedding-0.6B` at revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` encoded the same 2,608 historical
reports and 6,768 indication-question variants. Embeddings were normalized and
the model was used only as an exploratory baseline. Each of the findings,
impression, and acute roles had one canonical wording and two fixed
researcher-written paraphrases.

| System | Canonical nDCG@10 | Top-1 agreement with canonical | Top-10 Jaccard | Mean within-case-role nDCG SD |
|---|---:|---:|---:|---:|
| BM25 | 0.134156 | 0.114140 | 0.109597 | 0.038239 |
| Qwen3 dense text | 0.195633 | 0.357048 | 0.322581 | 0.059096 |
| Learned multimodal MLP | **0.327942** | **0.996897** | **0.888661** | **0.006195** |

Qwen3 dense text exceeded BM25 by 0.061476 nDCG@10 on the canonical questions,
but remained far below R4. Its ranking also changed materially with wording.
R4 was almost invariant because visual channels and their learned retrieval
state dominated the weak wording-sensitive BM25 component. This is useful
robustness evidence, but it should not be described as physician-authored
language validation.

## 4. Clinical generation metrics

All 5,480 frozen QA answers were scored locally with
`modern-RadGraph-XL`. F1-RadGraph is automated graph overlap, not diagnostic
correctness. Confidence intervals used 10,000 case-grouped bootstrap samples.
The supplemental protocol used the descriptive key
`g1_bm25_report_rag`; the immutable source rows use `g1_bm25_rag`. They are the
same condition and the alias is recorded in the result JSON.

| System | Entity F1 | Entity-relation F1 | Complete F1 |
|---|---:|---:|---:|
| G0 no retrieval | 0.158313 | 0.143371 | 0.124852 |
| G1 BM25 RAG | 0.137316 | 0.124856 | 0.103866 |
| G2 fixed multimodal RAG | **0.167188** | **0.149006** | **0.124971** |
| G3 learned multimodal RAG | 0.159906 | 0.142024 | 0.124803 |

G3 exceeded G1 on all metrics. For complete F1, the paired difference was
+0.020937 with 95% CI [0.012863, 0.028992]. G3 did not show a resolved
advantage over G0: complete-F1 difference -0.000049, CI
[-0.006803, 0.006897]. It also did not exceed G2: difference -0.000168, CI
[-0.008553, 0.008127]. These results qualify the Token-F1 result. Learned
retrieval improved over the weak text-RAG path, but automated clinical graph
overlap did not establish general generation superiority over no retrieval or
fixed multimodal RAG.

An official local F1CheXbert dependency was not installed. In accordance with
the protocol, the metric is reported as unavailable and no proxy was silently
substituted.

## 5. Structured-output reparse audit

Frozen raw generations were reparsed with balanced-object extraction, markdown
fence removal, and trailing-comma removal. Truncated objects were never closed
or fabricated.

| Scope | Rows | Original valid | Reparsed valid | Newly recovered | Unrecoverable |
|---|---:|---:|---:|---:|---:|
| All systems | 5,480 | 46.30% | 46.30% | 0 | 53.70% |
| G0 | 1,370 | 42.04% | 42.04% | 0 | 57.96% |
| G1 | 1,370 | 39.42% | 39.42% | 0 | 60.58% |
| G2 | 1,370 | 46.50% | 46.50% | 0 | 53.50% |
| G3 | 1,370 | 57.23% | 57.23% | 0 | 42.77% |

All 2,537 valid objects were already recovered by the frozen parser. The 2,943
remaining failures were not repairable with the prespecified formatting-only
policy and were predominantly token-ceiling truncations. Answer-change rate
was 0%. The incomplete-output limitation is therefore not explained by the
original greedy regular expression.

## 6. Final interpretation

The supplemental audit supports five conclusions:

1. The learned reranker's retrieval ordering survives a strict report
   near-duplicate exclusion, but same-source duplication is a material validity
   threat.
2. R4 ranks first under label-only, fact-only, and combined qrels, although its
   fact-only margin over image retrieval is small.
3. A current dense text baseline improves substantially over BM25 but does not
   approach the learned multimodal system.
4. R4 is much less sensitive to fixed wording changes than either text-only
   retriever.
5. Retrieval improvement transfers clearly relative to BM25-RAG, but neither
   Token-F1 nor F1-RadGraph supports a claim that learned RAG is universally
   superior to fixed multimodal RAG or target-image-only generation.

The V9 technical freeze remains unchanged. These findings are incorporated as
validity, robustness, and limitation evidence only.

## Public artifacts

- `data/splits/v9/v9_cross_split_duplicate_summary.json`
- `data/splits/v9/v9_cross_split_duplicate_audit.csv`
- `data/splits/v9/v9_qrel_sensitivity_summary.json`
- `data/splits/v9/v9_dense_text_robustness_summary.json`
- `data/splits/v9/v9_clinical_metrics_summary.json`
- `data/splits/v9/v9_structured_reparse_summary.json`

Per-answer clinical scores, dense embeddings, ranking rows, prompts, raw
generations, source reports, and image pixels remain local under repository
policy; their hashes are recorded in the aggregate artifacts.
