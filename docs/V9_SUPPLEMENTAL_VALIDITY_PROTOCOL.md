# V9 Supplemental Validity Audit Protocol

## Status

This protocol defines a post-hoc exploratory extension after the V9 primary
retrieval, QA, and agent outcomes were inspected. It is committed before the
supplemental analyses below are executed. It is not a preregistration and does
not reopen the frozen V9 model, prompt, cohort, qrel definition, success rule,
or primary statistical result.

The purpose is to test how sensitive the completed conclusions are to known
measurement and data-quality limitations. Favorable or unfavorable findings
must both be retained. No supplemental outcome may be used to retune V9.

## A1. Cross-split duplicate and near-duplicate audit

Reports are represented as `findings + newline + impression`, normalized with
Unicode NFKC, lowercase conversion, and whitespace collapse. Exact duplicates
use SHA-256 over UTF-8 text with no trailing newline. Near duplicates use a
deterministic character-boundary TF-IDF representation with 3-5 character
ngrams, `min_df=2`, sublinear term frequency, and L2 normalization.

For every Validation and Test case, the audit records its maximum similarity
to a Train report. Counts are reported at cosine thresholds 0.90, 0.95, and
0.99. The predefined retrieval sensitivity subset excludes Test cases with a
Train-report similarity of at least 0.95 and recomputes frozen per-query
retrieval aggregates without changing any ranking.

Image duplication uses a 64-bit difference hash computed after deterministic
9-by-8 grayscale resizing. Case distance is the minimum Hamming distance over
all cross-case view pairs. Counts are reported at distances 0, 4, and 8. Image
hashes diagnose duplication only; they are not patient identifiers.

## A2. Relevance-construct sensitivity

The frozen ranking systems are evaluated under three offline relevance views:

1. active report labels only;
2. RadGraph entity-relation facts only;
3. the frozen 0.60 label / 0.40 RadGraph combination.

The systems, rankings, model checkpoint, and candidate bank remain unchanged.
The analysis reports nDCG@10, thresholded MRR, and Recall@10. It asks whether
the ordering of system conclusions is robust to the operational definition of
similarity. It does not create physician-adjudicated relevance.

## A3. Clinical-structure generation metrics

Frozen G0-G3 answers are scored against their frozen findings or impression
references with F1-RadGraph entity, entity-relation, and complete rewards.
F1-CheXbert is also attempted if the official local dependency and checkpoint
can be installed and retained without sending report text to an online API.

Results use 10,000 case-grouped bootstrap samples with seed 7041. These metrics
are post-hoc and remain imperfect proxies for clinical correctness.

## A4. Modern dense-text baseline

`Qwen/Qwen3-Embedding-0.6B` at revision
`97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` encodes the same indication-plus-
question query and historical report bank used by BM25. The frozen radiology
retrieval instruction and normalized 1,024-dimensional embeddings are used.

The comparison is descriptive dense-text versus BM25 under the same V9 qrels.
It cannot select or replace a V9 model because the V9 Test outcomes are already
known.

## A5. Fixed paraphrase robustness

Each of the three V9 question roles has one canonical form and two fixed
paraphrases specified in the JSON configuration. BM25, Qwen3 dense retrieval,
and the frozen learned reranker are evaluated without changing role one-hot
features. Reported outcomes are per-variant nDCG@10, canonical-versus-paraphrase
Top-1 agreement, Top-10 Jaccard overlap, and within-case nDCG dispersion.

This analysis measures deterministic phrasing sensitivity only. The questions
are not physician authored, and it does not establish robustness to arbitrary
clinical questions, compositional reasoning, or negation.

## A6. Structured-output reparse audit

No generation is repeated. A deterministic parser may remove markdown fences,
extract a balanced JSON object, and remove trailing commas. It must not invent
missing fields or close a truncated object by fabrication. The audit reports
the original and recovered valid rates, answer-change rate, and unrecoverable
rate. Reparsed outputs are an engineering sensitivity and do not replace the
frozen V9 Token-F1 analysis.

## Reporting rule

Every result is labeled `post-hoc exploratory`. The final report must state
which analyses completed, which could not run because of local dependency or
licensing constraints, and whether each result strengthens, weakens, or leaves
unchanged the scoped V9 interpretation.
