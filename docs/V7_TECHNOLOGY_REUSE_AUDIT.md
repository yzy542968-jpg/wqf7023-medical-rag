# V7 Technology Reuse Audit

## 1. Audit status

This audit defines the technology boundary for the proposed V7 adaptive
multimodal fusion extension. It was prepared against repository commit
`dc53b42`, after the V5 supplemental analysis had been committed and after the
V6 model-modernized confirmation artifacts were already present in the
repository history.

The audit is a reuse and scope decision, not an experiment result. No V7 model
has been trained, no V7 confirmation case IDs have been instantiated, and no
V7 outcome has been inspected. The completed V6 confirmation remains frozen and
is not replaced by this extension.

## 2. Research boundary

V7 asks a narrower question than V6:

> Can a small trainable query-conditional fusion model improve closed-set
> multimodal report retrieval over a validation-tuned global fusion weight,
> while the text retriever, image encoder, report representation, candidate
> shortlist, generator, verifier, and source dataset remain controlled?

The trainable object is the fusion policy only. V7 does not fine-tune
MedSigLIP, Qwen3-Embedding, MedGemma, Qwen2.5, or the BioLinkBERT verifier.
The extension therefore adds a learning-based retrieval contribution without
confounding it with foundation-model parameter updates.

The V7 primary construct remains **case-scoped paired-report retrieval**. It is
not image diagnosis, patient identification outside the indexed corpus,
clinical correctness adjudication, external validation, or a deployment safety
study.

## 3. Reuse decision matrix

| Existing capability | Repository implementation | V7 decision | Boundary |
|---|---|---|---|
| BM25 text retrieval | `src/medical_rag/retrieval/bm25_retriever.py` and V6 runners | Reuse unchanged | `k1=1.5`, `b=0.75`; indication plus question query; deterministic case-ID tie order |
| Qwen3-Embedding baseline | V6 text-retrieval runner and frozen V6 revision | Reuse as secondary diagnostic | It does not replace BM25 and is not used to train the adaptive fusion model |
| MedSigLIP image-text encoder | `src/medical_rag/multimodal/medsiglip.py` | Reuse frozen | Same model revision and local-cache policy as V6; no gradient updates |
| 64-token report processing | `src/medical_rag/multimodal/v6_chunking.py` | Reuse frozen | Same sentence-aware, no-overlap, no-truncation chunk policy; aggregation is fixed before V7 training |
| Image-view aggregation | `aggregate_case_images` and `aggregate_view_embeddings` | Reuse unchanged | Normalize each view, mean, then normalize again |
| Text/image score normalization | `src/medical_rag/multimodal/fusion.py` | Reuse as a feature and baseline implementation | Independent min-max normalization within the BM25 Top-100 shortlist |
| Fixed fusion baseline | `shortlist_score_fusion` | Reuse as V6 anchor | `alpha=0.50`; retained as a historical diagnostic, not the V7 primary comparator |
| Global fusion baseline | New deterministic V7 sweep around the reused fusion function | Add small adapter | Validation grid `alpha=0.00..1.00` with step `0.01`; frozen before confirmation |
| Case-grouped retrieval metrics | `src/medical_rag/evaluation/metrics.py` and V6 evaluation code | Reuse metric definitions | MRR, Hit@1/5/10, target-outside-shortlist rate, and case-level grouping |
| Grouped bootstrap/statistical logic | V6 statistical outputs and grouped resampling pattern | Reuse concept, add V7 adapter | 5,000 case-grouped paired resamples, seed `7026`, no question-level independence assumption |
| V6 cohort hashing | `scripts/build_v6_confirmation_cohort.py` | Reuse design, write V7 builder | Same canonical ID and domain-separated SHA-256 approach; V7 IDs generated only after protocol freeze |
| V6 prior-use evidence | V5/V6 manifests, split files, and qualitative case pack | Reuse as audit inputs | A new machine-check must establish V7 case-ID disjointness before any V7 cohort is selected |
| MedGemma runtime | `src/medical_rag/multimodal/v6_generation.py` and V6 QA runner | Reuse for secondary QA only | Frozen model, prompt, decoding, and verifier; QA cannot select V7 retrieval policy |
| Qwen2.5 runtime | V6 generation runner | Reuse only for historical context if needed | Not required for the V7 primary endpoint |
| Semantic verifier | Existing V6 BioLinkBERT verifier configuration | Reuse frozen for secondary QA | It remains a measurement signal, not a correctness gold standard |
| Dashboard | `app.py` and existing Streamlit demo | Reuse after technical freeze | Demonstration only; no dashboard interaction may generate or tune evaluation outcomes |
| Test and CI structure | `tests/`, compile checks, and existing smoke tests | Reuse and extend | Add unit tests for alpha bounds, pairwise loss, feature leakage, and deterministic selection |

## 4. New components permitted by this audit

Only the following V7-specific components are authorized:

1. A deterministic feature builder that describes query and retrieval state
   without using the target case ID, reference answer, or any post-retrieval
   label.
2. A small trainable fusion learner with a linear candidate and a small MLP
   candidate. Both output a bounded `alpha_q` through a sigmoid and combine the
   already computed text and image scores.
3. A pairwise logistic ranking-loss trainer using only target-positive pairs
   available inside the frozen BM25 Top-100 shortlist.
4. A V7 prior-use audit and deterministic four-block cohort builder.
5. A V7 statistics adapter that reports primary equal-question/case-grouped
   metrics and the prespecified source-balanced sensitivity metric.
6. Optional secondary XGBoost `rank:pairwise` comparison, only if its exact
   version and execution are separately recorded. XGBoost is not a primary
   dependency and its absence must not block V7.

No other new model family, foundation model, retrieval framework, evaluator,
feature family, fusion objective, or clinical task may be introduced after the
V7 development protocol is frozen. Adding one would require terminating this
protocol and reclassifying the work as a new exploratory development cycle.

## 5. Components explicitly rejected for the primary V7 study

The following are outside the approved scope:

- fine-tuning or LoRA/QLoRA of MedSigLIP, Qwen3-Embedding, Qwen2.5, or
  MedGemma;
- replacing BM25 with a new corpus-wide dense retriever;
- LangChain, LlamaIndex, an autonomous Agent, or a new orchestration layer;
- adding a second image encoder to the primary comparison;
- replacing the frozen verifier or treating it as a physician gold label;
- using QA Token-F1 or verifier support to choose the retrieval model;
- generating V7 confirmation IDs before development decisions and the
  confirmation protocol are committed;
- adding human-evaluation scores that have not actually been collected;
- using an online API for restricted radiology text or image pixels;
- adding XGBoost and LightGBM together, or making either one a hidden required
  dependency;
- using `ranx` as a replacement for the existing case-grouped metric and
  bootstrap implementation.

## 6. Reuse risks and controls

### 6.1 Shortlist ceiling

V7 remains a reranking study. The image branch and adaptive learner cannot
recover a target absent from the BM25 Top-100 shortlist. Such queries remain in
validation and confirmation evaluation as genuine retrieval failures and are
reported using `target_outside_shortlist_rate`. They are excluded only from
pairwise gradient optimization because no valid in-shortlist positive-negative
pair exists.

### 6.2 Feature leakage

The feature builder must not use the target case ID, reference answer,
question answer source, qrels, target-in-shortlist flag, or any outcome from a
later QA stage. The target-in-shortlist flag is an evaluation diagnostic only;
it cannot be an input to `alpha_q`.

### 6.3 Foundation-model confounding

All foundation encoders and generators are frozen at the revisions recorded in
the V6 configuration. The only learned parameters are those of the V7 fusion
learner. This keeps the V7 claim about policy learning rather than a broad
model-refresh effect.

### 6.4 Validation reuse

Train A and Train B are used for cross-fitted development decisions. The
separate Validation block is not read until the candidate feature family,
training hyperparameters, and early-stopping epoch are frozen from the two
development folds. Validation then selects the global alpha, model complexity,
and optional gate threshold. Confirmation remains untouched until all of these
decisions are committed.

### 6.5 Generator-verifier interaction

V7 primary conclusions come from retrieval metrics. MedGemma QA transfer is a
secondary analysis run after retrieval is frozen, with the same V6 prompt and
verifier. A change in support rate cannot be used to claim improved clinical
correctness.

## 7. Audit outcome

The project should retain its current repository architecture. The V7
contribution is a bounded extension inside the existing retrieval/evaluation
modules, not a migration to a new framework. The next artifact is the
protocol, followed by a machine-generated prior-use audit. No V7 training or
cohort instantiation is authorized until the protocol commit exists.

## 8. Required evidence before development starts

The following files must be produced or verified before V7 development is
considered active:

- `docs/V7_DEVELOPMENT_PROTOCOL.md` committed independently;
- `config/v7_adaptive_fusion_development.json` matching that protocol;
- a development manifest with explicit case IDs and a canonical-ID SHA-256;
- a machine-checked overlap report against all prior V1-V6 and qualitative
  manifests;
- a report-indexed normal/abnormal/indeterminate spectrum summary;
- a source-frame fingerprint and a record that patient-level independence is
  not verifiable from the processed source.

Until those files exist, the only valid V7 status is **protocol preparation**.
