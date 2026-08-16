# Second-Pass Optimization Audit

Updated: 2026-08-14

## Bottom Line

There is substantial optimization space, but the highest-value work is no longer prompt engineering or a larger generator. The current bottleneck is the formulation and calibration of retrieval.

The strongest diagnostic result is the oracle experiment:

| System condition | Verified Token-F1 |
|---|---:|
| Actual adaptive retrieval | 0.206 |
| Correct target case supplied | 0.425 |

The same generator and verifier more than double their score when retrieval is correct. Retrieval and benchmark design should therefore receive most of the next research effort.

## Must Fix Before Strong Thesis Claims

### 1. Redesign the retrieval benchmark

The current questions are automatically formed from indication or problem metadata. Those fields are also present in the indexed document representation:

- indication-based questions: source text appears in 100% of BM25 documents and MedCPT titles;
- summary questions: the problem label appears in 100% of MedCPT titles;
- 68/360 rows use an identical query that points to more than one target case;
- 25/108 held-out rows are globally ambiguous in this way.

This creates both a retrieval shortcut and an identifiability problem. A v2 task should choose one of two defensible designs:

1. Patient-known evidence QA: the case/report is supplied, and retrieval selects relevant sentences or chunks only within that case.
2. Open-corpus semantic retrieval: queries have independently annotated multi-case relevance judgments instead of one synthetic target report.

The first option aligns best with the current case-safety contribution and can reuse nearly all existing generation, agent, and Dashboard code.

### 2. Complete blinded human evaluation

Automatic Token-F1, RadGraph, and Medical NLI are not sufficient clinical correctness measures. The prepared 36-case/144-response sheet should be independently scored for correctness, grounding, harmfulness, and preference. A second reviewer would permit inter-rater agreement.

### 3. Separate faithfulness from retrieval correctness

Wrong-retrieval answers have only 0.117 verified Token-F1 but 83.4% evidence support. This is expected: an answer can be faithful to the wrong patient's report. Final reporting must present:

- retrieval correctness;
- conditional evidence faithfulness;
- end-to-end answer correctness;
- abstention/coverage.

Do not combine these into a single “safety” score.

## High-Value Technical Improvements

### 1. Calibrated retrieval abstention

The current threshold policy raises selective accuracy to 31.0% but does not improve overall Hit@1. Use a development/calibration split or nested cross-validation to calibrate a risk-coverage curve. Conformalized abstention is a suitable future direction, but it needs an untouched calibration partition and more cases.

### 2. Better retrieval supervision

The current MedCPT reranker is zero-shot and its development gain did not transfer. Better options are:

- hard-negative mining from confusing OpenI cases;
- lightweight cross-encoder fine-tuning on development qrels;
- query-type-specific retrieval or separate indication/abnormality indexes;
- reciprocal-rank or learned fusion evaluated with nested case-level cross-validation.

Any trained reranker needs new validation data; the current held-out set has already been opened and should remain frozen.

### 3. Independent verifier calibration

The polarity hard guard now passes a 120-pair development stress test with 100% acceptance and 100% contradiction rejection. This only validates explicit negation flips. Human labels are still needed for paraphrases, unsupported severity/location, uncertain language, and mixed-claim sentences.

### 4. Stronger radiology metrics

RadGraph should remain, but newer radiology-report metrics such as GREEN and RaTEScore can add interpretable clinical error categories and entity-aware similarity. They should be treated as complementary metrics, not substitutes for human review.

## Optional Major Extension: True Multimodal RAG

The current system is text-only. Only 10 local X-ray examples are present, and images do not enter retrieval, generation, or verification. A genuine multimodal extension would require:

1. downloading a complete permitted image corpus;
2. image-text encoders such as BioViL/CXR-CLIP or another chest-X-ray-specific model;
3. image-to-case or image-text fusion retrieval;
4. a vision-language generator or image-grounded verifier;
5. image-aware evaluation and clinical review.

This could become a strong extension, but it changes the thesis scope and should not be claimed from the current implementation.

## What Not to Optimize First

- More prompt templates: direct prompting already wins clearly on development data.
- A larger LLM: the oracle result shows retrieval dominates the current error budget.
- An unconditional reranker: it improved development MRR but reduced held-out MRR.
- More autonomous agent loops without an evaluation target: extra planning steps do not solve non-identifiable queries.
- Test-set threshold tuning: the current held-out set has been analyzed and must not drive another configuration search.

## Recommended Next Research Order

1. Freeze the current system as Experiment v1.
2. Complete blinded human evaluation for v1.
3. Build an identifiable v2 patient-known evidence-retrieval benchmark from unused OpenI cases.
4. Reserve new case-level development, calibration, and final-test partitions.
5. Train or calibrate retrieval on development/calibration only.
6. Re-run the same Qwen and verifier before considering a larger model.
7. Add GREEN or RaTEScore and a second human reviewer.
8. Treat multimodal image use as an optional extension, not a hidden assumption.

## Audit Artifacts

- `experiments/final_optimized/validity_audit/research_validity_audit.json`
- `experiments/final_optimized/oracle_test/final_optimized_test_summary.json`
- `experiments/final_optimized/verifier_stress_test/development_polarity_stress_test.json`
- `experiments/final_optimized/statistics/held_out_test_grouped_bootstrap.json`
- `experiments/final_optimized/contamination/report_rag_cross_case_contamination.json`

## Method References

- GREEN: <https://aclanthology.org/2024.findings-emnlp.21/>
- RaTEScore: <https://aclanthology.org/2024.emnlp-main.836/>
- MIMIC-CXR-JPG official dataset: <https://physionet.org/content/mimic-cxr-jpg/2.1.0/>
- CAP conformal abstention: <https://proceedings.mlr.press/v304/tayebati26a.html>
