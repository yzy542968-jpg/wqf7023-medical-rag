# Multimodal V4 protocol

## Why this extension exists

The registered project title refers to paired radiology images and reports. Earlier implemented experiments consume report text while retaining image filenames only as case metadata. V4 closes that modality gap without discarding the report-only work: the existing report RAG becomes a baseline and evidence branch, while a biomedical vision-language encoder consumes the actual chest X-ray pixels.

V4 tests cross-modal case matching and downstream report-grounded QA. It does not claim to diagnose a previously unseen image. Given an image from an existing IU-Xray case, the system retrieves the matching report from a fixed candidate pool and answers a question using the retrieved evidence.

## Frozen data boundary

- Development: the existing 600-case V2 main cohort, 1,800 questions.
- Confirmation: the existing 120-case V2 confirmation cohort, 360 questions.
- Candidate pool: the union of those 720 cases.
- Unit of resampling: case ID, not question.
- Image source: the official NLM OpenI PNG archive.
- Raw images and model weights remain local and are never committed.

All case and content fingerprints are recorded in `config/multimodal_v4.json`. Cases are eligible when the normalized case record contains at least one declared image and that filename can be matched in the official archive. Any exclusions must be reported before metric computation.

## Inputs and systems

Every query contains a chest X-ray image, the case indication, and one of the three existing case-scoped questions. All available views are encoded independently; their normalized embeddings are averaged into one case image representation.

1. `report_only_bm25` searches reports using indication plus question.
2. `image_only_biomedclip` ranks report embeddings by cosine similarity to the case image embedding.
3. `paired_rrf_fusion` combines complete BM25 and BiomedCLIP rankings using weighted reciprocal-rank fusion.
4. `paired_agent_with_evidence_checking` uses the fused top report, generates or extracts an answer, checks report support, and may abstain.

BiomedCLIP is used without fine-tuning. The fusion text weight is selected from the registered grid by development MRR. Ties are resolved by distance to 0.5 and then by the smaller text weight. The selected policy must be committed before confirmation evaluation.

## Outcomes

Primary retrieval outcomes are Hit@1, Hit@5, Hit@10, and MRR. Primary downstream QA outcome is Token-F1 from deterministic routed extraction at the retrieved top-1 report. This isolates retrieval effects before adding a generator.

The local VLM comparison is secondary. It compares image-only answering with paired image-plus-retrieved-report answering using the frozen prompt and deterministic decoding in the configuration. Evidence checking is evaluated through answer support, false-answer rate, and abstention. Runtime, throughput, and peak CUDA memory are machine-specific cost measurements.

Confirmation comparisons use 5,000 paired case bootstrap resamples with seed 7023. No fusion weight, prompt, threshold, or model choice may be changed after confirmation outcomes are inspected.

## Interpretation limits

An image-to-own-report benchmark is a cross-modal retrieval task. It demonstrates that pixels participate in the system and can identify paired evidence, but it does not establish diagnostic accuracy on new patients. The report remains expert-authored primary evidence. VQA-RAD can later test image-question answering, and authorized RadQA can later test natural report questions; neither is silently treated as equivalent to this paired retrieval task.
