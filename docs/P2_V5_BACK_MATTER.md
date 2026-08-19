# References

Bae, S., Kyung, D., Ryu, J., et al. (2023). EHRXQA: A multi-modal question answering dataset for electronic health records with chest X-ray images. *Advances in Neural Information Processing Systems*.

Bannur, S., Hyland, S., Liu, Q., et al. (2023). Learning to exploit temporal structure for biomedical vision-language processing. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*.

Boecking, B., Usuyama, N., Bannur, S., et al. (2022). Making the most of text semantics to improve biomedical vision-language processing. *European Conference on Computer Vision*.

Demner-Fushman, D., Kohli, M. D., Rosenman, M. B., et al. (2016). Preparing a collection of radiology examinations for distribution and retrieval. *Journal of the American Medical Informatics Association, 23*(2), 304-310.

Es, S., James, J., Espinosa-Anke, L., and Schockaert, S. (2024). RAGAS: Automated evaluation of retrieval augmented generation. *Proceedings of the European Chapter of the Association for Computational Linguistics: System Demonstrations*.

Jin, Q., Kim, W., Chen, Q., et al. (2023). MedCPT: Contrastive pre-trained transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval. *Bioinformatics, 39*(11), btad651.

Lau, J. J., Gayen, S., Ben Abacha, A., and Demner-Fushman, D. (2018). A dataset of clinically generated visual questions and answers about radiology images. *Scientific Data, 5*, 180251.

Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems*.

Ngo, N. T., Nguyen, C. V., Dernoncourt, F., and Nguyen, T. H. (2024). Comprehensive and practical evaluation of retrieval-augmented generation systems for medical question answering. *arXiv:2411.09213*.

Pal, A., Umapathi, L. K., and Sankarasubbu, M. (2023). Med-HALT: Medical domain hallucination test for large language models. *Proceedings of CoNLL*, 314-334.

Qwen Team. (2025). Qwen2.5 technical report. *arXiv:2412.15115*.

Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning*.

Robertson, S., and Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333-389.

Romanov, A., and Shivade, C. (2018). Lessons from natural language inference in the clinical domain. *Proceedings of EMNLP*, 1586-1596.

Singhal, K., Azizi, S., Tu, T., et al. (2023). Large language models encode clinical knowledge. *Nature, 620*, 172-180.

Soni, S., Gudala, M., Pajouhi, A., and Roberts, K. (2022). RadQA: A question answering dataset to improve comprehension of radiology reports. *Proceedings of LREC 2022*, 6250-6259.

Xiong, G., Jin, Q., Lu, Z., and Zhang, A. (2024). Benchmarking retrieval-augmented generation for medicine. *Findings of ACL 2024*, 6233-6251.

# Appendices

## Appendix A: Frozen V5 Result Sources

- Configuration: `config/multimodal_v5.json`
- Cohort manifest: `data/processed/openi_multimodal_v5_cohort.json`
- Retrieval summary: `experiments/post_submission_v5/confirmation_retrieval_summary.json`
- Report-only QA summary: `experiments/post_submission_v5/qa_report_only/final_optimized_test_summary.json`
- Multimodal QA summary: `experiments/post_submission_v5/qa_multimodal/final_optimized_test_summary.json`
- Statistical analysis: `experiments/post_submission_v5/v5_statistics.json`
- Artifact manifest: `experiments/post_submission_v5/artifact_manifest.json`
- Runtime summary: `docs/V5_RUNTIME_SUMMARY.md`

The technical freeze is identified by commit `10f57ba` and tag `v5-technical-freeze`.

## Appendix B: Qualitative Audit Trail

- Frozen protocol v1.0: `docs/V5_QUALITATIVE_ANALYSIS_PROTOCOL.md`
- Refined taxonomy v1.1: `docs/V5_QUALITATIVE_TAXONOMY_V1_1.md`
- Review guide: `docs/V5_QUALITATIVE_REVIEW_GUIDE.md`
- Public 360-question numeric index: `experiments/post_submission_v5/qualitative_case_pack.csv`
- Public 24-question review index: `experiments/post_submission_v5/qualitative_representative_cases.csv`
- Researcher review record: `docs/V5_QUALITATIVE_RESEARCHER_REVIEW_RECORD.md`
- Final qualitative analysis: `docs/V5_QUALITATIVE_ERROR_ANALYSIS.md`

The qualitative freeze is identified by commit `f3fefbf` and tag `v5-qualitative-freeze`. Full report text, generated answers, prompt packs, and image pixels remain local under repository policy.

## Appendix C: Reproduction Entry Points

**Repository:** https://github.com/yzy542968-jpg/wqf7023-medical-rag

**Branch:** `post-submission-improvements`

```powershell
& ".\.venv\Scripts\python.exe" scripts\build_multimodal_v5_cohort.py
& ".\.venv\Scripts\python.exe" scripts\run_multimodal_v5_retrieval.py --split confirmation --device cuda
& ".\.venv\Scripts\python.exe" scripts\build_multimodal_v5_prompt_packs.py --split confirmation
& ".\.venv\Scripts\python.exe" scripts\analyze_multimodal_v5_statistics.py
& ".\.venv\Scripts\python.exe" scripts\build_v5_artifact_manifest.py
& ".\.venv\Scripts\python.exe" scripts\build_v5_qualitative_review_materials.py
& ".\.venv\Scripts\python.exe" -m pytest -q
```

The generation and semantic-evaluation commands, model identifiers, batch sizes, and thresholds are recorded in `docs/V5_TECHNICAL_FREEZE.md`.

## Appendix D: Researcher Review and Human-Evaluation Boundary

The researcher reviewed and accepted all 24 taxonomy v1.1 proposals on 19 August 2026. This was an author/researcher qualitative review supported by deterministic case extraction and assistant-proposed coding. It was not an independent radiologist evaluation.

The earlier blinded independent-rating packages remain available as future-work artifacts but were not completed. No independent human correctness, grounding, preference, harmfulness, or inter-rater agreement result is claimed.

## Appendix E: Dashboard Demonstration Boundary

The dashboard demonstrates image-conditioned retrieval, top-ranked candidate report selection, report-grounded generation, semantic evidence checking, and trace display. For an arbitrary uploaded image, the interface should describe the action as retrieving the top-ranked candidate report from the indexed corpus. It must not claim to identify the patient's true report, diagnose the image, or access an authenticated clinical record.
