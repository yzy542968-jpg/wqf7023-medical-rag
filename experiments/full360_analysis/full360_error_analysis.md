# Full360 Automated Error Analysis

This report is generated from the full 360-question Qwen2.5-1.5B outputs. It is an automatic analysis and should be followed by manual annotation.

## System-Level Summary

| System | N | Token-F1 | Top-1 Hit | Retrieved Hit | Evidence Support | Revision | Abstention | Unsupported Sentence Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LLM-only Qwen2.5-1.5B | 360 | 0.091 | 0.000 | 0.000 | N/A | N/A | N/A | N/A |
| Report-RAG BM25 Qwen2.5-1.5B + checker | 360 | 0.093 | 0.231 | 0.383 | 0.118 | 0.992 | 0.681 | 0.819 |
| Case-RAG BM25 top-1 Qwen2.5-1.5B + checker | 360 | 0.139 | 0.231 | 0.383 | 0.374 | 0.833 | 0.325 | 0.674 |
| Case-RAG Hybrid top-1 Qwen2.5-1.5B + checker | 360 | 0.145 | 0.242 | 0.422 | 0.305 | 0.886 | 0.419 | 0.711 |

## Question-Type Summary

| System | Question Type | N | Token-F1 | Retrieved Hit | Evidence Support | Abstention |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| case_bm25_top1 | abnormality_summary | 120 | 0.072 | 0.133 | 0.334 | 0.258 |
| case_bm25_top1 | findings_from_indication | 120 | 0.165 | 0.508 | 0.346 | 0.433 |
| case_bm25_top1 | impression_from_indication | 120 | 0.181 | 0.508 | 0.442 | 0.283 |
| case_hybrid_top1 | abnormality_summary | 120 | 0.072 | 0.250 | 0.310 | 0.292 |
| case_hybrid_top1 | findings_from_indication | 120 | 0.174 | 0.517 | 0.178 | 0.592 |
| case_hybrid_top1 | impression_from_indication | 120 | 0.188 | 0.500 | 0.426 | 0.375 |
| llm_only | abnormality_summary | 120 | 0.071 | 0.000 | N/A | N/A |
| llm_only | findings_from_indication | 120 | 0.145 | 0.000 | N/A | N/A |
| llm_only | impression_from_indication | 120 | 0.057 | 0.000 | N/A | N/A |
| report_bm25 | abnormality_summary | 120 | 0.044 | 0.133 | 0.065 | 0.775 |
| report_bm25 | findings_from_indication | 120 | 0.186 | 0.508 | 0.208 | 0.450 |
| report_bm25 | impression_from_indication | 120 | 0.050 | 0.508 | 0.081 | 0.817 |

## Representative Cases

### High-support success case

- System: Case-RAG BM25 top-1 Qwen2.5-1.5B + checker
- QID: `CXR2721_impression`
- Question type: `impression_from_indication`
- Top-1 hit: True; retrieved hit: True
- Token-F1: 0.857
- Evidence support: 1.000
- Revised: False; abstained: False

Question: For a chest X-ray case with the indication 'XXXX-year-old, preop carotid endarterectomy.', what is the radiology impression?

Reference: Chronic changes without acute cardiopulmonary findings.

Answer: ** Radiology impression: Chronic changes without acute cardiopulmonary findings.

### High-F1 but low-support case

- System: Report-RAG BM25 Qwen2.5-1.5B + checker
- QID: `CXR533_impression`
- Question type: `impression_from_indication`
- Top-1 hit: True; retrieved hit: True
- Token-F1: 0.735
- Evidence support: 0.200
- Revised: True; abstained: False

Question: For a chest X-ray case with the indication 'XXXX-year-old female, rule out dissection.', what is the radiology impression?

Reference: Increasing prominence of the superior mediastinum may be secondary to enlarging thyroid mass. With increasing tortuosity of the thoracic aorta and concern for dissection, chest CT is recommended for further evaluation.

Answer: With increasing tortuosity of the thoracic aorta and concern for dissection, chest CT is recommended for further evaluation.

### Retrieval miss with poor answer

- System: Report-RAG BM25 Qwen2.5-1.5B + checker
- QID: `CXR1027_summary`
- Question type: `abnormality_summary`
- Top-1 hit: False; retrieved hit: False
- Token-F1: 0.000
- Evidence support: 0.000
- Revised: True; abstained: True

Question: What does the chest X-ray report say about Costophrenic Angle, Technical Quality of Image Unsatisfactory?

Reference: 1. Limited chest radiograph examination without demonstration of an acute intrathoracic abnormality.

Answer: The retrieved report evidence is insufficient to answer this question.

### Heavy revision case

- System: Report-RAG BM25 Qwen2.5-1.5B + checker
- QID: `CXR1027_impression`
- Question type: `impression_from_indication`
- Top-1 hit: True; retrieved hit: True
- Token-F1: 0.595
- Evidence support: 0.154
- Revised: True; abstained: False

Question: For a chest X-ray case with the indication 'XXXX-year-old female with a history of an elevated XXXX-XXXX.', what is the radiology impression?

Reference: 1. Limited chest radiograph examination without demonstration of an acute intrathoracic abnormality.

Answer: Based on the provided radiology reports and findings: **CXR1027:** Limited chest radiograph examination without demonstration of an acute intrathoracic abnormality. Negative for pneumothorax, pleural effusion.

### Hybrid top-1 representative failure

- System: Case-RAG Hybrid top-1 Qwen2.5-1.5B + checker
- QID: `CXR1054_impression`
- Question type: `impression_from_indication`
- Top-1 hit: False; retrieved hit: True
- Token-F1: 0.000
- Evidence support: 0.250
- Revised: True; abstained: False

Question: For a chest X-ray case with the indication 'History of CHF', what is the radiology impression?

Reference: No acute cardiopulmonary abnormality.

Answer: The right lung appears clear.
