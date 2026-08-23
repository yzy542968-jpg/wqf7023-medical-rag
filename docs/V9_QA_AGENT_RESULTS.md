# V9 QA and Bounded-Agent Results

## Evaluation frame

The frozen downstream evaluation used all 685 V9 Test cases with complete
findings and impression references. Two fixed questions per case produced
1,370 questions and 5,480 MedGemma generations. Every system received one
target chest radiograph, clinical indication, and question. The RAG systems
also received three other-patient reports selected by a frozen retriever.

## QA results

| System | Token-F1 | Complete JSON |
|---|---:|---:|
| G0 target image, no retrieval | 0.145559 | 42.04% |
| G1 BM25 report RAG | 0.147947 | 39.42% |
| G2 fixed multimodal RAG | 0.179090 | 46.50% |
| **G3 learned multimodal RAG** | **0.184803** | **57.23%** |

The prespecified primary comparison was:

```text
G3 minus G0 Token-F1: +0.039244
95% case-bootstrap CI: [+0.032572, +0.045745]
cases: 685; iterations: 10,000
```

The primary criterion passed. The predefined 262-case project-history-
untouched sensitivity subset also favored G3 over G0 by `+0.047187`, 95% CI
`[+0.034911,+0.059704]`.

Secondary comparisons show the mechanism more precisely:

```text
G1 BM25 minus G0:      +0.002388  CI [-0.003189,+0.008120]
G2 fixed minus G0:     +0.033532  CI [+0.027790,+0.039502]
G3 learned minus G0:   +0.039244  CI [+0.032770,+0.045962]
G3 learned minus G2:   +0.005713  CI [-0.000958,+0.012280]
G3 learned minus G1:   +0.036856  CI [+0.030410,+0.043524]
```

Therefore multimodal similar-case RAG improved report-reference consistency
over no retrieval and text-only RAG. The learned reranker was superior to
image-only retrieval for the primary retrieval metric, but its downstream QA
advantage over fixed multimodal RAG was numerical rather than statistically
resolved.

The gain was larger for impression questions (`+0.059903`) than findings
questions (`+0.018585`). It remained positive in report-indexed normal cases
(`+0.060784`) and abnormal cases (`+0.025570`). These subgroup comparisons are
prespecified sensitivity analyses, not separate confirmatory families.

## Bounded agent

G4 checked only statements presented as historical support. It did not claim
to verify findings from the target image.

```text
G3 automated unsupported historical rate: 16.42%
G4 final unsupported historical rate:       0.00%
difference:                                -16.42 percentage points
95% case-bootstrap CI:                    [-18.47,-14.45] points
retry rate:                                16.42%
historical-evidence abstention rate:       15.99%
historical-support revision rate:          17.30%
mean retrieval calls:                       1.164
G4 Token-F1:                                0.184803
```

G4 met the frozen unsupported-history reduction and QA noninferiority rules.
The zero final unsupported rate is partly structural: after one failed backup
route, the agent removes the historical-support field and citations. The
result supports auditable claim suppression, not improved image diagnosis.

## Limitations

Absolute Token-F1 remains low, especially for impression questions. MedGemma
often reached the 192-token ceiling or returned non-strict JSON; only 57.23%
of G3 rows contained a fully valid requested object. The tolerant parser kept
all outputs rather than selectively regenerating failures. Token-F1 rewards
reference wording overlap and is not clinical correctness. The NLI checker is
automated and can over-reject or under-detect claims. No radiologist evaluated
the answers, images, retrieved similarity, or agent decisions.

Runtime for the 5,480 local generations was 13,496 seconds at 0.406 records/s,
with 5,184.5 MiB peak allocated GPU memory on the RTX 5070 Laptop GPU.

