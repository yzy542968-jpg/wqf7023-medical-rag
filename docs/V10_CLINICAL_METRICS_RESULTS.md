# V10 Automated Clinical Graph Metrics

Status: prespecified secondary F1RadGraph analysis complete; no retuning.

## Results

| Condition | Entity F1 | Entity + relation F1 | Complete F1 |
|---|---:|---:|---:|
| G0 target image | 0.113360 | 0.102817 | 0.082649 |
| G1 R4 whole-report RAG | 0.145481 | 0.129053 | 0.105065 |
| **G2 R5 whole-report RAG** | **0.149691** | **0.133924** | **0.110526** |
| G3 selective R5 RAG | 0.149653 | 0.133977 | 0.110408 |

G2-minus-G0 was positive for all graph levels. Complete F1 increased by
0.027877 (95% case-grouped bootstrap CI 0.019773 to 0.036393), consistent with
the Token-F1 evidence that historical cases improved source-report alignment.

G2-minus-G1 was 0.004210 for entity F1 (95% CI -0.001872 to 0.010337),
0.004871 for entity-plus-relation F1 (95% CI -0.000811 to 0.010745), and
0.005461 for complete F1 (95% CI 0.000050 to 0.010886). Only the complete score
had a lower bound narrowly above zero. This is reported as limited secondary
evidence, not broad confirmation that R5 improves every downstream semantic
metric.

G3 and G2 were nearly identical on graph metrics. Selective evidence use did
not produce a clear aggregate semantic advantage under the frozen operating
point.

## Availability boundary

A compatible F1CheXbert implementation was not installed locally. No surrogate
metric was relabelled as F1CheXbert. F1RadGraph remains an automated graph
overlap measure and is not physician-adjudicated correctness or safety.

## Artifact trail

- Source QA rows SHA-256: `0e82b3cf5d3913fdac82f49b6742451cf095849cad88caa2c5bedb070f793944`
- Local per-row metric file SHA-256: `e07f5e0144ebfa23c2c0ebebcd0e675ce7e35a3d223fad9386c8ab4d81fe901b`
- RadGraph model type: `modern-radgraph-xl`
- Rows scored: 4,544
