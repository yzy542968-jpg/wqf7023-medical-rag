# V16 Protocol Deviation: Reference Completeness

## Deviation

The frozen V16 confirmation protocol stated that technically eligible cases
would provide non-empty Findings and Impression references. The instantiated
568-case manifest inherited the technically eligible V10 QA frame and verified
the presence of both question types, but the implementation did not assert that
both underlying reference strings were non-empty.

A post-run freeze audit identified:

- 3,408 rows per generation arm;
- 243 empty-reference rows per arm;
- 81 affected cases;
- all 243 rows were Findings rows (81 cases x 3 evidence conditions);
- zero Impression rows had empty references.

The same references and row keys occur in both paired arms. No case was added,
removed, replaced, or regenerated after this finding.

## Effect on the primary comparison

The frozen primary analysis retains all 568 cases and all 3,408 rows per arm.
Empty-reference Findings rows receive Token-F1 zero in both arms. The V16 route
does not change Findings generation; it changes only retrieved-history
Impression generation. Therefore, the empty Findings references cannot create
the observed between-arm difference, although they lower absolute aggregate
scores.

Primary all-row result:

| Arm | Retrieved-history Token-F1 |
|---|---:|
| Base | 0.20570 |
| V16 impression gate | 0.25591 |
| Paired difference | +0.05020 [0.03973, 0.06108] |

## Post-hoc non-empty-reference sensitivity

The sensitivity retains the frozen 568-case denominator but excludes only rows
whose reference string is empty. It is explicitly post hoc and does not replace
the primary result.

| Arm | Non-empty retrieved-history row mean Token-F1 |
|---|---:|
| Base | 0.22150 |
| V16 impression gate | 0.27555 |

The case-grouped paired difference is **+0.04571**, 95% CI
**[+0.03371, +0.05763]**. The conclusion remains positive.

## Standard and clinical metrics

The frozen standard-NLG and clinical-structure aggregate results also retain the
empty-reference rows. BERTScore explicitly emitted empty-reference warnings and
assigned zero raw scores to these rows. Their presence is recorded rather than
silently filtered. Because the two arms have identical Findings outputs and
references, paired route-versus-base changes remain driven by the non-empty
retrieved-history Impression rows.

## Corrective action

The following corrective actions are allowed because they do not modify frozen
model outputs:

1. add explicit reference-completeness auditing to the paired evaluator;
2. publish counts and a non-empty-reference sensitivity;
3. update the manuscript, registry, and freeze record;
4. add a fail-fast non-empty-reference assertion to any future study protocol
   implementation before cohort instantiation.

The following are not allowed within V16: deleting the 81 cases from the
primary denominator, regenerating outputs, changing the route, or replacing the
primary metric after observing the result.

## Interpretation

This is a protocol-implementation deviation and a limitation of the absolute
report-reference scores. It does not reverse the paired V16 conclusion. It also
reinforces that the reported automated metrics are not clinical accuracy.
