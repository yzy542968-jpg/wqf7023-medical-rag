# V16 Final Technical Freeze

## Freeze status

V16 is the final integrated technical study for the thesis. It combines the
Validation-selected V12 retrieval method with the Validation-selected V16
section-aware generation adaptation and evaluates both on the existing V10
duplicate-cluster-disjoint Test partition.

The confirmation protocol was committed at
`fa26716f0a549176d5b83a848cfb3d736b7485cc` before the V12/V16 Test outputs were
generated. Retrieval results were committed at
`4f152bbcb9067953edc04001a6d4b82ef5ff09da`; generation results were committed
at `3fba297de2d541acffc89e0b7e0001281a63b9d5`.

The Test partition was previously evaluated for V10 and is not a globally
untouched project holdout. V12 LambdaMART, the V16 adapter, and the V16 route
were not evaluated on Test before the protocol. V16 is therefore described as
a held-out method confirmation, not as formal preregistration or a project-wide
never-seen test.

## Frozen task contract

For a target case, the system receives the chest radiograph, indication, and a
question. The target report is hidden. Retrieval searches only technically
eligible Train-bank historical image-report cases from other duplicate
clusters. The selected historical evidence is passed to the generator with
case and section provenance. The hidden target Findings or Impression section
is used only for automated evaluation.

## Frozen integrated method

```text
Target image + indication + question
    -> BM25 + MedCPT + MedSigLIP retrieval sources
    -> deterministic RRF Top-200 candidate set
    -> frozen 17-feature LambdaMART reranker
    -> Top-3 historical cases with fact-level provenance
    -> base MedGemma for Findings
    -> V16 QLoRA MedGemma for retrieved-history Impression
    -> compact answer plus deterministic provenance assembly
```

No target report content is available to retrieval or generation. Qrels and
reference answers are report-derived evaluation constructs only.

## Frozen data boundary

- OpenI/IU-Xray source cases: 3,851.
- Duplicate clusters: 3,013.
- Existing V10 split: Train 2,510; Calibration 383; Validation 384; Test 574.
- Implemented technically eligible Test frame: 568 cases. A post-run audit found
  81 cases with empty Findings references; the deviation and non-empty-reference
  sensitivity are recorded separately.
- Technically eligible Train historical bank: 2,506 cases.
- Test spectrum: 195 report-indexed normal, 359 report-indexed abnormal, and 14
  report-index indeterminate cases.
- Test manifest: `data/splits/v16/v16_confirmation_manifest_568.jsonl`.
- Case-ID and duplicate-cluster disjointness are verified.
- Identifier-based patient independence cannot be verified from the processed
  OpenI release and is not claimed.

No frozen case was replaced. Technical interruptions were resumed under the
same configuration; no outcome-driven rerun or case deletion occurred.

## Frozen retrieval result

| System | Combined-qrel nDCG@10 |
|---|---:|
| V10 R5 comparator | 0.55313 |
| V12 RRF Top-200 plus LambdaMART | **0.61590** |
| Difference | **+0.06277** |
| 95% case-grouped CI | **[+0.05460, +0.07082]** |

The label-only difference is +0.03928 [0.02450, 0.05443], and the fact-only
difference is +0.01326 [0.00405, 0.02243]. Candidate RRF alone and full-bank
LambdaMART are negative mechanism controls; the confirmed gain depends on both
the multi-source Top-200 frame and learned reranking.

## Frozen generation result

| Retrieved-history arm | Token-F1 |
|---|---:|
| Frozen base MedGemma | 0.20570 |
| V16 impression-gated route | **0.25591** |
| Difference | **+0.05020** |
| 95% case-grouped CI | **[+0.03973, +0.06108]** |

The V16 route also exceeds no history (0.16922) and random history (0.19608).
Contract and provenance validity are 100%. The token-ceiling rate decreases
from 0.87852 to 0.56602. All six standard NLG metric intervals favor the route,
and RadGraph complete F1 improves by +0.02687 [0.01833, 0.03562].

CheXbert is mixed: micro-F1 changes by -0.00545 with an interval crossing zero,
while reference-positive recall decreases by -0.01081 with a fully negative
interval. This secondary negative signal remains part of the frozen result.

The all-row primary result retains 243 empty Findings-reference rows from 81
cases. These rows are identical zero-reference comparisons in both arms and do
not generate the Impression-only route difference, but they lower absolute
scores. The post-hoc non-empty-reference sensitivity remains positive at
+0.04571 [0.03371, 0.05763].

## Final evidence interpretation

V16 supports the following bounded conclusions:

1. the multi-source candidate frame plus learned reranking improves automated
   report-derived historical-case ranking over the strong V10 R5 comparator;
2. retrieved historical cases improve downstream report-reference consistency
   over both no-history and random-history controls;
3. section-aware QLoRA adaptation improves Impression generation while
   preserving the stronger base Findings route;
4. compact output and deterministic provenance can remain valid while
   substantially reducing token-ceiling events;
5. improvement is not uniform across every automated clinical-label metric.

V16 does not demonstrate physician-rated clinical similarity, diagnostic
accuracy, safety, patient benefit, verified patient-level independence,
external-dataset generalization, or readiness for deployment.

## Frozen artifact registry

| Artifact | SHA-256 |
|---|---|
| V16 confirmation protocol | `bbc586daa61caa8dc3ef3c2f1c4a6cf47ce2bdc353272f54f4d4709063a9e10d` |
| V16 confirmation manifest | `68f0cb7e836c1b92e6210f8d3ea9c3dfc2e7b99c7198f896608c66647da9de11` |
| V16 Test case-ID list | `13b34fb825ec2d7a11ddcab070c482c48930829d4821dcd88328fdbceb1536c3` |
| V16 aggregate retrieval result | `2367ec141f39c59b8200f01f2c6e1c1d14a3de765c4695d007faf68c739e612a` |
| Base generation rows | `9f026288281174ceb9ec59a219cd08f8a2841edbb6d0baef015884f2c345fb6f` |
| QLoRA generation rows | `177a7358ea95b52a32d4ada2f47bd53bf90288c2d7543a8ce0f8f37b8b039ddb3` |
| Impression-gated rows | `e9871a21abc381797af7dc0649a0d29c017ab9da7a7b14ef2ee93c36d56b35ce8` |
| Primary paired evaluation with completeness audit | `7a8ba6f03ad514ae3d4e66476cb293ea2439a01a4418875aae02363c33e8425e` |
| Clinical-metric evaluation | `95461571c2e5ddb3961d7bc91611673c89cd1b10a30ddd598c40ca8284acd462` |
| Standard-NLG evaluation with completeness audit | `10f539a36f6c9459537395e798cfddf1be961d52e4ac9276a6d913604479d826` |

Large generated rows, model assets, metric caches, report text, and image pixels
remain local. Version-controlled aggregate artifacts contain no report text or
image pixels.

## Post-freeze rules

After this freeze, permitted work is limited to deterministic audits, manuscript
and dashboard integration, formatting, release packaging, and correction of
documented software defects that do not alter the frozen scientific outputs.

The following are prohibited within V16: model retraining, Test-driven routing
or prompt changes, threshold or metric substitution, qrel changes, case
deletion/replacement, selective result suppression, and undocumented reruns.
Any future model or data extension must receive a new version and must not
overwrite the V16 artifacts.

Independent blinded clinical review and authorized patient-level MIMIC-CXR
replication remain Future Work.
