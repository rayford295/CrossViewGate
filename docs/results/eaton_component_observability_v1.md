# Eaton damaged-component observability v1 — development result

## Decision

**`STOP_PRE_ANNOTATION_STRUCTURAL_FUTILITY`.** This is an outcome-blind
development decision, not a negative H-B1/H-B2 result and not evidence that the
proposed mechanism is absent.

The frozen `study_development` role contains 28 spatial blocks, while the
pre-registered gate requires at least 30 mechanism-eligible spatial blocks.
Mechanism-eligible blocks are necessarily a subset of all role blocks, so no
possible human annotation can make this role pass the registered gate. The
registered workflow therefore stops before human annotation, signed-direction
construction, H-B1/H-B2 estimation, or any access to `spatial_confirmation`.
The independently hash-bound ensemble audit strengthens the same decision: all
97 unsigned disagreements occupy only 24 blocks, below the registered upper
bounds of 250 eligible disagreements and 30 eligible blocks.

## Single research question

Among Eaton structures whose street and overhead single-view models disagree
on three-level ordinal severity, is the signed disagreement direction
systematically associated with the dominant damaged component and whether that
component is assessable in each view?

The protocol tests only this same-event image-evidence mechanism. It does not
test a learned selector, routing, stop/defer policy, causal damage attribution,
or cross-event generalization.

## Frozen cohort and trust anchors

| Artifact | SHA-256 / count |
| --- | --- |
| Protocol config | `79bfc18f3e8d690ace9d460e7fada1c23c032e75da9dc18006c96e0033eadbb9` |
| Protocol summary | `e8894ec1862762c549673622411d0986e3d0e5482a536b1a078ce9316f28926e` |
| Joined Eaton manifest | `c413c272a761b0c25269bab1158438b28f2f3727cb8bfdd7edab42d0b281ffcf` |
| `study_development` manifest | `ae90bae3676911325df1086ab582a80f5f481e09c35e32b547e221eb10ca98f7`; 1,534 pairs / 28 blocks |
| `spatial_confirmation` manifest | `3f7cd19f6afe55bbb761c7a44e6e6dfeea31744fe84d0665c597388629cdcbfe`; 5,484 pairs / 159 blocks |
| Media hash ledger | `8f539f045cd3b986e72f76159bffb32b514889e1b99eeeaa3283f12340c3dbda` |
| Pre-run commitment file | `8b90249ac123fccac61d0b1cf0e5f617a6aa69bc5b8fb75b36706427669af230` |
| Fresh prediction CSV | `5a99017c38b5600b9c08c380156682c572cabc4ba8267762535055caf2b65c25` |
| Prediction metadata | `8c68a6ea830a7fa1820aa0f3edd3fa6cfc4e7291761c57fcec10abb3e4b3d1e9` |
| Post-run commitment | `226a39067f292685930461a2041446ed5e14b980c323e4593c50e45ce6aa0c31` |
| Core futility artifact | `24dd4c4e43a539fc2673b28ca68abed9c2a5df81a3c9d7633b181772af655896` |

The tracked commitment file is
[`configs/eaton_component_direction_v1_commitments.json`](../../configs/eaton_component_direction_v1_commitments.json).
The post-run prediction/checkpoint/code commitments are in
[`configs/eaton_component_direction_v1_development_run_commitments.json`](../../configs/eaton_component_direction_v1_development_run_commitments.json),
and the path-free machine-readable STOP artifact is
[`eaton_component_preannotation_futility_v1.json`](artifacts/eaton_component_preannotation_futility_v1.json).
The complete cohort construction and annotation rules are in
[`docs/protocols/eaton_component_direction_v1.md`](../protocols/eaton_component_direction_v1.md).

## Fresh model run

The formal run trains street-only and overhead-only ResNet-18 models for seeds
`42, 123, 456, 789, 1011`, calibrates each model only on `model_validation`, and
exports all 1,534 `study_development` pairs. Seed rows are averaged before
forming one ensemble prediction per view and are never treated as independent
observations.

The completed sidecar has status `PROTOCOL_DEVELOPMENT_RUN`, records ten fresh
checkpoints, and states `spatial_confirmation_read_or_scored=false`. Aggregate
quality control across the five seeds is:

| View | Validation accuracy, mean [min, max] | Validation macro-F1 | Development accuracy | Development macro-F1 | Development repairable-class F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Street | 0.942 [0.928, 0.958] | 0.684 [0.675, 0.696] | 0.947 [0.924, 0.970] | 0.686 [0.673, 0.711] | 0.126 [0.085, 0.176] |
| Overhead | 0.942 [0.924, 0.969] | 0.721 [0.708, 0.735] | 0.926 [0.906, 0.957] | 0.676 [0.663, 0.699] | 0.131 [0.057, 0.205] |

The high overall accuracy is dominated by the two endpoint classes. Only 19
development pairs are `damaged_repairable`, and its per-seed F1 is low and
unstable; this is a measurement limitation, not evidence about the registered
component-observability mechanism.

After the prediction and metadata hashes were committed, the unsigned ensemble
audit found **97 disagreements across 24 spatial blocks**. This independently
cannot reach the registered 250 mechanism-eligible disagreements or 30 eligible
blocks. No signed-direction statistic was constructed or reported.

## Why annotation and confirmation were not run

The gate failure is mathematically conclusive before annotation:

```text
maximum possible eligible development blocks <= all development blocks = 28
registered eligible-block minimum = 30
28 < 30  =>  STOP

maximum possible mechanism-eligible disagreements <= all disagreements = 97
registered mechanism-eligible minimum = 250
97 < 250  =>  STOP
```

Generating ratings cannot change that inequality. Asking two human raters and
an adjudicator to label the packet would therefore add cost without making the
registered experiment executable. The blinded packet and validator remain
tested infrastructure, but no labels from them are used as a registered result.

`spatial_confirmation` remains unscored and unannotated. H-B1, H-B2, their
confidence intervals, and the registered GO/NO-GO mechanism decision are all
**not estimable under v1**.

## Reproduction

```powershell
python scripts/run_eaton_component_models.py `
  --split-dir data/splits/eaton_component_direction_v1 `
  --output-dir outputs/eaton_component_direction_v1/development_models_protocol_v1 `
  --num-workers 4 `
  --device cuda

python scripts/run_eaton_preannotation_futility_gate.py `
  --expected-protocol-summary-sha256 e8894ec1862762c549673622411d0986e3d0e5482a536b1a078ce9316f28926e `
  --expected-prediction-csv-sha256 5a99017c38b5600b9c08c380156682c572cabc4ba8267762535055caf2b65c25 `
  --expected-prediction-metadata-sha256 8c68a6ea830a7fa1820aa0f3edd3fa6cfc4e7291761c57fcec10abb3e4b3d1e9
```

The second command consumes only complete-population identities and unsigned
prediction inequality. It does not consume annotations or construct the signed
ordinal difference.

## Next scientifically valid move

Do not reinterpret this STOP as a mechanism failure and do not rescue it by
lowering thresholds after seeing outcomes. A v2 study would need a prospectively
larger development block allocation and a justified eligible-yield/power design,
while preserving the current confirmation role as untouched for this v1 claim.
