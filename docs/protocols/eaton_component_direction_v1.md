# Eaton damaged-component observability: directional disagreement protocol v1

**Status (2026-07-11):** spatial roles, models, and confirmation-set commitments are frozen; `spatial_confirmation` remains unscored and unblinded. The development role triggered `STOP_PRE_ANNOTATION_STRUCTURAL_FUTILITY` before any human annotation: the entire development role holds only 28 blocks, whereas the registered gate requires at least 30 eligible blocks; the five-seed ensemble also yields only 97 unsigned disagreements covering 24 blocks, below the 250/30 gate. Version 1 therefore requests no human annotation and reports no H-B1/H-B2. The formal result is in [`../results/eaton_component_observability_v1.md`](../results/eaton_component_observability_v1.md).

## Single research question

In the Eaton fire, when street-level and overhead images yield disagreeing three-level ordinal severity predictions, **is the dominant damaged component, together with that component's assessability in each view, systematically associated with the sign of the disagreement?**

The pre-registered mechanism has exactly two expectations:

- when damage is roof-dominant and the roof is assessable only from overhead, the overhead prediction should be more severe than the street prediction;
- when damage is facade-dominant and the facade is assessable only from the street, the street prediction should be more severe than the overhead prediction.

This is a within-event spatial association and an interpretation of image evidence. It is not a causal effect, not a causal attribution to field ground truth, not cross-event extrapolation, and not evidence for next-view selection, routing, stop/defer, or a deployable selector. Even if the protocol succeeds, the only permitted central statement is: **cross-view severity disagreement in Eaton shows the pre-registered directional association with the view-observability of the damaged component.**

The frozen configuration is [`configs/eaton_component_direction_v1.json`](../../configs/eaton_component_direction_v1.json).
External hash anchors for the original configuration, summary, four-role manifests, and media ledger are in [`configs/eaton_component_direction_v1_commitments.json`](../../configs/eaton_component_direction_v1_commitments.json); hashes for the formal model outputs (predictions, metadata, checkpoints, and executing code) are in [`configs/eaton_component_direction_v1_development_run_commitments.json`](../../configs/eaton_component_direction_v1_development_run_commitments.json).

## Analysis cohort and spatial isolation

The unit of analysis is one media-only street–overhead image pair per DINS structure. Only records with both media readable, valid coordinates, and an ordinal source label are retained; `Inaccessible` is a field-access constraint and is not part of ordinal severity. The three ordered levels are fixed as:

1. `no_or_trace_damage`: `No Damage` or `Affected (1-9%)`;
2. `damaged_repairable`: `Minor (10-25%)` or `Major (26-50%)`;
3. `destroyed`: `Destroyed (>50%)`.

When one DINS structure has several attachments, only the pair whose street image has the largest pixel area is kept; ties are broken by the smallest numeric `attachment_id`, then the lexically smallest `pair_id`. Constant images, unreadable or missing media, and images with EXIF Orientation ≠ 1 are excluded before any training, so that neither blank evidence nor rotation differences between the model and the human viewer can enter. From 19,780 joined rows this yields 18,291 canonical structures; the 1,489 source exclusions are recorded row by row in the frozen audit file.

The spatial protocol uses a `0.002°` grid and seed `20260711`. Spatial blocks that share an exact street/overhead SHA-256 are first merged into indivisible dependency components; roles are then assigned and a cross-role `50 m` buffer is enforced. The buffer removes 3,234 canonical pairs, leaving 15,057:

| Protocol role | Pairs / structures | Spatial blocks | Permitted use |
| --- | ---: | ---: | --- |
| `model_fit` | 6,755 | 349 | training single-view models only |
| `model_validation` | 1,284 | 30 | early stopping, temperature calibration |
| `study_development` | 1,534 | 28 | development predictions, blinded annotation workflow, pre-unblinding gate |
| `spatial_confirmation` | 5,484 | 159 | one-shot same-event spatial confirmation; scoring currently prohibited |

The frozen pairwise audit between any two roles reports: `pair_id` overlap = 0, DINS structure overlap = 0, spatial block overlap = 0, street/overhead exact-media SHA overlap = 0, and actual 512×512 crop overlap on the same orthomosaic tile = 0. Every cross-role nearest distance is strictly greater than 50 m; the minimum across the full audit is about 50.004 m, and the minimum same-tile crop-centre distance is about 669.0 pixels. Role proportions were specified before buffering, so the final proportions in the table need not equal the configured 0.5/0.1/0.1/0.3 exactly.

The confirmation manifest has 5,484 rows and 159 blocks; its SHA-256 commitment is:

```text
3f7cd19f6afe55bbb761c7a44e6e6dfeea31744fe84d0665c597388629cdcbfe
```

The frozen artifacts and the full distance audit live under [`data/splits/eaton_component_direction_v1/`](../../data/splits/eaton_component_direction_v1/) (local, not tracked). This confirmation set is a **same-event spatial confirmation**, not an external-event confirmation.

## Prediction generation and the ensemble unit

One `ResNet-18` single-view three-class model is trained per view (street, overhead). The protocol fixes image size 224, 15 epochs, patience 4, balanced class weighting, learning rate `1e-4`, and seeds `42, 123, 456, 789, 1011`.

Per-seed, per-view temperatures may be fitted only on `model_validation`. For each pair, the temperature-calibrated class probabilities of the five seeds of the same view are averaged first, then `argmax` is taken in the fixed class order. After this step every pair has exactly one street prediction and one overhead prediction.

**Model seeds are not independent observations.** The five seed rows must not be treated as five samples to inflate sample size, block count, or significance. The unit of inference is always the seed-ensembled structure/pair, resampled by spatial block.

Per-view Eaton/Altadena predictions do exist from the legacy `altadena_3class` experiment, so the earlier statement that "no Eaton predictions exist" was inaccurate. Those predictions, however, come from consumed legacy pilot/analysis splits that were not spatially isolated under this protocol; they cannot substitute for the fresh role-isolated predictions of this protocol and cannot be used for registered tests or confirmation decisions. At most they serve claim-free code sanity checks or pilot yield estimates.

## Human annotation construct

Annotation packets are generated only for **all unique pairs within the designated role on which the ensembles disagree (street vs overhead)**; sampling may use "disagrees or not" but never the signed direction. Annotators see only an opaque `pair_id`, the two media files, the media SHA-256 values, the protocol identifier, and the fields to fill. They must not see:

- the native or derived severity label;
- any view's prediction, probability, or confidence;
- the signed disagreement direction;
- the spatial block or protocol role;
- the other annotator's answers.

`component_dominance` takes exactly one of the following values per pair:

| Value | Decision rule |
| --- | --- |
| `roof` | Taking both images together, recognisable damage evidence is dominated by roof/roofing components |
| `facade` | Recognisable damage evidence is dominated by vertical facade components such as exterior walls, doors, and windows |
| `mixed` | Both roof and facade show substantive damage evidence and no single dominant component can reasonably be assigned |
| `none_detected` | No recognisable roof/facade damage is found in the current image evidence; this does not mean no damage on site |
| `unknown` | Coverage, occlusion, scale, or quality prevent judging which damaged component dominates |

`street_roof_assessability`, `street_facade_assessability`, `overhead_roof_assessability`, and `overhead_facade_assessability` are also filled separately:

- `assessable`: the view offers enough coverage and quality of that component to judge its damage state;
- `not_assessable`: the component is out of frame, occluded, too small in pixels, or of insufficient quality to judge;
- `indeterminate`: on the boundary between the two, and the annotator cannot decide reliably.

"Assessable" does not mean "damaged", and "not assessable" does not mean "undamaged". DINS construction/material/presence/exposure fields and `dins_wherefirestartedonstructure` must not stand in for damage dominance; component labels produced by a VLM or any other model must not stand in for the human reference.

The whole packet is completed independently by two fixed raters. Any disagreement on a registered field must go to a third-party adjudicator; the final provenance-bearing reference must retain both raw ratings, the adjudication record, the protocol hash, and both media hashes, and must not silently drop pairs through an inner join.

## Mechanism eligibility, H-B1, and H-B2

Let

```text
Y_i = 1[overhead_prediction_i > street_prediction_i]
```

Only pairs on which the ensemble predictions disagree enter the directional analysis. Mechanism eligibility is fixed before unblinding as:

- `roof` stratum: `component_dominance=roof`, with `overhead_roof_assessability=assessable` and `street_roof_assessability=not_assessable`;
- `facade` stratum: `component_dominance=facade`, with `street_facade_assessability=assessable` and `overhead_facade_assessability=not_assessable`.

Records labelled `mixed`, `none_detected`, or `unknown`, and records whose required assessability is `indeterminate`, are excluded from the primary analysis.

### H-B1 (primary)

The primary estimand is the severity-standardised directional contrast:

```text
Delta_std = P_std(Y=1 | mechanism-eligible roof disagreement)
          - P_std(Y=1 | mechanism-eligible facade disagreement)
```

Both strata are standardised to the frozen three-level target distribution of all mechanism-eligible disagreements. Intervals use a whole-spatial-block bootstrap with 10,000 replicates and seed `20260711`. If any positively weighted severity stratum lacks roof or facade common support, the primary estimate cannot be interpreted as passing.

### H-B2 (supportive)

H-B2 is the direction-consistent share among mechanism-eligible disagreements: `Y=1` in the roof stratum, or `Y=0` in the facade stratum. Additionally reported:

- mechanism eligibility coverage / all disagreements;
- full-disagreement explained fraction = direction-consistent eligible disagreements / all disagreements.

H-B2, coverage, and the explained fraction are supportive results and **cannot rescue a failed H-B1**.

## Pre-unblinding gate

Before any signed-direction statistic is inspected, all of the following must hold:

- every registered annotation field has raw agreement ≥ 0.75 and Cohen's kappa ≥ 0.67;
- mechanism-eligible disagreements ≥ 250;
- roof-eligible ≥ 75 and facade-eligible ≥ 75;
- eligible spatial blocks ≥ 30, with ≥ 15 blocks each for roof and facade;
- frozen simulated power ≥ 0.80 for the minimum meaningful effect `Delta_std = 0.20`.

If any item fails, `STOP_UNDERPOWERED_OR_UNRELIABLE_CONSTRUCT` is recorded; H-B1/H-B2 may not be inspected or interpreted, and the stop may not be read as "the mechanism does not exist". The development role exists to verify annotation reliability, yield, power, and the complete workflow; the confirmation role may not be scored or directionally annotated until the models, ensemble, codebook, gate, estimator, and all hashes are frozen.

## One-shot confirmation decision

Only after the pre-unblinding gate passes may the registered analysis be executed once on `spatial_confirmation`. `GO_FOR_REGISTERED_EATON_IMAGE_MECHANISM` requires all of:

1. `Delta_std >= 0.20`;
2. the 95% spatial-block bootstrap lower bound of H-B1 > 0;
3. roof positive-direction share > 0.5;
4. facade positive-direction share < 0.5;
5. mechanism eligibility coverage ≥ 0.30.

Otherwise the conclusion is fixed as `NO_GO_FOR_REGISTERED_EATON_IMAGE_MECHANISM`. Supportive estimands cannot change that decision. Whether GO or NO-GO, the outcome applies only to the frozen same-event spatial confirmation cohort of Eaton.

## Current legal boundary

The spatial manifests, media hash ledger, role isolation, fresh development predictions, and confirmation hash are in place, but they are not a mechanism result. The formal models are produced by [`scripts/run_eaton_component_models.py`](../../scripts/run_eaton_component_models.py); the direction-free capacity gate is executed by [`scripts/run_eaton_preannotation_futility_gate.py`](../../scripts/run_eaton_preannotation_futility_gate.py); the registered statistics are implemented in [`crossview_conflict/analysis/eaton_component_direction.py`](../../crossview_conflict/analysis/eaton_component_direction.py).

Version 1 stopped legally before human annotation, because no annotation can raise the 28 total blocks to 30 eligible blocks, and the 97 total disagreements cannot reach 250 mechanism-eligible disagreements. H-B1/H-B2 therefore have no legally reportable values, and the confirmation role may not be accessed. This boundary may not be crossed with DINS proxies, legacy predictions, VLM labels, or synthetic annotations; the current state is a **structurally underpowered STOP**, not a positive or negative mechanism result.
