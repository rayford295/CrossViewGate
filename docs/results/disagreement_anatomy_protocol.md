# RQ1 disagreement-anatomy protocol (development)

## Claim scope

This document specifies a **development/exploratory** analysis of the *causal
structure of cross-view severity disagreement*: how much of it is systematically
produced by view-conditional attestability (what each view can see), versus
residual noise. It supports the attestability revision
([`2026-07-10_crossviewguard_attestability_revision.md`](../2026-07-10_crossviewguard_attestability_revision.md)).

- All Study A results are **exploratory**. The analysis reads only
  `gate_fit` and `risk_calibration` role rows from the p0.2-v1 per-sample
  evidence exports. **`final_test` rows are never read.** A single confirmatory
  pass on `final_test` is allowed only after this protocol and the analysis
  code are frozen in `main`.
- Labels are used **for evaluation attribution only** (which view was correct),
  never to fit any model that later claims out-of-role performance. The one
  fitted object (H-A3 logistic probe) is fitted on `gate_fit` and evaluated on
  `risk_calibration`.
- Spatial dependence: all interval estimates use a cluster bootstrap that
  resamples `spatial_block_id` clusters with replacement. Per-seed statistics
  are reported for all five seeds; no cross-seed pooling that hides seed
  variance.

## Study A — CVIAN directional anatomy (executable now)

**Inputs.**
`outputs/analysis/reliability_gate_ian_spatial_v1/predictions/ian_original_seed{42,123,456,789,1011}_{gate_fit,risk_calibration}.csv`
(schema `p0.2-v1`). Input file SHA-256 fingerprints are recorded in the output
manifest.

**Definitions.**

- *Disagreement row*: `street_prediction != remote_prediction`.
- *Signed direction*: `remote_prediction - street_prediction` (positive =
  overhead reports more severe).
- *Decidable disagreement*: exactly one of the two views equals `target`.
- *Visibility features*: `building_ratio`, `center_building_ratio`,
  `center_minus_global`, `centroid_distance_norm` (pre-existing p0.2-v1
  columns; no new feature extraction).

**Hypotheses and tests.**

- **H-A1 (directional asymmetry).** Among disagreement rows, the share with
  positive signed direction differs from 0.5.
  *Test:* share + 95% cluster-bootstrap CI per seed.
  *Development criterion:* CI excludes 0.5 in >= 4/5 seeds, same direction.
- **H-A2 (visibility conditions correctness).** Among decidable disagreements,
  street-correct rows have higher street building visibility than
  remote-correct rows.
  *Test:* difference in median `center_building_ratio`
  (street-correct minus remote-correct) with 95% cluster-bootstrap CI;
  rank-biserial effect size reported alongside.
  *Development criterion:* CI excludes 0 in >= 4/5 seeds, positive direction.
- **H-A3 (explainable share).** Visibility features predict which view is
  correct on held-out-role rows.
  *Test:* logistic probe `street_correct ~ visibility features`, fitted on
  `gate_fit` decidable rows, AUROC evaluated on `risk_calibration` decidable
  rows. Two ablations decompose the signal: visibility-only vs
  confidence-only (`street_confidence`, `remote_confidence`) vs combined.
  *Development criterion:* visibility-only AUROC > 0.5 with cluster-bootstrap
  CI excluding 0.5 in >= 3/5 seeds. The combined-vs-confidence-only margin
  estimates what visibility adds beyond what the reliability gate already
  uses.

**What Study A cannot show.** CVIAN has no component-level damage fields, so
Study A can establish *that* visibility structures disagreement, not *which
building component* drives it. Component attribution is Study B.

**Outputs.** `outputs/analysis/disagreement_anatomy_ian_spatial_v1/`:
`per_seed_statistics.csv`, `summary.json` (with input fingerprints, role
whitelist, bootstrap settings), and a Markdown summary block for the results
document. The runner refuses to start if any input path contains
`final_test`.

```powershell
python scripts/analyze_disagreement_anatomy.py `
  --predictions-dir outputs/analysis/reliability_gate_ian_spatial_v1/predictions `
  --seeds 42,123,456,789,1011 `
  --output-dir outputs/analysis/disagreement_anatomy_ian_spatial_v1
```

## Study B — Eaton component anatomy (feasibility executed; inference blocked)

**Input status.** The P0.4 `dins_field_join` artifacts are locally available
and audited: 19,776/19,780 attachment rows match 18,411 DINS structures. The
component-like fields record construction/material/exposure attributes, not
component-damage presence or dominance. Eaton per-view severity predictions
and a frozen Eaton `spatial_block_id` are not available. See
[`disagreement_anatomy_eaton_v1.md`](disagreement_anatomy_eaton_v1.md).

**Prerequisite.** The P0.4 supplement in the revision memo: freeze an
**image-visible field whitelist** before any test is run. Candidate fields
(roof, eaves, siding, window pane, deck/porch) are individually ruled
visible/invisible per view at source resolution; excluded fields may appear
only as stress tests. The whitelist is frozen in
[`configs/eaton_image_visible_fields_v1.json`](../../configs/eaton_image_visible_fields_v1.json).
Its view labels are *in-principle, conditional visibility* labels; they do not
turn inspector records into image-level observations or damage-component
labels.

**Hypotheses.**

- **H-B1 (component-directional coupling).** Buildings whose recorded damage
  is roof-dominated show overhead-more-severe disagreement; facade-dominated
  damage shows street-more-severe disagreement.
  *Test:* signed-direction distribution conditioned on component-dominance
  strata, cluster-bootstrapped over Eaton spatial groups.
- **H-B2 (explained share).** The fraction of disagreement rows whose
  direction is consistent with component-visibility strata, versus the
  residual treated as noise/annotation error. This fraction is the headline
  RQ1 quantity: it bounds how much disagreement is *information*.

**Claim rule.** Until H-B1/H-B2 are executed, the phrase "disagreement is
information, not error" must not appear as a conclusion in any manuscript;
Study A alone supports only the weaker "disagreement is structured by
visibility".

**Current execution state.** The independent runner
[`scripts/analyze_eaton_component_anatomy.py`](../../scripts/analyze_eaton_component_anatomy.py)
completed artifact, construct, media, and restricted-proxy power diagnostics.
It deliberately produced no H-B1/H-B2 estimates. It requires (a) genuine
street/remote ordinal severity predictions with spatial blocks and (b) a
separate provenance-bearing `damage_dominance` reference. It rejects
construction/material semantics used as dominance and refuses development
inputs marked `final_test`.

## Consumed-data ledger

| Role | Status after Study A |
| --- | --- |
| `gate_fit` | consumed for exploratory anatomy + H-A3 fitting |
| `risk_calibration` | consumed for exploratory anatomy + H-A3 evaluation |
| `final_test` | **untouched**; reserved for one frozen confirmatory pass |
