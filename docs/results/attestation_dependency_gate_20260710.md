# RQ2/RQ3 attestation dependency gate (2026-07-10)

## Decision

**NO-GO for empirical RQ2 view × field matrix training and NO-GO for RQ3
attestation-coverage utility on the currently available data.** This is an
upstream dependency decision, not a negative RQ2/RQ3 result. No model was
trained and no label or prediction was synthesized during this audit.

The decision follows the revision's own ordering rule: RQ1 must establish a
credible view-conditional attestability signal before RQ2, and RQ3 must consume
a frozen, validated RQ2 matrix. Neither dependency currently exists:

- CVIAN Study A met zero of its three development criteria. Its four visibility
  features are not held-out predictors of which view is correct.
- Eaton Study B is inferentially blocked. Its available component-like fields
  are construction/material/exposure attributes, not component-damage presence
  or dominance. It also lacks genuine paired-view severity predictions and a
  frozen spatial split.

The only legal continuation now is annotation/data-design work, schema and
validator plumbing, power analysis after eligible counts exist, or the existing
label-cost active-view study under its own baseline claim. Label-cost utility
must not be renamed attestation-coverage utility.

## Strict dependency gate

Every item in a row must pass; a missing or ambiguous item is a fail.

| Gate | Required evidence | CVIAN | Eaton |
| --- | --- | --- | --- |
| RQ1 upstream | pre-specified evidence that visibility/component location structures disagreement | **NO-GO**: 0/3 Study A criteria | **BLOCKED**: no valid dominance reference or predictions |
| RQ2 construct | provenance-bearing component-damage state for roof/eaves/siding/window/deck; ordinal severity kept separate | **NO-GO** | **NO-GO** |
| RQ2 view target | image-level `visible/not_visible/indeterminate` and view judgment or `abstain` for each view × field | **NO-GO** | **NO-GO** |
| RQ2 design | frozen dependency groups, role isolation, field × view support and power audit | partial: groups/roles exist; support does not | **NO-GO**: groups/roles absent |
| RQ2 geometry | validated RQ2 matrix plus camera/building geometry, orientation, and occlusion references | **NO-GO**: compass/coordinates are only partial geometry | **NO-GO** |
| RQ3 upstream | frozen, calibrated, out-of-role RQ2 attestation probabilities with abstention | **NO-GO** | **NO-GO** |
| RQ3 utility/test | frozen decision-field weights, candidate views, costs, policy isolation, attestation-labeled prospective/event test | **NO-GO** | **NO-GO** |

Therefore the executable scope is:

- **CVIAN:** severity-only exploratory reliability analysis is possible and was
  already performed in Study A. It is not the declared multi-field RQ2 matrix.
- **Eaton:** media/join/construct auditing is possible. RQ1 H-B1/H-B2, RQ2, and
  RQ3 inference are not executable.
- **Synthetic fixtures:** schema, leakage guards, abstention logic, and coverage
  arithmetic may be tested, but cannot produce empirical estimates.

## Exact missing inputs

### CVIAN

Available inputs include source media, coordinates, sequence/spatial roles,
overall severity labels, and full-view/sector **severity** predictions. The
following declared-attestability inputs are absent:

1. provenance-bearing `roof_damage`, `eaves_damage`, `siding_damage`,
   `window_damage`, and `deck_damage` references;
2. per-view, per-field human observability and judgment/abstention annotations;
3. calibrated field-level attestation and abstention predictions;
4. a field × view support/power audit based on eligible real annotations;
5. complete camera-to-building distance/bearing, building orientation, and
   occlusion references (coordinates and panorama compass alone are partial);
6. frozen decision relevance weights for the six fields; and
7. an attestation-labeled prospective or event-held-out test. The existing
   137-row sequence test has only severity labels and historical base exposure.

The four Study A visibility features and segmentation ratios are features, not
field-level ground truth. A full-view severity prediction is not a roof/siding
attestation prediction.

### Eaton

Available inputs include 19,780 joined image pairs, coordinates, overall damage
categories, and DINS construction/material/exposure attributes. The following
are absent:

1. component-damage **presence/state** and damage-dominance references with
   annotation provenance;
2. per-view, per-field observability and judgment/abstention annotations;
3. genuine Eaton street/remote severity predictions required by RQ1 Study B;
4. all calibrated field-level attestation/abstention predictions required by
   RQ2/RQ3;
5. frozen Eaton spatial dependency groups and role-isolated manifests;
6. camera pose, building orientation, and occlusion references;
7. a frozen sequential candidate-view/action-cost protocol and decision-field
   weights; and
8. an attestation-labeled prospective or event-held-out test.

`dins_roofconstruction`, `dins_eaves`, `dins_exteriorsiding`, window/deck and
other whitelist columns describe construction, material, presence, or exposure.
They are not damage-state labels. `dins_wherefirestartedonstructure` records
fire origin for a restricted subset; it is neither the location nor dominance
of observed damage. Both source classes are hard-rejected by the validator.

## Minimum fail-closed artifacts

The frozen contract is
[`configs/crossviewguard_attestation_contract_v1.json`](../../configs/crossviewguard_attestation_contract_v1.json).
Its minimum long-form real annotation row is unique on
`(entity_id, view_id, field_id)` and contains:

- event, dependency group, and split role;
- view id/type and a frozen damage field id;
- exact reference semantics and reference state;
- the image-level view observation or `abstain`, plus observability state;
- reference source type/field and both reference and view-annotation
  provenance;
- record origin and SHA-256 fingerprints for source and media.

The prediction artifact is unique on `(seed, entity_id, view_id, field_id)` and
adds calibrated attestation probability, abstention probability, predicted
observation state, and model/annotation artifact hashes. It is a downstream
RQ2 product; it may not be manufactured from the reference label.

The machine-readable current inventory is
[`configs/crossviewguard_attestation_dependency_inventory_20260710.json`](../../configs/crossviewguard_attestation_dependency_inventory_20260710.json).
The validator in
[`crossview_conflict/decision/attestation.py`](../../crossview_conflict/decision/attestation.py)
fails on synthetic rows in claim mode, material/fire-origin references, invalid
field semantics, non-abstaining invisible views, cross-role dependency groups,
invalid probabilities, or any unmet upstream dependency.

## Unlock conditions

RQ2 may move to model fitting only after a frozen real annotation artifact
passes the contract, each field × view × role cell has an audited eligible
count, and a dependence-aware power/calibration plan is frozen. The geometry
subquestion additionally requires measured geometry and occlusion provenance.

RQ3 may start only after RQ2 produces a frozen, calibrated, out-of-role matrix;
decision-field weights and acquisition costs are frozen without looking at the
test; policy-fit/validation/test dependencies are disjoint; and the prospective
or external test itself carries real attestation annotations. Any defect found
after test scoring consumes that test.

## Verification

The annotation, prediction, and utility records used by the tests in
[`tests/test_attestation_contract.py`](../../tests/test_attestation_contract.py)
are synthetic fixtures only; the gate tests additionally read the boolean audit
inventory above. They verify plumbing and fail-closed behavior, do not consume
empirical labels or predictions, and do not train a model or estimate RQ2/RQ3
performance.

```powershell
python -m pytest -q tests/test_attestation_contract.py
```

Current result: **11 passed**.
