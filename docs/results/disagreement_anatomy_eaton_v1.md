# RQ1 Study B — Eaton component-anatomy feasibility result

> **Historical audit, superseded for current execution status.** The earlier
> inventory statement that no Eaton per-view prediction export or spatial split
> existed is no longer current. Fresh role-isolated models and a frozen spatial
> protocol are documented in
> [`eaton_component_observability_v1.md`](eaton_component_observability_v1.md).
> This file is retained to preserve the original construct/media audit; its
> DINS fields still must not be substituted for human damage-dominance labels.

## Outcome

**BLOCKED for H-B1/H-B2 inference; data-availability, construct, media, and
best-case power audits executed.** The local DINS-derived artifacts are now
available and pass the join/media checks below, but they do not contain a
component-damage presence or component-damage dominance reference. A repository
inventory found zero CSV/Parquet paths containing both `eaton` and
`prediction`; no genuine Eaton street/remote per-view severity export or frozen
Eaton spatial block assignment is present. The runner did not synthesize either
one.

Consequently, this result does **not** estimate component-conditioned direction
shares or the explained fraction of disagreements. The claim “disagreement is
information, not error” remains unsupported. This is a construct-validity
blocker, not a negative H-B1/H-B2 result.

## Frozen image-visible field whitelist

The whitelist was frozen before any Study B outcome test in
[`configs/eaton_image_visible_fields_v1.json`](../../configs/eaton_image_visible_fields_v1.json).
Every value below remains an **inspector-recorded reference attribute**. A
view status only says that the attribute is in principle visible at the source
resolution when the component is actually in frame; it is not image-level
visibility ground truth.

| Inspector field | Reference semantics | Street | Overhead |
| --- | --- | --- | --- |
| `dins_roofconstruction` | construction/material | primary, conditional | primary, conditional |
| `dins_eaves` | enclosure type | primary, conditional | excluded |
| `dins_exteriorsiding` | siding material | primary, conditional | excluded |
| `dins_windowpane` | pane construction | stress test only | excluded |
| `dins_deckporchongrade` | presence/material | primary, conditional | stress test only |
| `dins_deckporchelevated` | presence/material | primary, conditional | stress test only |
| `dins_ventscreen` | vent/mesh detail | excluded | excluded |
| `dins_patiocovercarport` | presence/combustibility | stress test only | stress test only |
| `dins_fenceattachedtostructure` | presence/combustibility | stress test only | stress test only |

The whitelist explicitly marks all nine fields
`component_damage_reference=false`. Missing, `Unknown`, and “component absent”
are kept distinct. In particular, `roofconstruction=Asphalt` is not evidence
that the roof was damaged, and no material value is converted into a
roof/facade dominance label.

The source-resolution freeze was based on the first 1,000 joined pairs, before
outcome analysis: 708 street JPEGs were 1280×960, 290 were 640×480, two had
irregular dimensions, and all 1,000 overhead JPEGs were 512×512.

## Joined-artifact and media audit

| Quantity | Result |
| --- | ---: |
| Joined manifest rows / unique `pair_id` | 19,780 / 19,780 |
| Matched / unmatched DINS rows | 19,776 / 4 |
| Matched unique DINS structures | 18,411 |
| Structures with multiple attachments | 1,177 |
| Maximum attachments for one structure | 7 |
| Comparable overall-severity labels agreeing | 19,772 / 19,776 |
| Street files present | 19,780 / 19,780 |
| Overhead files present among declared paths | 19,754 / 19,754 |
| Rows without an overhead relative path | 26 |

The inspector/service join is therefore usable as a reference-attribute table,
but attachment rows are not independent structures. The service publishes no
direct or subtype domains for the component fields, so observed categorical
values cannot be certified through a published domain. The four overall-label
mismatches and any rare categorical anomalies must remain auditable rather than
being silently normalized.

Artifact fingerprints and the complete per-field null/`Unknown` counts are in
the machine audit. The joined manifest SHA-256 is
`c413c272a761b0c25269bab1158438b28f2f3727cb8bfdd7edab42d0b281ffcf`.

## Why the available component-like field is not the missing reference

`dins_wherefirestartedonstructure` is a fire-**origin** field requested for
`Affected (1-9%)` inspections. It is not the location or dominance of observed
damage. At the unique-structure level, the restricted diagnostic found:

- 762 affected structures with a known origin value;
- 33 with origin `Roof`;
- 583 with a facade-like origin (`Siding`, `Window`, or `Eaves`).

Even under an unrealistically favorable independent-structure calculation in
which every structure produced a disagreement, this 33-vs-583 split has an
approximate 80%-power minimum detectable difference of **0.251** in positive
direction share (two-sided α=.05 around p=.5). Spatial dependence,
disagreement filtering, and missing predictions would make the detectable
effect larger. This proxy is retained only as a feasibility stress test and is
forbidden for H-B1/H-B2.

For H-B2 precision, approximately 97 independent eligible disagreements are
needed for a worst-case 95% half-width of 0.10, or 385 for 0.05, before any
spatial design effect. The eligible disagreement count is currently unknown
because no Eaton per-view predictions or valid component-dominance references
exist; main-study power is therefore not estimable.

## Fail-closed prerequisites for execution

The independent runner
[`scripts/analyze_eaton_component_anatomy.py`](../../scripts/analyze_eaton_component_anatomy.py)
will execute component-direction statistics only after both external tables are
supplied:

1. A genuine Eaton prediction export unique on `(seed, pair_id)` with
   `street_prediction`, `remote_prediction`, and a frozen
   `spatial_block_id`. Ordinal values must share one documented class order.
2. A provenance-bearing component reference unique on `pair_id`, with
   `component_dominance ∈ {roof, facade, mixed, unknown}`,
   `reference_semantics=damage_dominance`, and non-empty
   `annotation_provenance`.

The runner rejects `final_test` rows in this development path, rejects missing
spatial blocks, and rejects construction/material semantics presented as
damage dominance. When valid inputs exist, it reports per-seed, spatial-block
cluster-bootstrap estimates for:

- H-B1: roof positive-direction share minus facade positive-direction share,
  where positive means overhead reports more severe;
- H-B2: the fraction of eligible disagreements whose direction matches the
  pre-specified roof/facade visibility direction.

No street or overhead component prediction was manufactured in this audit.
Per-view component prediction is a later RQ2 modeling object, not a substitute
for the missing Study B reference.

## Reproduction

```powershell
python scripts/analyze_eaton_component_anatomy.py `
  --joined-manifest $EATON_ROOT/dins_field_join/eaton_manifest_with_dins.csv `
  --field-domains $EATON_ROOT/dins_field_join/field_domains.json `
  --whitelist configs/eaton_image_visible_fields_v1.json `
  --output-dir outputs/analysis/disagreement_anatomy_eaton_v1
```

Current machine outputs:

- `outputs/analysis/disagreement_anatomy_eaton_v1/audit.json` — status,
  fingerprints, blockers, join/media audit, and restricted power diagnostic;
- `outputs/analysis/disagreement_anatomy_eaton_v1/field_availability.csv` —
  attachment- and unique-structure-level field availability;
- no `component_direction_statistics.csv`, by design, while prerequisites are
  missing.
