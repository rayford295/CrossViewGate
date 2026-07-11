# Operational Label and Cost Ontology

This document specifies the P0.5/P0.6 contract for turning dataset-native
damage labels into operational decisions. The machine-readable source of truth
is [`configs/operational_ontology.json`](../configs/operational_ontology.json),
loaded through
[`crossview_conflict.decision.ontology`](../crossview_conflict/decision/ontology.py).

## Non-equivalence rule

The Eaton, Ian, and Milton label systems are retained exactly as released.
Their integer indices are **dataset-local identifiers**, not shared semantic
classes. For example, all three datasets have an index `0`, but it means `No
Damage`, `0_MinorDamage`, and `mild_damage`, respectively. Equal indices must
never be used to join rows, pool confusion matrices, or claim label
equivalence.

The action mapping below is an explicit versioned policy. Two native labels
mapping to the same action means only that the current policy gives them the
same response. It does not make their definitions, visual evidence, or native
evaluation targets equivalent. Native-label metrics must therefore still be
reported by event.

## Native labels and action levels

The ordinal decision actions are:

| Rank | Action | Operational meaning |
| ---: | --- | --- |
| 0 | `routine_monitoring` | No immediate field response; retain for routine monitoring. |
| 1 | `priority_inspection` | Prioritize verification or repair-oriented inspection. |
| 2 | `urgent_response` | Escalate for urgent life-safety or major-loss response. |

`human_escalation` is deliberately non-ordinal. It is an intervention, not a
fourth severity level.

The native-to-action mapping is:

| Dataset | Native index | Native label | Action | Decision eligible |
| --- | ---: | --- | --- | --- |
| Eaton wildfire | 0 | `No Damage` | `routine_monitoring` | yes |
| Eaton wildfire | 1 | `Affected (1-9%)` | `routine_monitoring` | yes |
| Eaton wildfire | 2 | `Minor (10-25%)` | `priority_inspection` | yes |
| Eaton wildfire | 3 | `Major (26-50%)` | `priority_inspection` | yes |
| Eaton wildfire | 4 | `Destroyed (>50%)` | `urgent_response` | yes |
| Eaton wildfire | 5 | `Inaccessible` | `human_escalation` | **no** |
| Ian / CVIAN | 0 | `0_MinorDamage` | `routine_monitoring` | yes |
| Ian / CVIAN | 1 | `1_ModerateDamage` | `priority_inspection` | yes |
| Ian / CVIAN | 2 | `2_SevereDamage` | `urgent_response` | yes |
| Milton / GenDisasterSVI | 0 | `mild_damage` | `routine_monitoring` | yes |
| Milton / GenDisasterSVI | 1 | `moderate_damage` | `priority_inspection` | yes |
| Milton / GenDisasterSVI | 2 | `severe_damage` | `urgent_response` | yes |

### Read-only Eaton three-class summary

The main Eaton `wildfire_3class` protocol is exposed as a **derived summary**,
not another source-native label set:

| Derived index | Derived label | Source-native provenance | Action |
| ---: | --- | --- | --- |
| 0 | `no_or_trace_damage` | `No Damage`; `Affected (1-9%)` | `routine_monitoring` |
| 1 | `damaged_repairable` | `Minor (10-25%)`; `Major (26-50%)` | `priority_inspection` |
| 2 | `destroyed` | `Destroyed (>50%)` | `urgent_response` |

`Inaccessible` is explicitly excluded from this derived scheme and remains in
the six-class source provenance. The configuration requires every source-native
label to appear in exactly one derived group or in the explicit exclusion list.
It also requires each group to preserve the action mapping of all its source
labels. The returned scheme and label objects are frozen; deriving a summary
never mutates the native record.

## Cost policy

Costs are normalized policy weights, not measured dollars or validated
life-safety utilities. They must be versioned, preregistered for a headline
experiment, and accompanied by sensitivity analysis before operational use.

| Cost term | Weight | Definition |
| --- | ---: | --- |
| `correct` | 0.0 | Predicted and actual action levels agree. |
| `adjacent_error` | 1.0 | Predicted action differs by one ordinal level. |
| `extreme_error` | 4.0 | Fallback for errors spanning more than one level. |
| `severe_miss` | 8.0 | Actual `urgent_response`, predicted `routine_monitoring`. |
| `false_alarm` | 3.0 | Actual `routine_monitoring`, predicted `urgent_response`. |
| `human_review` | 1.5 | Explicit human escalation. |
| `view_acquisition` | 0.5 | Acquire one additional view. |

Rows in the decision cost matrix are actual actions and columns are predicted
actions:

| Actual \ Predicted | Routine | Priority | Urgent |
| --- | ---: | ---: | ---: |
| Routine | 0 | 1 | 3 |
| Priority | 1 | 0 | 1 |
| Urgent | 8 | 1 | 0 |

The two endpoint errors in the current three-action ontology have named,
directional costs (`severe_miss` and `false_alarm`). `extreme_error` remains a
validated fallback so a later policy can add ordinal levels without silently
treating a multi-level error as adjacent.

Human review and view acquisition are resource costs. They are not extra rows
of severity ground truth. A policy may compare the expected cost of an
automatic action with `human_review` or `view_acquisition`, but it must record
the chosen intervention explicitly.

## `Inaccessible` semantics

Eaton `Inaccessible` records a field-access constraint observed in the
inspection workflow. It does not mean that an image is blurry, that a building
is not visible, that one view is missing, or that a model is uncertain.

Permitted uses are limited to:

1. a separately reported stress-test stratum; and
2. a trigger or audit stratum for explicit human escalation.

It is forbidden as a generic uncertainty label, abstention target, primary
severity class, or missing-view marker. It is excluded from all default
severity cost matrices. The API fails closed if code attempts to score it as a
severity truth or prediction.

This distinction also applies in the other direction: a model may defer a
non-`Inaccessible` case because its evidence is uncertain, but that defer
decision must be recorded as `human_escalation`, not relabeled as
`Inaccessible`.

## Cross-event aggregation

Cross-event results may be combined only after explicit action mapping. Each
metric is first computed separately for each event, retaining its denominator,
then macro-averaged over events. A pooled sample-level number cannot replace
the event breakdown.

| Metric | Cross-event aggregation | Reducer |
| --- | --- | --- |
| `operational_expected_cost` | allowed on decision-eligible action levels | event macro |
| `severe_miss_rate` | allowed on the explicit endpoint definition | event macro |
| `false_alarm_rate` | allowed on the explicit endpoint definition | event macro |
| `human_review_rate` | allowed for explicit escalation decisions | event macro |
| `view_acquisition_rate` | allowed for explicit acquisition decisions | event macro |
| `native_confusion_matrix` | forbidden | event only |
| `native_macro_f1` | forbidden | event only |
| `native_class_recall` | forbidden | event only |

An unregistered metric fails closed: it must receive a reviewed aggregation
rule before it can appear as a cross-event headline. Reports must include the
ontology id, schema version, per-event values, sample denominators, excluded
non-decision labels, and the macro aggregate.

## Python API

```python
from crossview_conflict.decision import load_operational_ontology

ontology = load_operational_ontology()

# Dataset context is mandatory for native integer indices.
assert ontology.label_from_index("ian_hurricane", 0) == "0_MinorDamage"
assert ontology.action_level("ian_hurricane", 2) == "urgent_response"

# The Eaton 3-class view is derived and retains its source-native names.
derived = ontology.to_derived_label(
    "eaton_wildfire", "wildfire_3class", "Affected (1-9%)"
)
assert derived is not None
assert derived.name == "no_or_trace_damage"
assert derived.source_native_labels == ("No Damage", "Affected (1-9%)")
assert ontology.to_derived_label(
    "eaton_wildfire", "wildfire_3class", "Inaccessible"
) is None

# Actual labels are rows; predictions are columns.
matrix = ontology.cost_matrix("ian_hurricane")
severe_miss_cost = matrix.cost("2_SevereDamage", "0_MinorDamage")

# Resource decisions are explicit and separate from severity labels.
review_cost = ontology.intervention_cost("human_review")
next_view_cost = ontology.intervention_cost("view_acquisition")

# Inaccessible can be audited but cannot become generic uncertainty.
ontology.validate_label_use("eaton_wildfire", "Inaccessible", "stress_test")

# Only registered action-level metrics can be macro-averaged across events.
overall = ontology.aggregate_event_values(
    "operational_expected_cost",
    {
        "eaton_wildfire": 1.2,
        "ian_hurricane": 1.0,
        "milton_hurricane": 0.8,
    },
)
```

`load_operational_ontology(path)` validates custom JSON before returning an
object. Validation covers exact native labels, dataset-local indices, derived
provenance partitions, action ranks, cost ordering, `Inaccessible`
restrictions, and cross-event rules.
