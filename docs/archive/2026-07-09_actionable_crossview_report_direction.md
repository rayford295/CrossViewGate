# From Damage Classification to Actionable Cross-View Reports

> Date: 2026-07-09
> Status: research framing note, no new experiments executed

## Core Question

The current CrossViewGate project is already stronger than a standard disaster
classification paper. Its main claim is not simply that cross-view fusion
improves accuracy. The more important claim is that cross-view evidence lets us
ask a different operational question:

> Which view should be trusted, where, and why?

This opens a natural next step. The output should not stop at a disaster class
label. It can become an actionable, auditable situation report that helps a
response team decide which places need field verification, which sensor should
be trusted, and where limited response resources should go first.

## Current Position of CrossViewGate

The current repository already supports this transition through four findings.

1. **Oracle gap**: on conflict cases, an oracle that picks the correct single
   view beats all tested fusion methods by 0.37-0.41 accuracy. This means the
   central missing capability is not another generic fusion architecture, but
   per-sample view selection.
2. **Reliability gate**: the gate uses building visibility, calibrated
   confidence, entropy, and cross-view disagreement to predict per-sample view
   reliability. These are exactly the ingredients needed for an explanation,
   not just a prediction.
3. **Causal field-of-view intervention**: building-centered crops increase
   fusion benefit, while random crops do not. This gives a physical reason for
   the reliability rule: cross-view evidence matters when the ground view
   actually sees the target.
4. **Conflict-density mapping**: spatial density of cross-view conflicts tracks
   aggregate damage better than ordinary uncertainty. This turns disagreement
   from an error case into a spatial damage signal.

Taken together, these results support a report-level framing:

> Cross-view disagreement is not merely a failure mode. It is an operational
> signal that reveals where disaster evidence is incomplete, obstructed, or
> view-dependent.

## Proposed Reframing

The project can be reframed as a three-level system.

### Level 1: Damage Classification

This is the conventional output:

- per-building or per-property damage class
- ordinal severity, such as no/trace damage, repairable damage, destroyed
- standard accuracy, macro-F1, QWK, MAE, and extreme-error metrics

This remains necessary, but it is not enough for disaster response.

### Level 2: Reliability-Aware Triage

This is the current distinctive contribution of CrossViewGate:

- identify whether street and overhead views agree or conflict
- estimate which view deserves more trust
- explain the trust decision using visibility, confidence, entropy, and
  disagreement features
- flag cases that should be routed to human review

This changes the model from a classifier into an evidence arbitration system.

### Level 3: Actionable Situation Report

This is the next research direction:

- summarize likely damage by area or tile
- rank places by review urgency
- identify why the model is uncertain or conflicted
- recommend the next data-collection or response action
- attach evidence that a human analyst can audit

This is the step from `See` and `Describe` toward `Govern`.

## What a Cross-View Report Should Contain

A useful report should be structured, not free-form. The model should generate
auditable fields that can later be rendered as Markdown, HTML, PDF, or a map
dashboard.

### Per-Sample Report

Each property or image pair can produce one row:

| Field | Meaning |
| --- | --- |
| `sample_id` | property, building, or paired-image identifier |
| `location` | latitude/longitude or tile id when available |
| `damage_prediction` | final damage class |
| `severity_probability` | probability of the highest operational severity |
| `street_prediction` | street-only class |
| `remote_prediction` | overhead-only class |
| `crossview_prediction` | cross-view model class |
| `gate_street_weight` | how much the gate trusts street view |
| `gate_remote_weight` | how much the gate trusts overhead view |
| `views_disagree` | hard street/remote disagreement flag |
| `js_divergence` | soft cross-view conflict strength |
| `building_visibility` | whether the street image sees the target structure |
| `confidence_summary` | calibrated confidence and entropy from both views |
| `review_priority` | low, medium, high, or urgent |
| `recommended_action` | field check, remote monitoring, recapture, or direct triage |
| `rationale` | short evidence-based reason |

### Tile-Level Situation Report

Each area or tile can produce one row:

| Field | Meaning |
| --- | --- |
| `tile_id` | geographic tile or overhead image tile |
| `n_samples` | number of properties in the tile |
| `mean_damage_score` | aggregate ordinal severity estimate |
| `severe_probability_mean` | mean severe/destroyed probability |
| `hard_conflict_density` | fraction of hard street/remote disagreements |
| `soft_conflict_density` | mean JS divergence |
| `uncertainty_density` | ordinary uncertainty control |
| `priority_rank` | response priority among all tiles |
| `top_review_cases` | sample ids for the most important review cases |
| `area_recommendation` | action recommendation for this tile |

## Action Taxonomy

The report should avoid overclaiming. It should not pretend to decide all
emergency operations by itself. A defensible action taxonomy is:

| Condition | Recommended action |
| --- | --- |
| high damage probability, high agreement | prioritize response or damage documentation |
| high damage probability, high conflict | urgent field verification |
| low target visibility, high uncertainty | recapture street-level imagery or use overhead-only monitoring |
| overhead severe, street mild, low street visibility | trust overhead more; verify roof or debris damage |
| street severe, overhead mild, high street visibility | trust street more; verify facade, debris, or interior-adjacent damage |
| high tile conflict density | prioritize area-level reconnaissance |
| low conflict, low severity, high agreement | deprioritize or monitor remotely |

This keeps the system in a decision-support role. It recommends what evidence
to collect or review next, rather than claiming to replace field judgment.

## Evaluation Plan

The report direction needs its own metrics. Accuracy alone is not enough.

### Per-Sample Evaluation

- **Precision@K for urgent review**: among the top K properties ranked by
  review priority, how many are severe or destroyed?
- **Recall@K for severe cases**: how many severe cases appear in the top K?
- **Extreme-error reduction**: does routing conflict cases reduce
  no-damage/destroyed confusions?
- **Human-review workload reduction**: how much of the dataset can be safely
  assigned low priority while retaining high recall for severe damage?
- **Rationale validity**: do high street weights correspond to high target
  visibility and confident street evidence?

### Tile-Level Evaluation

- Spearman correlation between tile priority and mean observed damage
- NDCG for ranking damaged tiles
- Precision@K for highest-priority tiles
- comparison against uncertainty-only ranking
- stability across grid size and minimum tile sample count

### Report-Level Evaluation

A small human evaluation could ask GIScience or emergency-management reviewers
to compare:

1. class-only damage map
2. class map plus uncertainty
3. CrossViewGate report with conflict, reliability, and action fields

The expected advantage is not that the report changes every decision. The
expected advantage is that it makes the reason for prioritization auditable.

## Paper and Dissertation Framing

The strongest framing is:

> Cross-view disaster GeoAI should not end at damage classification. Because
> ground and overhead views observe different physical evidence, their
> disagreement can diagnose which evidence is missing, which view should be
> trusted, and which places need human or field attention first.

In dissertation language:

- `See`: detect damage from ground and overhead imagery
- `Describe`: explain which view supports the damage judgment and why
- `Govern`: produce review priorities, evidence packets, and action
  recommendations for response workflows

This makes CrossViewGate a bridge from reliable perception to decision support.

## Recommended Next Implementation

The next code module should be small and output-focused:

1. `scripts/build_actionable_report.py`
   - read test predictions, visibility features, gate outputs, and split CSVs
   - write `per_sample_report.csv`
   - write `tile_sitrep.csv`
   - write `actionable_report.md`
2. `docs/results/actionable_report_results.md`
   - report ranking metrics and example cases
3. `docs/assets/site/actionable_report_preview.png`
   - optional visual preview for the project website

No new model training is required for the first version. The first version can
be built entirely from existing predictions and analysis outputs.

## Bottom Line

Yes, CrossViewGate can and should move beyond classification. The cleanest next
step is not a generic language-model report generator. It is a structured
cross-view situation report that converts damage predictions, view reliability,
and spatial conflict density into auditable response priorities.

