# Routing and actionable-report MVP protocol (Weeks 3–4)

## Status and scope

This protocol converts the sample-level reliability artifact into a spatially
auditable inspection queue. It is an operational MVP for the Conflict-to-Route
workstream, not a road-navigation system. Its three outputs are:

| Artifact | Purpose |
| --- | --- |
| `tile_priority.csv` | Tile evidence components, policy scores/ranks, reference route ranks, and selected-route distance audit |
| `inspection_route.geojson` | Ordered inspection points plus an origin-to-stop line in GeoJSON coordinate order |
| `actionable_report.md` | Human-readable assumptions, queue, baseline comparison, provenance, and optional label metrics |

The default distance is WGS84 Haversine great-circle distance. It is a
straight-line/geodesic proxy only. It is **not road-network routing**, does not
model water/road barriers, closures, accessibility, vehicle speed, or travel
time, and excludes a return-to-origin leg. A road-graph experiment is deferred
until its graph source, date, snapping rule, unreachable-node policy, and
licensing are preregistered.

## Input contract

The primary input is the P0.7 `per_sample_decision.csv` produced by
`scripts/eval_selective_triage.py`. Required spatial/decision concepts are:

- latitude (`latitude` or `lat`);
- longitude (`longitude`, `lon`, or `lng`);
- tile identifier (`tile_id`, with documented aliases);
- `risk_score` and `disposition` for P0.7;
- a severity probability, preferably an explicit `severity_probability` or the
  highest class from the first available `gate3`, `gate2`, `crossview`,
  `remote`/`overhead`, or `street` probability family;
- conflict from `hard_conflict`/`views_disagree`, `js_divergence`, or both.

P0.2 evidence is accepted as a compatibility input. When it lacks
`risk_score`, the script uses `1 - max(class probability)` from the first
available probability family. When it lacks `disposition`, it records
`evidence_only`; it does not invent a calibrated accept/defer decision. Every
fallback is disclosed in the report. Missing severity or conflict fields become
zero-score components with an explicit warning so non-dependent policies can
still run.

Routing also fails closed on older P0.7 artifacts: any row marked with a
non-`risk-controlled` status or `policy_authorized=0` cannot remain `accept` and
is converted to `defer_human` with a report warning. An `accept` row that lacks
both authorization fields is likewise deferred.

The output is label-optional. The precise claim is: **target-event routing
inference requires no new target labels**. This is not an end-to-end label-free
claim if upstream models were trained using labels from the same event.

## Tile aggregation and priority policies

Samples are grouped by `tile_id`. Tile latitude/longitude are sample-coordinate
means; evidence and escalation components use the maximum within a tile so a
single high-risk property is not averaged away. Sample and labeled/severe
counts remain in the audit table.

Component scores are in `[0, 1]`:

- severity: highest-severity class probability;
- uncertainty: `risk_score` unchanged when already in `[0,1]`, otherwise a
  deterministic min-max normalization across the input event;
- conflict: `max(hard_conflict, clip(JS / ln(2), 0, 1))`;
- escalation: 1 for human review/acquisition/inspection/verification actions,
  otherwise 0.

The preregistered baselines and MVP policies are:

| Policy | Rule |
| --- | --- |
| `random` | Fixed-seed random tile score (default seed 42) |
| `nearest_neighbor` | Greedily choose the geographically closest feasible next tile |
| `severity_only` | severity |
| `uncertainty_only` | uncertainty |
| `conflict_only` | conflict |
| `reliability_aware` | `0.50 severity + 0.30 uncertainty + 0.20 escalation` |
| `joint` | `0.40 severity + 0.25 uncertainty + 0.25 conflict + 0.10 escalation` |

The weights are fixed MVP defaults, not final-test-tuned parameters. Score
policies select the highest-scoring feasible next tile. Equal scores break by
shorter next leg and then lexical `tile_id`; nearest-neighbor ties break by
`tile_id`. Random priorities and every tie break are deterministic.

## Resource budgets

Two resource constraints are implemented:

1. fixed `K` stops (`--max-stops K` or `--k-stops K`);
2. cumulative travel distance (`--distance-budget-km B` or
   `--travel-budget-km B`).

They can be combined, in which case both limits apply. If neither is supplied,
the default is 20 stops. The first leg from the declared origin counts. With no
explicit `--start-lat/--start-lon`, the mean tile coordinate is the common
synthetic origin for every baseline. A candidate whose next leg exceeds the
remaining distance is skipped; other feasible candidates may still be chosen.
Therefore every emitted stop has auditable `leg_distance_km`,
`cumulative_distance_km`, and `remaining_distance_budget_km`, and cumulative
distance never exceeds the declared budget (up to a `1e-9 km` numerical
tolerance).

## Optional target metrics

Routes never use `target`. When a non-empty target column is present, the report
adds descriptive:

- severe discoveries (reached severe samples);
- severe Recall@budget (reached severe samples / all severe samples);
- NDCG@budget using per-tile severe-sample count as graded relevance;
- severe-tile discoveries and distance to first severe tile.

Pass `--severe-labels` to preregister severe/destroyed target values. If it is
omitted, the script infers the highest numeric target, the greatest numeric
prefix such as `2_Destroyed`, or a destroyed/severe keyword label, and discloses
the result. These metrics must not be used to tune weights or choose a policy on
the final test event. Without labels, the same three operational artifacts are
written and label metrics are `n/a`.

## Reproduction

Fixed-stop example:

```powershell
python scripts/build_actionable_report.py `
  --input-csv outputs/analysis/selective_triage/ian_seed42/per_sample_decision.csv `
  --output-dir outputs/analysis/actionable_report/ian_seed42_k20 `
  --policy joint `
  --max-stops 20 `
  --random-seed 42 `
  --severe-labels 2
```

Distance-budget example:

```powershell
python scripts/build_actionable_report.py `
  --input-csv outputs/analysis/selective_triage/ian_seed42/per_sample_decision.csv `
  --output-dir outputs/analysis/actionable_report/ian_seed42_25km `
  --policy reliability_aware `
  --distance-budget-km 25 `
  --start-lat 26.6406 `
  --start-lon -81.8723 `
  --random-seed 42
```

Run the focused contract tests with:

```powershell
python -m pytest tests/test_routing_report.py -q
```

They cover fixed-seed determinism, fixed-stop selection, cumulative distance
enforcement, GeoJSON `[longitude, latitude]` order, P0.2 fallbacks, optional
label metrics, and fully label-free execution.
