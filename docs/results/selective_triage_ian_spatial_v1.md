# Ian repaired-split selective triage and routing result

**Status:** engineering run complete for five seeds; no finite-sample
`risk-controlled` claim is supported by the present calibration design.

## Fixed protocol

- image split: 0.005-degree spatial blocks with a 25 m boundary buffer;
- gate-fit: 219 rows in 3 spatial blocks;
- risk-calibration: 191 rows in 4 spatial blocks;
- final test: 415 rows in 6 spatial blocks;
- gate artifact: one SHA-256-identified temperature/normalization/two-view/
  three-view gate per seed, identical across all three evidence roles;
- primary loss: severe false-negative rate conditional on accepted severe
  targets;
- risk sampling unit: `spatial_block_id`, using block-macro loss and a
  Hoeffding upper bound;
- registered `alpha=0.10`, `delta=0.05`, and threshold grid
  `{0.0, 0.1, ..., 1.0}`;
- event: `hurricane_ian_cvian`; cross-role minimum distances are 556.39 m,
  30.59 m, and 26.26 m.

The risk denominator contains 53 severe calibration rows, but those rows occur
in only **two** calibration blocks. Treating 191 correlated images as 191
independent samples would create a misleadingly narrow bound, so the evaluator
aggregates at the registered spatial-block unit.

## Result

All five seeds returned:

- status: `risk-aware`;
- reason: `calibration_did_not_certify_a_threshold`;
- selected threshold: null;
- authorized automated coverage: 0%;
- final dispositions: 415/415 `defer_human` for each seed.

Even the lowest upper bound on the registered grid is 1.0 for each seed because
only two severe-bearing calibration blocks are available. The full-coverage
block-macro severe FNR ranges from 0.207 to 0.357 across seeds. This is a useful
negative result: the current split supports a spatial generalization benchmark,
but not a finite-sample 10% severe-FNR guarantee. More independent calibration
blocks or a separately justified cluster-aware design are required; adding more
images inside the same blocks is not a remedy.

## Gate result

The strict gate-fit protocol produced the following five-seed final-test
macro-F1 values:

| gate | macro-F1 mean ± SD |
| --- | ---: |
| two-view linear | 0.6526 ± 0.0161 |
| three-view linear | 0.6501 ± 0.0139 |
| two-view MLP | 0.6443 ± 0.0226 |
| three-view MLP | 0.6483 ± 0.0168 |

The gate does not establish a material improvement over the repaired street
baseline (0.6500 macro-F1). The reliability signal remains useful for ranking
and analysis, but this run does not satisfy the selective-triage go criterion.

## Actionable routing output

The seed-42 `risk-aware` decisions were rendered into a fail-closed review
queue. A fixed three-stop `joint` route covers 3/6 blocks and 69.9% of samples
over 11.14 km of Haversine straight-line proxy distance. Descriptively, it
reaches 92/116 severe cases (79.3% Recall@budget, NDCG 0.803). Labels are used
only to evaluate this route, not to choose its stops.

In this one route case, `severity_only` reaches the same 92 severe cases and a
higher NDCG (1.000), while `joint` reaches its first severe block sooner. This
does **not** establish incremental conflict utility beyond severity; the routing
workstream remains an operational product output until that gain repeats across
events/budgets with a preregistered comparison.

The output is not a road route and does not model access, barriers, closures,
or travel time. The operational artifacts are generated under:

- `outputs/analysis/selective_triage_ian_spatial_v1/`;
- `outputs/analysis/actionable_report_ian_spatial_v1/seed42_k3/`.

The latter contains `tile_priority.csv`, `inspection_route.geojson`, and
`actionable_report.md`.
