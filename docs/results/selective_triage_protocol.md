# Selective Triage Risk-Control Protocol (P0.7)

This protocol implements the minimum claim discipline for CrossViewGuard's
selective-triage experiments. It is an evaluation protocol, not evidence that
the present models already achieve operational risk control.

## Locked data roles

Every run requires three physically different CSV files with disjoint stable
identifiers:

1. **gate-fit**: data used to fit the reliability gate or risk score. The P0.7
   evaluator does not refit the gate; this file exists to audit the boundary.
2. **risk-calibration**: the only labels used to choose an acceptance threshold.
3. **final-test**: evaluated once after the threshold is locked. Test labels do
   not change the threshold, loss, review budget, score, or claim scope.

By default, the record audit uses `dataset,sample_id` when both columns exist
and `sample_id` otherwise. A separate group audit checks every shared candidate
among `sequence_id`, `spatial_block_id`, `tile_id`, `group_id`, and `objectid`;
formal runs should explicitly register the intended grouping with
`--group-columns`. Repeated group values are allowed inside a role but forbidden
across roles. `--spatial-separation-m` additionally verifies a WGS84 minimum
distance and is mandatory for a `risk-controlled` status. Missing event/group,
spatial, or fixed-gate identity metadata makes the claim `risk-aware`; a clean
sample-ID audit alone does not establish independence.

## Pre-registered policy

Before reading risk-calibration labels, register all of the following:

- the fixed upstream model, reliability gate, and scalar risk score;
- the score direction (this implementation reviews high scores and accepts
  `risk_score <= threshold`);
- one bounded primary loss;
- target risk `alpha` and error probability `delta`;
- the finite numeric threshold grid;
- the event scope and stable split/group identifiers.

All three evidence roles must carry the same `gate_artifact_id`, a SHA-256
fingerprint over temperatures, gate-fit evidence, normalization statistics,
and both fitted gate state dictionaries. This prevents independently fitted
role CSVs from masquerading as one frozen policy.

The supported losses are:

- `severe_miss`: conditional severe false-negative risk. The loss is `1` when
  a severe/destroyed target is predicted outside that set, and the risk
  denominator contains accepted severe targets only. Non-severe samples still
  count toward automated coverage but cannot dilute the severe FNR;
- `extreme_error`: `1` when ordinal target-prediction distance is at least the
  registered distance, otherwise `0`;
- `cost_weighted`: a complete registered cost-matrix entry divided by its
  declared maximum. The evaluator rejects, rather than clips, costs above the
  declared maximum.

Native label meanings differ across Eaton, Ian, and Milton. `--severe-labels`,
`--label-order`, and any cost matrix must therefore be registered per event or
through a separately justified operational ontology. Equal class indices are
not by themselves a cross-event ontology.

## Finite-sample upper bound and selection

For a fixed threshold, losses first use the registered row-level denominator:
all accepted samples except for `severe_miss`, where it is accepted severe
targets. They are then averaged within the registered `--risk-group-col` and
bounded across group means. Thus `n` is the number of independent groups, not
the number of correlated images. Empty risk denominators or groups are not
treated as zero-risk policies and cannot be certified.

The implementation uses:

- **ungrouped binary diagnostic**: the exact one-sided Clopper-Pearson upper
  confidence limit, computed by numerical inversion of the binomial lower tail;
- **general `[0,1]` loss**: the one-sided Hoeffding limit

  `U = min(1, R_hat + sqrt(log(1 / delta_eff) / (2 n)))`.

Group means are generally fractional, so formal grouped runs use Hoeffding.
The output records both row-level empirical risk and equal-weight group-macro
risk; the bound applies to the latter.

For a pre-registered grid of `m` thresholds, `delta_eff = delta / m`. This
Bonferroni correction makes all grid bounds simultaneous without assuming that
the threshold results are independent. Among thresholds with `U <= alpha`, the
calibration procedure selects the one accepting the most samples; an equal-count
tie selects the smaller threshold. If none passes, the selected threshold is
null, all final-test samples are deferred, and the run is `risk-aware`.

This is a finite-grid, learn-then-test-style rule. It is not a claim for an
arbitrary continuously optimized threshold.

### Assumptions and claim limits

The finite-sample interpretation requires all of the following:

- calibration observations are independent/exchangeable draws from the target
  deployment population, or the sampling design otherwise justifies the bound;
- the gate, score, loss, label ontology, grid, `alpha`, and `delta` are fixed
  without using calibration losses;
- the registered loss is truly bounded in `[0,1]`;
- the accepted observations for each fixed selector represent the same
  conditional population that will be encountered at deployment;
- there is no final-test policy tuning or selective omission of runs.

Disaster imagery often violates naive i.i.d. assumptions through spatial,
property, sequence, or event clustering. Use grouped/spatial holdouts and state
the sampling unit. The code cannot validate exchangeability from a CSV.

An in-event run is labeled `risk-controlled` only when calibration certifies a
threshold **and** the one-threshold held-out diagnostic upper bound is no larger
than `alpha`. A failed held-out diagnostic downgrades the result to `risk-aware`;
the script does not search for a replacement threshold on test. A
`cross_event_stress_test` is always `risk-aware`, even if its empirical risk is
small. A target-event calibration set or a separately justified shift-aware
method is required for a new-event risk-control claim.

## Metrics and artifacts

The final-test artifacts are:

| File | Contents |
| --- | --- |
| `summary.json` | selected calibration threshold/count/upper bound, locked final-test result, `risk-controlled` or `risk-aware` state, and claim reason |
| `calibration_threshold_grid.csv` | every registered threshold, accepted count, coverage, empirical risk, simultaneous upper bound, and selected flag |
| `risk_coverage.csv` | samples accepted from low to high risk, with coverage and cumulative bounded risk |
| `recall_at_review_budgets.csv` | Recall@review budgets 5/10/20/30/50 by default |
| `per_sample_decision.csv` | auditable final-test evidence plus decision skeleton |
| `split_audit.json` | role counts, identifier fields, and zero-overlap audit |

The reported AURC groups equal scores and integrates selective risk using each
distinct threshold's coverage increment. Recall@review also never splits an
equal-score group: if a tie crosses the nominal budget, the full group is
reviewed and the larger realized budget is recorded. These metrics are row-order
invariant and descriptive; they do not select the threshold. `severe_target`
recall measures severe/destroyed cases found;
`loss_event` measures samples with positive registered loss; and
`classification_error` measures all misclassifications. The selected recall
target is recorded in the output.

`per_sample_decision.csv` retains the input prediction/evidence fields and adds:

- `decision_source`: `street`, `overhead`, or `gated_mixture`;
- `disposition`: `accept`, `defer_human`, or optionally `acquire_view`;
- `acquisition_target`: populated only for `acquire_view` rows;
- `reason`: whether the risk score is above/below the locked threshold, no
  threshold was certified, or the final policy was not authorized;
- `threshold_eligible`: score-only eligibility under the locked threshold;
- `policy_authorized`: one only when the held-out run retains
  `risk-controlled` status;
- `risk_control_status` and its reason;
- `risk_score`, `locked_threshold`, `accepted`, and `bounded_loss`.

Operational actions fail closed: any `risk-aware` result, including a failed
held-out bound or cross-event stress test, emits no `accept` dispositions even
when some rows are below the locked threshold.

## CSV compatibility and example

For compact prediction CSVs, `prediction`, `target`, `confidence`, and
`sample_id` are sufficient to compute descriptive risk metrics; auto scoring
uses `1-confidence`. A `risk-controlled` claim additionally requires verified
event and group metadata (and coordinates when spatial separation is requested).
For reliability
evidence CSVs, auto mode prefers a supplied `risk_score`, then gate confidence,
then `1-max(gate3_probability_*)` or `1-max(gate2_probability_*)`. Entropy and
JS divergence are accepted as high-is-risky scores. Explicitly pass
`--risk-score-col` and `--score-transform` when the intended rule differs.

```bash
python scripts/eval_selective_triage.py \
  --gate-fit-csv outputs/protocol/ian_gate_fit.csv \
  --risk-calibration-csv outputs/protocol/ian_risk_calibration.csv \
  --final-test-csv outputs/protocol/ian_final_test.csv \
  --id-columns dataset,sample_id \
  --group-columns spatial_block_id \
  --spatial-separation-m 25 \
  --risk-group-col spatial_block_id \
  --loss severe_miss \
  --severe-labels 2 \
  --alpha 0.10 \
  --delta 0.05 \
  --threshold-grid 0:1:0.01 \
  --claim-scope in_event \
  --output-dir outputs/analysis/selective_triage/ian_seed42
```

For a cost-weighted run, pass a complete JSON matrix and its matching label
order, for example `--label-order 0,1,2 --cost-matrix-json costs_ian.json`.
Running multiple losses, grids, or alphas and reporting only the best run is an
unregistered multiple-comparison procedure and must not be described as the
single P0.7 risk-control result.
