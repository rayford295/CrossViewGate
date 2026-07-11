# CVIAN spatial-v1 active next-view benchmark

**Status:** five-seed frozen-encoder sequential-reveal development experiment complete; development GO criterion not met.

## Development protocol

- 4,121 CVIAN panoramas were georeferenced before splitting; retained roles are 3,253 train, 410 validation, and 415 development test, with 43 boundary exclusions.
- Each 1024x512 equirectangular panorama is represented by eight overlapping 90-degree sectors at 45-degree relative-yaw intervals.
- Every episode starts from post-event overhead plus fixed forward `sector_0`; the main comparison uses total street-view budget `k=3`.
- Frozen five-seed cross-view encoders produce sector/overhead embeddings. They were trained on the full spatial train role, so the perception stage is not role-disjoint. Downstream only, 187 blocks fit the mask-aware classifier and 47 disjoint blocks fit the selector (seed 42 counts; partitions vary by seed).
- The run fingerprint binds the historical v1 embedding and visibility metadata. Checkpoint hashes are recorded, but the exact SegFormer revision is unresolved; that signal is restricted to the privileged development baseline.
- Selector training states contain `k=1..3`, with one-step targets reaching `k=4`. Learned decisions for endpoints `k=5..8` are budget extrapolations; the main `k=3` result is within the trained state range.
- Validation selects model epochs and temperature. The 415-row, 6-block spatial test was inspected during implementation and is now consumed; this result is a development evaluation, not a confirmatory locked-test claim.
- Online policies see overhead, revealed sector embeddings/logits, the revealed mask, current uncertainty, step, and budget. Hidden visibility/confidence and labels never enter the learned observation. State aggregation is reveal-order invariant under fixed canonical sector indexing, not sector-index permutation invariant or cyclically equivariant.
- Operational cost matrix is `[[0,1,4],[1,0,1],[8,8,0]]`; each additional sector costs 0.5. The initial sector is sunk cost.
- Claim scope is offline sequential evidence reveal and `risk-aware`, not real-world acquisition and not finite-sample `risk-controlled` deployment.

## Five-seed development-test result at k=3

| Policy | Scope | Macro-F1 | Severe recall | Operational cost | Total cost (view cost 0.5) | Regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `oracle_cost` | privileged | 0.829 ± 0.033 | 0.928 ± 0.030 | 0.344 ± 0.095 | 1.344 ± 0.095 | 0.000 ± 0.000 |
| `entropy_reduction_privileged` | privileged | 0.606 ± 0.036 | 0.897 ± 0.045 | 0.708 ± 0.115 | 1.708 ± 0.115 | 0.225 ± 0.031 |
| `max_confidence_privileged` | privileged | 0.645 ± 0.052 | 0.869 ± 0.040 | 0.713 ± 0.088 | 1.713 ± 0.088 | 0.314 ± 0.012 |
| `farthest` | online | 0.659 ± 0.040 | 0.828 ± 0.075 | 0.769 ± 0.172 | 1.769 ± 0.172 | 0.749 ± 0.104 |
| `max_building_privileged` | privileged | 0.649 ± 0.026 | 0.821 ± 0.058 | 0.791 ± 0.100 | 1.791 ± 0.100 | 0.686 ± 0.058 |
| `learned` | online | 0.649 ± 0.036 | 0.814 ± 0.044 | 0.811 ± 0.104 | 1.811 ± 0.104 | 0.688 ± 0.082 |
| `random` | online | 0.644 ± 0.032 | 0.810 ± 0.054 | 0.811 ± 0.085 | 1.811 ± 0.085 | 0.744 ± 0.073 |
| `clockwise` | online | 0.632 ± 0.046 | 0.797 ± 0.049 | 0.855 ± 0.134 | 1.855 ± 0.134 | 0.782 ± 0.099 |

At equal view budget, the learned selector does not beat the predefined building-centered privileged heuristic, random selection, or the online farthest-sector coverage rule. The development GO criterion is therefore **NO-GO**.

The learned policy operational cost is 0.811, versus random 0.811, farthest 0.769, building 0.791, and the label-aware oracle 0.344. The oracle result shows substantial actionable view-selection headroom even though the present learned selector does not capture it.

Relative to the label-aware greedy one-step oracle at `k=3`, learned closes 13.1% of the operational-cost gap, random closes 13.1%, farthest closes 21.0%, and hidden maximum-confidence closes 31.5%.

The configured acquisition cost is also consequential: two additional views cost 1.0 at `k=3`, larger than the average error-cost reduction even for the oracle. Thus a fixed policy that acquires two views for every sample is not cost-optimal at the current ontology cost; a future adaptive stop/defer policy must target only high-value cases.

## Statistical reading

- `random` minus learned cost improvement: -0.088, 4-component bootstrap 95% CI [-0.262, +0.025]; severe-miss improvement -0.001 [-0.034, +0.034].
- `clockwise` minus learned cost improvement: +0.044, 4-component bootstrap 95% CI [-0.054, +0.169]; severe-miss improvement +0.012 [-0.017, +0.046].
- `farthest` minus learned cost improvement: -0.073, 4-component bootstrap 95% CI [-0.142, -0.031]; severe-miss improvement -0.015 [-0.020, -0.010].
- `max_building_privileged` minus learned cost improvement: -0.158, 4-component bootstrap 95% CI [-0.404, +0.013]; severe-miss improvement -0.024 [-0.072, +0.024].
- `max_confidence_privileged` minus learned cost improvement: -0.162, 4-component bootstrap 95% CI [-0.288, -0.078]; severe-miss improvement -0.065 [-0.089, -0.040].

The test has six spatial blocks, but cross-block sequence overlap connects them into only four dependency components. Intervals therefore resample those four components and remain descriptive; they cannot support a strong risk-control guarantee. Headline table values are sample-weighted means across model seeds, whereas comparison centers are equal-component macro means. The test was consumed during implementation, so future selector variants require a new sequence/event holdout. Sectors and model seeds are not treated as independent samples. Random uses one deterministic trajectory per model seed, so its seed spread also mixes model and policy randomness.

## Interpretation and next experiment

1. The active-view task is viable as a benchmark because the greedy one-step label-aware oracle gap is large; it is not a globally optimal k-step bound.
2. The current supervised action-classification target is weak (validation action macro-F1 about 0.17-0.20) and does not generalize reliably across seeds.
3. Farthest angular coverage is the strongest admissible simple rule on average; hidden confidence is a strong privileged signal, suggesting utility regression with candidate-side features available only after a cheap preview is worth testing.
4. The next model should predict utility with spatial-block OOF targets and add adaptive `stop/defer_human`; it must be evaluated before any Milton transfer.
5. Spatial-v1 still has sequence overlap, so a sequence-grouped sensitivity run is required before claiming capture-session generalization.

Machine-readable outputs are under `outputs/analysis/active_view_cvian_spatial_v1/experiment/`.
