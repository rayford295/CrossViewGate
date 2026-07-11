# CVIAN sequence utility experiment v1

**Primary decision: NO-GO.**

Claim scope: within-CVIAN development confirmation; label-cost Phase-1 baseline.

This report is generated only after all five one-time prospective-test seed artifacts have been staged. The primary estimand is equal dependency-component macro operational cost at fixed k=3; adaptive results are secondary and cannot change the decision.

## Frozen primary comparisons

Positive cost and severe-miss differences favor utility_regression (baseline minus utility).

| Baseline | Mean cost difference | Paired component bootstrap 95% CI | Five seeds lower | Severe miss no worse | Result |
|---|---:|---:|:---:|:---:|:---:|
| farthest | -0.050935 | [-0.114026, -0.000229] | no | no | NO-GO |
| max_building_privileged | -0.025325 | [-0.054784, 0.001634] | no | no | NO-GO |

## Per-seed component-macro contrasts

| Baseline | Seed | Utility cost | Baseline cost | Baseline - utility | Utility lower |
|---|---:|---:|---:|---:|:---:|
| farthest | 42 | 1.940226 | 1.842747 | -0.097479 | no |
| farthest | 123 | 1.729862 | 1.738685 | 0.008824 | yes |
| farthest | 456 | 1.807580 | 1.785714 | -0.021866 | no |
| farthest | 789 | 1.657970 | 1.637768 | -0.020202 | no |
| farthest | 1011 | 1.882820 | 1.758866 | -0.123954 | no |
| max_building_privileged | 42 | 1.940226 | 1.734916 | -0.205309 | no |
| max_building_privileged | 123 | 1.729862 | 1.714922 | -0.014939 | no |
| max_building_privileged | 456 | 1.807580 | 1.840896 | 0.033316 | yes |
| max_building_privileged | 789 | 1.657970 | 1.745548 | 0.087577 | yes |
| max_building_privileged | 1011 | 1.882820 | 1.855551 | -0.027268 | no |

## Decision rule and interpretation

The locked percentile bootstrap uses 10,000 dependency-component resamples with random seed 42. Each sampled component retains all five model-seed and paired-policy values.

GO requires both baselines to pass all three requirements: a strictly positive component-macro contrast in every seed, a strictly positive 95% CI lower bound, and a nonnegative baseline-minus-utility severe-miss contrast.

For random_mc32 descriptive metrics, trajectories are averaged within sample before dependency-component aggregation.

Adaptive STOP/ACQUIRE/DEFER results are secondary heuristics. No formal STOP calibration was performed because the current protocol has no independent policy-trajectory calibration role; the fail-closed variant disables STOP and the threshold-1 variant is risk-aware, not risk-controlled.

This is a within-CVIAN development confirmation with historical base exposure, not external confirmation. It evaluates the label-cost Phase-1 baseline and does not complete the separate attestation-coverage claim.
