# Result documents

Headline protocol: converged 5-seed suite (`_v2` suffix). Non-suffixed files
are the earlier 3-epoch pilot, retained as a low-budget ablation.

## Headline (v2, converged 5 seeds)

| Document | Content |
| --- | --- |
| `multiseed_v2_results.md` | Main table, pooled-test headlines, v1-vs-v2 accounting |
| `reliability_gate_results_v2.md` | Gate variants, transfer matrix, linear coefficients |
| `calibration_decomposition_v2.md` | Per-view ECE/temperatures, calibrated fusion baselines, oracle-gap closure |
| `pooled_seed_tests_v2.md` | Pooled sign-flip paired tests for all key comparisons |
| `ordinal_metrics_v2.md` | QWK / MAE / extreme-error rates |
| `fov_intervention_results.md` | Causal field-of-view intervention (building vs random crops) |
| `conflict_density_maps.md` | Unsupervised conflict-density damage maps + uncertainty control + Moran's I |
| `cvdisaster_comparison.md` | Positioning vs CVDisaster / CVIAN (Li et al., 2025) |

## Pilot (3-epoch, 3 seeds)

`multiseed_main_results.md`, `reliability_gate_results.md`,
`calibration_decomposition.md`, `pooled_seed_tests.md`, `ordinal_metrics.md`,
`conflict_statistics_multiseed.md`, `fusion_baselines_multiseed.md`,
`threshold_sensitivity_multiseed.md`, `backbone_sanity_results.md`,
`building_alignment_main_seed42.md`, `label_sensitivity_*.md`.
