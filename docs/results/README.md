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
| `cvian_georeference_split_audit.md` | P0.1 official position join, legacy leakage, and new spatial/sequence protocols |
| `ian_split_performance_comparison.md` | P0.1 converged 5-seed legacy-vs-spatial performance correction |
| `milton_manifest_split_audit.md` | P0.3 source-val accounting, optional views, and repaired spatial protocol |
| `per_sample_evidence_schema.md` | P0.2 auditable probabilities, weights, visibility, conflict, and location schema |
| `dins_field_join_provenance.md` | P0.4 official DINS field snapshot, conservative spatial join, domains, and provenance |
| `selective_triage_protocol.md` | P0.7 finite-sample threshold calibration, claim rule, and output schema |
| `selective_triage_ian_spatial_v1.md` | Five-seed repaired-Ian gate/risk result and fail-closed three-stop routing case |
| `reliability_gate_ian_spatial_v1.md` | Strict gate-fit five-seed gate metrics and coefficients on repaired Ian |
| `routing_mvp_protocol.md` | Conflict-to-route priority policies, resource budgets, GeoJSON, and fail-closed report contract |
| `active_view_protocol.md` | CVIAN 8-sector development protocol: observation boundary, downstream-head-only role split, costs, dependencies, and future confirmatory requirements |
| `active_view_cvian_spatial_v1.md` | Five-seed exploratory fixed-budget result, descriptive block bootstrap, greedy one-step label-aware reference, No-Go decision, and consumed-test status |

## Pilot (3-epoch, 3 seeds)

`multiseed_main_results.md`, `reliability_gate_results.md`,
`calibration_decomposition.md`, `pooled_seed_tests.md`, `ordinal_metrics.md`,
`conflict_statistics_multiseed.md`, `fusion_baselines_multiseed.md`,
`threshold_sensitivity_multiseed.md`, `backbone_sanity_results.md`,
`building_alignment_main_seed42.md`, `label_sensitivity_*.md`.
