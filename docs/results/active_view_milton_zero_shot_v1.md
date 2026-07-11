# Milton zero-shot active-view sensitivity v1

**Status: sensitivity only; no GO/NO-GO decision and no confirmatory claim.**

The frozen CVIAN classifier and selectors were applied without Milton fitting, calibration, threshold selection, or policy selection. Milton influenced prior hypothesis development and has neither observed sequence IDs nor compass metadata.

## Registered estimand

The main policy rolls sector content so each physical origin maps to local sector 0. Random trajectories are averaged first, then eight origins, then five frozen model seeds within each observation before equal-unit macro aggregation. Positive contrasts below mean comparator cost minus relative-policy cost.

## Descriptive contrasts

| Comparator | Scope | Unit | Cost difference | 95% descriptive CI | Rotation median | Rotation range | Worst rotation |
|---|---|---|---:|---:|---:|---:|---:|
| farthest | comparator | dependency_group_id | 0.007531 | [-0.012983, 0.040839] | 0.003890 | 0.044559 | -0.010010 |
| farthest | comparator | spatial_block_id | 0.006218 | [-0.009734, 0.029054] | 0.006282 | 0.060151 | -0.030767 |
| clockwise | comparator | dependency_group_id | 0.061382 | [-0.060293, 0.159261] | 0.064829 | 0.091109 | 0.017373 |
| clockwise | comparator | spatial_block_id | 0.068581 | [-0.040095, 0.147292] | 0.055484 | 0.137630 | -0.006463 |
| random_mc32 | comparator | dependency_group_id | 0.024557 | [0.013279, 0.044898] | 0.025104 | 0.052999 | -0.003920 |
| random_mc32 | comparator | spatial_block_id | 0.021502 | [0.013730, 0.034251] | 0.026212 | 0.042650 | -0.002168 |
| absolute_aware_utility_ood_diagnostic | off_support_ood_diagnostic | dependency_group_id | -0.010051 | [-0.028279, 0.020493] | -0.006008 | 0.066845 | -0.043698 |
| absolute_aware_utility_ood_diagnostic | off_support_ood_diagnostic | spatial_block_id | -0.013303 | [-0.036674, 0.024762] | -0.013484 | 0.083917 | -0.051863 |
| max_confidence_privileged | privileged_diagnostic | dependency_group_id | 0.028447 | [-0.072601, 0.098270] | 0.033776 | 0.071919 | -0.014398 |
| max_confidence_privileged | privileged_diagnostic | spatial_block_id | 0.036724 | [-0.061391, 0.100551] | 0.041083 | 0.036012 | 0.018859 |
| greedy_label_oracle_privileged | privileged_diagnostic | dependency_group_id | -0.546511 | [-0.584329, -0.457449] | -0.550351 | 0.085328 | -0.580498 |
| greedy_label_oracle_privileged | privileged_diagnostic | spatial_block_id | -0.541983 | [-0.579659, -0.448803] | -0.543978 | 0.073770 | -0.578743 |

The primary descriptive bootstrap clusters the joint dependency-group/spatial-block connected components; dependency-group and spatial-block rows remain separate sensitivity estimands. Model seeds and origins are repeated measurements, not resampling units.

The absolute-aware result is explicitly an unrolled classifier-off-support OOD diagnostic. Max-confidence and label-oracle rows are privileged diagnostics. Neither may be promoted to headline transfer evidence.
