# AlphaEarth External Validation

This note compares pre-event AlphaEarth annual embeddings against the
post-disaster street-view assessment baseline. The central quantity is:

```text
recovery ratio = AlphaEarth macro-F1 / post-disaster street-view macro-F1
```

## Main Comparison

| Dataset / event | Hazard type | Post-disaster street baseline | AlphaEarth pre-event year | AlphaEarth macro-F1 | Post-disaster street macro-F1 | Recovery ratio | Gap |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| PrepStreet / Milton 2024 | Hurricane | `oracle_post_image_deep` | 2023 | **0.5972 +/- 0.0733** | **0.6457 +/- 0.0949** | **92.5%** | -0.0485 |
| CVIAN / Ian 2022 | Hurricane | `street_only` post-SVI test | 2021 | **0.5687** | **0.6500 +/- 0.0131** | **87.5%** | -0.0812 |
| Eaton / Altadena 2025 | Wildfire | `street_only` post-disaster field image | 2024 | **0.4825** | **0.6856 +/- 0.0168** | **70.4%** | -0.2030 |

## Reading

The hurricane results are directionally strong across two independent datasets:
AlphaEarth recovers about 88-93% of the post-disaster street-view baseline.
This supports the claim that annual overhead foundation embeddings encode a
substantial pre-event vulnerability signal for hurricane damage, not merely a
Horseshoe Beach-specific artifact.

The wildfire result is meaningfully weaker. Eaton AlphaEarth still carries
signal, but the gap to post-disaster field imagery is much larger. This is
consistent with wildfire damage being more structure-specific and event-process
specific: ember exposure, defensible space, construction details, suppression,
local fuel continuity, and burn progression are not fully recoverable from an
annual pre-event overhead embedding.

The practical interpretation is not that AlphaEarth replaces street-view
evidence. A better framing is:

1. AlphaEarth is a strong low-cost pre-event risk prior for hurricane damage.
2. Street-view evidence remains essential for explaining household- or
   structure-level vulnerability and for wildfire cases.
3. Cross-hazard validation should report both positive transfer and failure
   modes, rather than treating one score as universal.

## Protocol Notes

- Milton uses the existing PrepStreet 5-fold spatial GroupKFold results.
- Ian uses the repaired CVIAN spatial-block split, with AlphaEarth 2021 as the
  main pre-event embedding year for Hurricane Ian 2022.
- Eaton uses AlphaEarth 2024 as the pre-event embedding year for the 2025 Eaton
  fire.
- Ian and Eaton AlphaEarth rows are single-seed tabular probes. Existing
  post-disaster street baselines are five-seed image-model means where
  available.
- For Ian/Eaton, the linear `logistic_balanced` AlphaEarth probe is reported in
  the main table because random forests overfit the training split. The RF
  alternatives were: Ian test macro-F1 0.5206; Eaton study-development macro-F1
  0.4972.
- Eaton also produced a spatial-confirmation AlphaEarth result
  (`logistic_balanced` macro-F1 0.4367; `random_forest_balanced` macro-F1
  0.4426), but the current outputs do not include a matched post-disaster
  street-view baseline for that same role, so it is excluded from the recovery
  ratio table.

## Generated Local Artifacts

These generated files are intentionally ignored by git:

```text
data/features/ian_alphaearth_2021.csv
data/features/eaton_alphaearth_2024.csv
outputs/experiments/ian_alphaearth_2021/metrics.csv
outputs/experiments/eaton_alphaearth_2024/metrics.csv
outputs/experiments/alphaearth_post_street_comparison/comparison.csv
outputs/experiments/alphaearth_post_street_comparison/comparison.md
```

## Reproduction

See `docs/alphaearth_api_usage.md` for Earth Engine project setup, leakage
rules, and the tracked AlphaEarth artifacts.

```powershell
earthengine set_project extended-acumen-502205-v5

python scripts\ian_alphaearth_validation.py `
  --extract `
  --ee-project extended-acumen-502205-v5

python scripts\eaton_alphaearth_validation.py `
  --extract `
  --ee-project extended-acumen-502205-v5
```
