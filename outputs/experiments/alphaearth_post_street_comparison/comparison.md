# AlphaEarth vs Post-Disaster Street-View Baseline

| Dataset / event | Hazard | Eval split | AlphaEarth year | AlphaEarth model | AlphaEarth macro-F1 | Post-disaster street macro-F1 | Recovery | Gap |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| PrepStreet / Milton 2024 | Hurricane | 5-fold spatial GroupKFold | 2023 | random_forest_balanced | 0.5972 +/- 0.0733 | 0.6457 +/- 0.0949 | 92.5% | -0.0485 |
| CVIAN / Ian 2022 | Hurricane | spatial-block test | 2021 | logistic_balanced | 0.5687 | 0.6500 +/- 0.0131 | 87.5% | -0.0812 |
| Eaton / Altadena 2025 | Wildfire | study_development spatial role | 2024 | logistic_balanced | 0.4825 | 0.6856 +/- 0.0168 | 70.4% | -0.2030 |

Notes:

- Ian/Eaton AlphaEarth rows are single-seed tabular probes; post-street baselines are existing five-seed image runs where available.
- Eaton also produced a spatial-confirmation AlphaEarth result, but no matched post-street baseline exists for that role in the current outputs.
- RF alternatives: Ian AlphaEarth RF test macro-F1 = 0.5206; Eaton AlphaEarth RF study-development macro-F1 = 0.4972.
