# AlphaEarth API Usage

This repository uses Google Earth Engine to sample AlphaEarth Foundations
annual embeddings for disaster-damage validation experiments.

## Earth Engine Project

Current project used for the local runs:

```text
Project ID: extended-acumen-502205-v5
Project number: 890743338207
```

The project must have:

1. Google Earth Engine API enabled.
2. Earth Engine project registration completed.
3. Local OAuth credentials created with `earthengine authenticate`.

Set the default project locally:

```powershell
earthengine set_project extended-acumen-502205-v5
```

Do not commit Earth Engine credentials. The local credential file is:

```text
%USERPROFILE%\.config\earthengine\credentials
```

## AlphaEarth Dataset

Earth Engine dataset:

```text
GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL
```

Band schema:

```text
A00 ... A63
```

The validation scripts export 128 features per sample:

1. 64 point-sampled AlphaEarth dimensions.
2. 64 neighborhood-mean dimensions from a 30 m buffer.

## Leakage Rule

Use annual embeddings strictly before the disaster event year for hurricane
preparedness-style validation:

| Dataset / event | Event year | Main AlphaEarth year | Note |
| --- | ---: | ---: | --- |
| CVIAN / Ian | 2022 | 2021 | 2022+ is leakage-probe only |
| PrepStreet / Milton | 2024 | 2023 | 2024+ is leakage-probe only |
| Eaton / Altadena wildfire | 2025 | 2024 | 2024 is pre-event |

## Reproduction Commands

```powershell
cd <path-to-CrossViewGate>

earthengine set_project extended-acumen-502205-v5

python scripts\ian_alphaearth_validation.py `
  --extract `
  --ee-project extended-acumen-502205-v5

python scripts\eaton_alphaearth_validation.py `
  --extract `
  --ee-project extended-acumen-502205-v5
```

## Tracked AlphaEarth Artifacts

The branch `codex/alphaearth-external-validation` intentionally tracks the
AlphaEarth feature and result artifacts needed to reuse the experiment without
re-querying Earth Engine:

```text
data/features/ian_alphaearth_2021.csv
data/features/ian_alphaearth_locations.csv
data/features/eaton_alphaearth_2024.csv
data/features/eaton_alphaearth_locations.csv

outputs/experiments/ian_alphaearth_2021/
outputs/experiments/eaton_alphaearth_2024/
outputs/experiments/alphaearth_post_street_comparison/
```

The OAuth credential file and any local Google account tokens remain untracked.

## Result Summary

The main interpretation table is documented in:

```text
docs/results/alphaearth_external_validation.md
```
