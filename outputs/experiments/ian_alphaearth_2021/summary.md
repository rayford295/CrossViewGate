# Ian AlphaEarth Validation

Features: `data/features/ian_alphaearth_2021.csv`
Embedding year: 2021
Seed: 13
Rows with all AlphaEarth features missing: 0

Trained on `train`; evaluated on `train`, `val`, and `test`.

| Model | Eval split | n | Macro-F1 | Balanced acc | Accuracy | F1 minor | F1 moderate | F1 severe |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_balanced | test | 415 | 0.5687 | 0.5731 | 0.5663 | 0.6466 | 0.5099 | 0.5496 |
| logistic_balanced | train | 3253 | 0.6321 | 0.6399 | 0.6317 | 0.7113 | 0.5374 | 0.6476 |
| logistic_balanced | val | 410 | 0.4528 | 0.4555 | 0.4512 | 0.4878 | 0.4379 | 0.4328 |
| random_forest_balanced | test | 415 | 0.5206 | 0.5276 | 0.5229 | 0.4898 | 0.5152 | 0.5568 |
| random_forest_balanced | train | 3253 | 0.9812 | 0.9818 | 0.9806 | 0.9778 | 0.9765 | 0.9893 |
| random_forest_balanced | val | 410 | 0.4594 | 0.4724 | 0.4683 | 0.3619 | 0.4762 | 0.5401 |

Reading guide:

- `test` is the repaired spatial-block test split.
- 2021 is the main pre-event embedding year for Hurricane Ian.
- 2022+ embeddings must be reported only as leakage probes.
