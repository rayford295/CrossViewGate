# Eaton AlphaEarth External Validation

Features: `C:\Users\yyang295\Documents\New project\CrossViewGate\data\features\eaton_alphaearth_2024.csv`
Embedding year: 2024 (pre-event for Eaton 2025)
Seed: 13
Rows with all AlphaEarth features missing: 0

Trained on `model_fit`; evaluated on all frozen Eaton roles.

| Model | Eval role | n | Macro-F1 | Balanced acc | Accuracy | F1 no/trace | F1 repairable | F1 destroyed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic_balanced | model_fit | 6755 | 0.5219 | 0.7098 | 0.6768 | 0.7182 | 0.0976 | 0.7501 |
| logistic_balanced | model_validation | 1284 | 0.4497 | 0.4928 | 0.6012 | 0.6553 | 0.0474 | 0.6463 |
| logistic_balanced | spatial_confirmation | 5484 | 0.4367 | 0.4250 | 0.6016 | 0.6284 | 0.0093 | 0.6724 |
| logistic_balanced | study_development | 1534 | 0.4825 | 0.4750 | 0.6551 | 0.6906 | 0.0144 | 0.7426 |
| random_forest_balanced | model_fit | 6755 | 0.9953 | 0.9953 | 0.9930 | 0.9928 | 1.0000 | 0.9931 |
| random_forest_balanced | model_validation | 1284 | 0.4273 | 0.4319 | 0.6363 | 0.6477 | 0.0000 | 0.6341 |
| random_forest_balanced | spatial_confirmation | 5484 | 0.4426 | 0.4453 | 0.6601 | 0.6553 | 0.0000 | 0.6724 |
| random_forest_balanced | study_development | 1534 | 0.4972 | 0.5036 | 0.7412 | 0.7459 | 0.0000 | 0.7457 |

Reading guide:

- `model_validation` is the development validation role.
- `spatial_confirmation` is the reserved same-event spatial confirmation role.
- Eaton class imbalance is severe in the middle class; macro-F1 is the primary score.
