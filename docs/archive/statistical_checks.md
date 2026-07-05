# Statistical Checks

## What was added

To make the paper more submission-ready, we added:

1. permutation tests on the conflict subset
2. threshold-sensitivity analysis for soft conflict definitions
3. per-sample alignment correlation
4. a qualitative conflict figure

## 1. Permutation tests

| Dataset | crossview accuracy | label-independence p | crossview vs street p | crossview vs remote p |
|---|---:|---:|---:|---:|
| Eaton wildfire conflict | 0.7612 | < 1e-4 | 0.0147 | 0.5276 |
| IAN hurricane conflict | 0.6190 | 0.2639 | 0.7266 | 1.0000 |

Interpretation:

- wildfire conflict performance is statistically persuasive
- hurricane points in the same direction but the current conflict subset is too small for a hard significance claim

## 2. Threshold sensitivity

Crossview remains the best-performing setting under softer conflict thresholds.

![Wildfire threshold sensitivity](./assets/wildfire_threshold_sensitivity.png)
![Hurricane threshold sensitivity](./assets/hurricane_threshold_sensitivity.png)

This supports the idea that the crossview advantage is not an artifact of a single hard conflict definition.

## 3. Per-sample alignment correlation

| Setting | Spearman r | p-value |
|---|---:|---:|
| wildfire building_ratio vs crossview_correct | -0.1150 | 0.3528 |
| hurricane building_ratio vs crossview_correct | 0.4212 | 0.0609 |
| combined building_ratio vs crossview_correct | 0.0708 | 0.5173 |

![Alignment gain correlation](./assets/alignment_gain_correlation.png)

This is an important negative result:

- the alignment proxy is strong at the dataset-regime level
- but it does not yet become a strong monotonic predictor of per-sample crossview success

So the current paper should not overclaim sample-level mechanism.

## 4. Qualitative figure

![Qualitative conflict examples](./assets/qualitative_conflict_examples.png)

The figure shows:

- wildfire success cases where crossview resolves disagreement
- hurricane success cases where crossview also helps
- a wildfire failure case where crossview follows the wrong branch

This makes the paper more balanced and visually interpretable.
