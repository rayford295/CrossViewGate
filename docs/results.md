# Results Summary

## Main comparison

| Dataset | Ground-view regime | street_only F1 | remote_only F1 | crossview F1 |
|---|---|---:|---:|---:|
| Eaton wildfire | building / property-centric | 0.9604 | 0.9653 | 0.9713 |
| IAN hurricane | 360 / panoramic environment-centric | 0.8912 | 0.9082 | 0.9208 |

## Conflict subset comparison

| Dataset | Conflict rate | street_only | remote_only | crossview |
|---|---:|---:|---:|---:|
| Eaton wildfire test | 0.0338 | 0.4179 | 0.5821 | 0.7612 |
| IAN hurricane test | 0.1050 | 0.4286 | 0.5714 | 0.6190 |

## Conflict subset bootstrap intervals

| Dataset | street_only | remote_only | crossview |
|---|---|---|---|
| Eaton wildfire test | 0.4179 [0.2985, 0.5373] | 0.5821 [0.4627, 0.7015] | 0.7612 [0.6567, 0.8507] |
| IAN hurricane test | 0.4286 [0.2381, 0.6190] | 0.5714 [0.3810, 0.7619] | 0.6190 [0.4286, 0.8095] |

## Working interpretation

Three patterns are already clear.

1. Cross-view fusion is the best overall setting in both disasters.
2. The strongest gain appears on the conflict subset rather than on easy average cases.
3. The gain is larger when the ground image is tightly aligned with the target structure.

At the same time, the wildfire-vs-hurricane gap on the conflict subset should
currently be described as a supported trend rather than a hard significance
claim, because the bootstrap intervals still overlap.

This makes the paper story mechanism-oriented:

> Cross-view fusion acts primarily as a conflict resolver, and its value depends
> on how directly the ground-view image captures the target building.

## Lightweight alignment proxy

We estimated a simple target-alignment proxy on the conflict subsets using a
frozen semantic-segmentation model with a `building` class.

| Dataset | building ratio mean | center building ratio mean | centroid distance mean |
|---|---:|---:|---:|
| Eaton wildfire conflict | 0.2684 | 0.4101 | 0.2958 |
| IAN hurricane conflict | 0.0154 | 0.0271 | 0.3965 |

Interpretation:

- wildfire images contain much more visible building area
- wildfire building evidence is more centrally located
- hurricane images are far more environment-dominant

This supports the claim that cross-view gain is stronger in wildfire because
the ground-view image is more tightly aligned with the target structure.

## Permutation tests

| Dataset | crossview accuracy | label-independence p | crossview vs street p | crossview vs remote p |
|---|---:|---:|---:|---:|
| Eaton wildfire conflict | 0.7612 | < 1e-4 | 0.0147 | 0.5276 |
| IAN hurricane conflict | 0.6190 | 0.2639 | 0.7266 | 1.0000 |

Interpretation:

- wildfire conflict performance is statistically convincing
- hurricane remains directionally positive, but not yet strong enough for a hard significance claim

## Threshold sensitivity

Crossview remains the strongest setting across softer conflict definitions.

Selected points:

| Dataset | Threshold | Conflict rate | street | remote | crossview |
|---|---:|---:|---:|---:|---:|
| Eaton wildfire | 0.1 | 0.0549 | 0.6239 | 0.7248 | 0.8349 |
| Eaton wildfire | 0.3 | 0.0393 | 0.4872 | 0.6410 | 0.7821 |
| Eaton wildfire | 0.5 | 0.0327 | 0.4154 | 0.5846 | 0.7538 |
| IAN hurricane | 0.1 | 0.1550 | 0.5806 | 0.6774 | 0.7097 |
| IAN hurricane | 0.3 | 0.1250 | 0.4800 | 0.6000 | 0.6400 |
| IAN hurricane | 0.5 | 0.1050 | 0.4286 | 0.5714 | 0.6190 |

## Per-sample alignment correlation

We tested whether the building-ratio proxy also predicts crossview success at the
single-example level on the conflict subset.

| Setting | Spearman r | p-value |
|---|---:|---:|
| wildfire building_ratio vs crossview_correct | -0.1150 | 0.3528 |
| hurricane building_ratio vs crossview_correct | 0.4212 | 0.0609 |
| combined building_ratio vs crossview_correct | 0.0708 | 0.5173 |

This does **not** currently support a strong per-sample monotonic effect.
The alignment story is therefore better framed as a `view-regime / dataset-level`
mechanism than as a within-conflict ranking signal.

## Qualitative figure

A paper-ready qualitative figure is now available:

![Qualitative conflict examples](./assets/qualitative_conflict_examples.png)

## Label note for hurricane

The hurricane benchmark uses endpoint-to-endpoint binary classification:

- `MinorDamage -> 0`
- `SevereDamage -> 1`
- `ModerateDamage` excluded

This is intentional. The goal is to maximize label clarity and isolate the
effect of ground-view regime, rather than optimize for a broader but noisier
binary collapse.
