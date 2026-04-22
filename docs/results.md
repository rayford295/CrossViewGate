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

## Label note for hurricane

The hurricane benchmark uses endpoint-to-endpoint binary classification:

- `MinorDamage -> 0`
- `SevereDamage -> 1`
- `ModerateDamage` excluded

This is intentional. The goal is to maximize label clarity and isolate the
effect of ground-view regime, rather than optimize for a broader but noisier
binary collapse.
