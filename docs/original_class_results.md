# Original-Class Results

These results use each dataset's native class space rather than a binary collapse.

Current status:

- IAN and Milton below are full-split 3-epoch pilot runs with ResNet18,
  balanced class weighting, and native 3-class labels.
- Altadena / Eaton original 6-class full-split training is prepared but not
  completed here because local image I/O is slow on the full split. A small
  stratified sanity run is available under `outputs/debug_original`.

| dataset | mode | accuracy | macro_f1 | weighted_f1 | conflict_rate | accuracy_on_conflicts |
| --- | --- | --- | --- | --- | --- | --- |
| ian_original | street_only | 0.7433333333333333 | 0.7440966893041887 | 0.7440966893041888 | 0.32666666666666666 | 0.6428571428571429 |
| ian_original | remote_only | 0.62 | 0.6232960011252409 | 0.6232960011252409 | 0.32666666666666666 | 0.2653061224489796 |
| ian_original | crossview | 0.7 | 0.7027560021602867 | 0.7027560021602868 | 0.32666666666666666 | 0.6122448979591837 |
| milton_original | street_only | 0.7322834645669292 | 0.738553035435484 | 0.7319050345045013 | 0.25984251968503935 | 0.5757575757575758 |
| milton_original | remote_only | 0.6811023622047244 | 0.6885254101640657 | 0.6771588556682515 | 0.25984251968503935 | 0.3787878787878788 |
| milton_original | crossview | 0.7716535433070866 | 0.7757172940247433 | 0.7727048456597571 | 0.25984251968503935 | 0.6060606060606061 |

## Reading

Milton gives the cleanest current native-label cross-view result:

- `crossview` is best overall by macro-F1 (`0.7757`)
- `crossview` is also best on conflict cases (`0.6061`)
- `remote_only` is weaker overall and on conflicts, even though satellite
  evidence is useful for severe damage

IAN is more split-sensitive in this short pilot:

- `street_only` is best overall (`0.7441` macro-F1)
- `crossview` is close but lower overall (`0.7028`)
- `crossview` is still much better than `remote_only`, and is close to
  `street_only` on conflict cases (`0.6122` vs `0.6429`)

So the native-label conclusion should be more nuanced than the old binary
story: cross-view clearly helps on Milton, remains useful on IAN, but does not
automatically dominate the strongest single view in every hurricane split.
