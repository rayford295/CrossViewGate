# Original-Class Results

These results use each dataset's native class space rather than a binary collapse.

Current status:

- Altadena / Eaton wildfire is now complete on the full original 6-class split.
- IAN and Milton are complete on full original 3-class splits.
- All runs use ResNet18, balanced class weighting, and the three view modes:
  `street_only`, `remote_only`, and `crossview`.

| dataset | mode | accuracy | macro_f1 | weighted_f1 | conflict_rate | accuracy_on_conflicts |
| --- | --- | --- | --- | --- | --- | --- |
| altadena_original | street_only | 0.84665991902834 | 0.40750110685574964 | 0.8608540373285545 | 0.20951417004048584 | 0.4106280193236715 |
| altadena_original | remote_only | 0.8446356275303644 | 0.3951352954888799 | 0.8557888162564686 | 0.20951417004048584 | 0.40096618357487923 |
| altadena_original | crossview | 0.868421052631579 | 0.4345426185357619 | 0.8824993340898941 | 0.20951417004048584 | 0.5434782608695652 |
| ian_original | street_only | 0.7433333333333333 | 0.7440966893041887 | 0.7440966893041888 | 0.32666666666666666 | 0.6428571428571429 |
| ian_original | remote_only | 0.62 | 0.6232960011252409 | 0.6232960011252409 | 0.32666666666666666 | 0.2653061224489796 |
| ian_original | crossview | 0.7 | 0.7027560021602867 | 0.7027560021602868 | 0.32666666666666666 | 0.6122448979591837 |
| milton_original | street_only | 0.7322834645669292 | 0.738553035435484 | 0.7319050345045013 | 0.25984251968503935 | 0.5757575757575758 |
| milton_original | remote_only | 0.6811023622047244 | 0.6885254101640657 | 0.6771588556682515 | 0.25984251968503935 | 0.3787878787878788 |
| milton_original | crossview | 0.7716535433070866 | 0.7757172940247433 | 0.7727048456597571 | 0.25984251968503935 | 0.6060606060606061 |

## Reading

Altadena / Eaton wildfire now supports the main cross-view mechanism under the
native 6-class setting:

- `crossview` is best overall by macro-F1 (`0.4345`) and weighted-F1 (`0.8825`)
- `crossview` is clearly best on conflict cases (`0.5435`)
- conflict-case gain over the best single view is about `+0.1329`

Milton gives the cleanest hurricane-side cross-view result:

- `crossview` is best overall by macro-F1 (`0.7757`)
- `crossview` is also best on conflict cases (`0.6061`)
- `remote_only` is weaker overall and on conflicts

IAN is the split-sensitive case:

- `street_only` is best overall (`0.7441` macro-F1)
- `crossview` remains much better than `remote_only`, but does not beat
  `street_only` on this short original-label run
- on conflict cases, `crossview` is close to `street_only`

The native-label conclusion is therefore more nuanced than the earlier binary
story: cross-view clearly helps on Altadena and Milton, while IAN shows that
the strongest single view can still dominate in some hurricane splits.
