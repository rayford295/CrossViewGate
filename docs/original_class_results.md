# Native/Ordinal-Class Results

These results avoid the old binary collapse. Hurricanes keep their native
3-class labels. The wildfire main setting uses the ordinal 3-class mapping:

- `0 = no_or_trace_damage`: `No Damage + Affected (1-9%)`
- `1 = damaged_repairable`: `Minor (10-25%) + Major (26-50%)`
- `2 = destroyed`: `Destroyed (>50%)`
- `Inaccessible` is excluded

The earlier wildfire 6-class run is kept as a native-label audit, but this
3-class mapping is the cleaner main comparison because the rare intermediate
classes are too small to support a stable headline result.

| dataset | mode | accuracy | macro_f1 | weighted_f1 | conflict_rate | accuracy_on_conflicts |
| --- | --- | --- | --- | --- | --- | --- |
| altadena_3class | street_only | 0.8835685483870968 | 0.7018802256996127 | 0.9155221992051884 | 0.14818548387096775 | 0.4217687074829932 |
| altadena_3class | remote_only | 0.9012096774193549 | 0.6871896564740064 | 0.9230050898968127 | 0.14818548387096775 | 0.5408163265306123 |
| altadena_3class | crossview | 0.9309475806451613 | 0.7083298743246212 | 0.9422754858383552 | 0.14818548387096775 | 0.7517006802721088 |
| ian_original | street_only | 0.7433333333333333 | 0.7440966893041887 | 0.7440966893041888 | 0.32666666666666666 | 0.6428571428571429 |
| ian_original | remote_only | 0.62 | 0.6232960011252409 | 0.6232960011252409 | 0.32666666666666666 | 0.2653061224489796 |
| ian_original | crossview | 0.7 | 0.7027560021602867 | 0.7027560021602868 | 0.32666666666666666 | 0.6122448979591837 |
| milton_original | street_only | 0.7322834645669292 | 0.738553035435484 | 0.7319050345045013 | 0.25984251968503935 | 0.5757575757575758 |
| milton_original | remote_only | 0.6811023622047244 | 0.6885254101640657 | 0.6771588556682515 | 0.25984251968503935 | 0.3787878787878788 |
| milton_original | crossview | 0.7716535433070866 | 0.7757172940247433 | 0.7727048456597571 | 0.25984251968503935 | 0.6060606060606061 |

## Reading

Altadena / Eaton wildfire supports the main cross-view mechanism under the
3-class ordinal setting:

- `crossview` is best overall by accuracy, macro-F1, and weighted-F1
- `crossview` is much stronger on conflict cases: `0.7517`
- conflict-case gain over the best single view is about `+0.2109`

Milton gives a matching hurricane-side cross-view result:

- `crossview` is best overall by macro-F1 (`0.7757`)
- `crossview` is also best on conflict cases (`0.6061`)

IAN remains the split-sensitive hurricane case:

- `street_only` is best overall (`0.7441` macro-F1)
- `crossview` is still much better than `remote_only`, but does not beat the
  strongest single view in this short native-label pilot

The cleanest current story is therefore:

> Cross-view fusion is strongest and most reliable as a conflict resolver. It
> clearly helps on the wildfire 3-class setting and Milton, while IAN shows that
> the identity of the best single view can still be dataset/split dependent.

## Robustness Layer

The paper-facing robustness layer is scripted and summarized through:

- `scripts/run_main_multiseed.ps1`: 3-seed main protocol with `concat`
  baseline, late-fusion baselines, bootstrap CIs, paired tests, and threshold
  sweeps. Main summaries are written to `docs/multiseed_main_results.md`,
  `docs/fusion_baselines_multiseed.md`, and
  `docs/conflict_statistics_multiseed.md`; threshold stability is summarized in
  `docs/threshold_sensitivity_multiseed.md`.
- `scripts/run_label_sensitivity_multiseed.ps1`: wildfire 3-class, wildfire
  6-class, and legacy binary sensitivity.
- `scripts/analyze_building_alignment_multi.py`: building pixel ratio,
  centered-building ratio, target-visibility proxy, and conflict-gain relation.
- `scripts/run_backbone_sanity.ps1`: ResNet50 or ConvNeXt sanity checks with
  conflict-subset analysis when single-view and cross-view modes are present.
