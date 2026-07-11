# CVIAN Sequence-Held-Out Four-Role Protocol

**Status:** protocol built and audited on 2026-07-10. The prospective test has
not been scored under this protocol.

## Outcome

The new protocol separates all active-view learning decisions into four roles:

| role | rows | sequences | spatial blocks | minor | moderate | severe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base fit | 1,130 | 119 | 153 | 451 | 392 | 287 |
| selector fit | 662 | 18 | 44 | 201 | 267 | 194 |
| validation | 372 | 8 | 29 | 121 | 126 | 125 |
| prospective test | 137 | 22 | 22 | 66 | 50 | 21 |

Every pair of roles has zero shared `sample_id`, zero shared `sequence_id`,
zero shared `spatial_block_id`, and no point pair within or on the 25 m buffer.
The minimum pairwise distances range from 25.30 m to 488.35 m.

The generated artifacts are under
`data/splits/ian_hurricane_sequence_four_role_v1/`. This directory is ignored
by Git and must be regenerated locally.

## What “prospective test” means

The spatial-v1 active-view study fitted a different selector subset for each
of seeds 42, 123, 456, 789, and 1011. Its validation labels selected both the
classifier and selector epochs, and its development test was consumed while
correcting the experiment. Therefore, merely creating another random
sequence split would not create a defensible new policy test.

The builder exactly reconstructs the five historical selector-fit block
assignments. It marks as selection-exposed the union of:

- 2,331 unique spatial-v1 training samples used by at least one selector fit;
- all 410 validation samples used for early stopping and temperature fitting;
- all 415 samples in the consumed development test.

This union contains 3,156 samples. Although 965 CVIAN samples are outside that
union, most share a Mapillary sequence with an exposed sample. The prospective
test is fixed to **all** 22 complete sequences with zero direct selection
exposure, leaving 137 rows. It was not chosen by policy performance.

The status string is
`selector_selection_holdout_with_historical_base_exposure`. All 137 test rows
were present in the old spatial-v1 training role and therefore entered the
historical base/model-fit path. Consequently:

- the old cross-view checkpoints, fitted heads, sector embeddings, and logits
  must not be reused;
- every base encoder and classifier must be retrained only on `base_fit.csv`;
- the result is not a historically never-seen or external-event confirmation;
- after full retraining, it can support one development confirmation of the
  selector/policy under this declared split.

The machine-readable commitment is `test_commitment.json`. Its current sample
set SHA-256 is
`4e138ba9610ba69d96cdfc9addf2df25782fb33cfad6a8cb589b08528663ee3b`.

## Spatial and sequence isolation

Rows are first assigned by whole `sequence_id`. The prospective test is
protected first, followed by validation, selector fit, and base fit. Any row
whose 0.005-degree block is already owned by a higher-priority role is removed.
A 25 m point buffer is then applied in the same order.

| role | raw rows | shared-block exclusions | distance exclusions | retained |
| --- | ---: | ---: | ---: | ---: |
| base fit | 2,775 | 1,644 | 1 | 1,130 |
| selector fit | 802 | 139 | 1 | 662 |
| validation | 407 | 35 | 0 | 372 |
| prospective test | 137 | 0 | 0 | 137 |

The strict isolation removes 1,820 rows. This cost is not accidental: the
sequence/spatial-block bipartite graph has 48 connected components, and its
largest component contains 3,267 rows, 77 sequences, 121 blocks, and 1,094 of
the 1,135 severe samples. Keeping every row while simultaneously making four
balanced sequence- and block-disjoint roles is therefore incompatible with
this dataset geometry. Twenty-seven assigned fit sequences lose all rows under
the strict block priority. The exclusion CSV records every removed sample and
the protected role responsible for its removal.

## Required experimental order

1. Train every image encoder, cross-view fusion model, and mask-aware
   classifier only on `base_fit.csv`.
2. Freeze those artifacts and build out-of-fit embeddings for
   `selector_fit.csv` before fitting utility/action/stopping models.
3. Use `validation.csv` for architecture, utility target, budget, view cost,
   stopping/defer rule, temperature, and epoch selection.
4. Freeze code, configuration, input hashes, and all model artifacts.
5. Score `prospective_test.csv` once. Do not return to steps 1–3 after viewing
   its outcomes.

With only 137 samples, 22 sequences, and 21 severe examples, this test cannot
support tight distribution-free risk claims. A strong final claim still needs
an event-held-out evaluation such as Milton or newly collected data.

## Reproduction

```powershell
python scripts/build_cvian_sequence_four_role_manifests.py `
  --source-split-dir data/splits/ian_hurricane_original_sequence `
  --historical-spatial-split-dir data/splits/ian_hurricane_original `
  --output-dir data/splits/ian_hurricane_sequence_four_role_v1 `
  --base-fit-fraction 0.70 `
  --selector-fit-fraction 0.20 `
  --validation-fraction 0.10 `
  --spatial-buffer-m 25 `
  --historical-selector-seeds 42,123,456,789,1011 `
  --seed 20260710
```

The builder writes role manifests, the test commitment, source and output
hashes, historical exposure records, sequence assignments, dependency
components, exclusions, class counts, and pairwise leakage/distance audits. It
fails on incomplete coordinates, duplicate ids, missing classes, sequence or
block overlap, or a cross-role pair at or below the spatial buffer.
