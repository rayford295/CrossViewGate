# Milton Manifest and Spatial-Split Audit

**Status:** source-row accounting and temporal/missing-view data support completed
on 2026-07-10; image-generation provenance verification remains open.

## Source accounting

All three local source tables contain 2,555 unique `pair_id` rows with the same
released partition:

| source split | rows |
| --- | ---: |
| train | 2,047 |
| validation | 254 |
| test | 254 |

The prior builder ignored every released validation row, split 307 rows back
out of source train, and therefore retained only 2,301 of 2,555 records. The
builder now audits all source values and retains the released train/validation/
test split when `--split-strategy source` is requested. Missing required
post-street or post-overhead images fail fast rather than being silently
dropped.

All 2,555 local rows currently have readable pre-street, post-street, and
post-overhead files. `CrossViewTriageDataset` also supports these views as
optional temporal inputs and returns separate real-missing and artificial-
dropout masks. The current triage model does not yet consume those masks in a
mask-aware fusion layer; that is a model workstream rather than a completed
data-loader claim.

## Spatial protocol

The released source partition is not a spatial holdout: each pair of partitions
contains zero-distance coordinate matches, and thousands of cross-partition
point pairs are within 25 m.

The repaired main protocol groups the compact Horseshoe Beach study area into
0.001-degree blocks (roughly 100 m) and purges lower-priority boundary samples
within 25 m of held-out data. Test is preserved before validation before train.

| split | rows | mild | moderate | severe |
| --- | ---: | ---: | ---: | ---: |
| train | 1,850 | 575 | 813 | 462 |
| validation | 255 | 90 | 93 | 72 |
| test | 270 | 77 | 138 | 55 |

The 180 excluded boundary rows (164 train, 16 validation) are recorded in
`buffer_exclusions.csv`. The retained split has zero shared spatial blocks and
zero cross-split point pairs within 25 m. Minimum cross-split distances are
25.46 m (train/validation), 25.37 m (train/test), and 25.47 m
(validation/test).

## Remaining provenance question

The table names, prompts, and source paths indicate a GenDisasterSVI generation
workflow. The code now preserves `prompt`, `split_source`, and all view paths,
but this audit does not establish which post-event street images are collected
versus generated. Until that source-level provenance is verified, Milton should
remain a sensitivity/transfer dataset and real CVIAN should carry the main
active-view claim.
