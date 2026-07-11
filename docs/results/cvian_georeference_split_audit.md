# CVIAN Georeference and Spatial-Split Audit

**Status:** roadmap P0.1 completed on 2026-07-10, including the converged
five-seed performance comparison.

## Official source and integrity

The position file is the official
`CVIAN/02_Position/CVIAN_position.geojson` from the
[TUM CVIAN release](https://doi.org/10.14459/2024mp1749324), whose structure is
documented by the [CVDisaster repository](https://github.com/tum-bgd/CVDisaster).

- File size: 1,434,792 bytes
- SHA-512:
  `579e4e9ee2d130a1922fd7e91600292737f694cf06473d66b257beb5fd0a60c4ec1e9a5d90a49f4f9f5f7e9caf46ff8977d37b0aa2b38f2d471cdced1625960a`
- CRS: OGC CRS84 / WGS84, point coordinates ordered as longitude, latitude
- Position features: 4,121 unique Mapillary image ids
- Mapillary sequences: 194
- Bounds: longitude -82.197829 to -81.829535; latitude 26.423504 to 26.728957

The local files had been renamed to six-digit ids. Joining by GeoJSON feature
order would be wrong. `scripts/georeference_ian_hurricane.py` instead hashes
every local image and looks it up in the release's `checksums.sha512`:

- 8,242 / 8,242 local images matched an official checksum;
- every satellite/SVI pair recovered the same source Mapillary id and severity;
- 4,121 / 4,121 recovered ids joined one-to-one with the position features;
- no image, position, or id was silently dropped.

The script preserves the raw CSVs and writes auditable sidecars under the local
dataset's `georeferenced/` directory: the id map, enriched pairs/splits, a
local-id GeoJSON, and a provenance summary.

## Legacy split leakage

The released local split cannot be treated as a spatial or object-grouped
holdout:

- 299 / 300 legacy test samples share `sequence_id` with train;
- 113 / 114 legacy test sequences also occur in train;
- 297 / 300 legacy test samples occupy a 0.005-degree block also seen in train;
- after creating the old random validation subset, 151 sequence groups and 171
  spatial blocks occur in more than one train/validation/test partition.

The old builder compounded this problem by assigning a row number as
`objectid`, a unique sample id as `group_id`, and null coordinates. Those fields
have been replaced with the official Mapillary id, sequence id, coordinates,
and spatial block.

## New protocols

The main protocol is a deterministic, approximately label-stratified split over
0.005-degree spatial blocks. It also purges lower-priority boundary samples
within 25 m of a held-out split, preserving test before validation before train.

| split | rows | minor | moderate | severe |
| --- | ---: | ---: | ---: | ---: |
| train | 3,253 | 1,122 | 1,202 | 929 |
| validation | 410 | 132 | 161 | 117 |
| test | 415 | 140 | 159 | 116 |

Forty-three boundary samples are recorded in `buffer_exclusions.csv` (36 train,
7 validation). The retained protocol has:

- zero shared `spatial_block_id` values across splits;
- zero cross-split pairs within 25 m;
- minimum train/validation, train/test, and validation/test distances of 25.37,
  25.19, and 26.26 m, respectively.

A sequence-grouped sensitivity split is also generated. It has zero shared
sequence ids but 46 spatial blocks cross partitions, so it is not the primary
spatial-holdout protocol. The two variants expose the trade-off rather than
claiming that a single grouping key removes every form of dependence.

## Reproduction

```powershell
python scripts/georeference_ian_hurricane.py `
  --dataset-root "C:\path\to\IAN_hurricane"

python scripts/build_ian_hurricane_manifests.py `
  --dataset-root "C:\path\to\IAN_hurricane" `
  --output-dir data/splits/ian_hurricane_original `
  --task original `
  --split-strategy spatial-block `
  --spatial-grid-degrees 0.005 `
  --spatial-buffer-m 25

python scripts/check_split_leakage.py `
  --train data/splits/ian_hurricane_original/train.csv `
  --val data/splits/ian_hurricane_original/val.csv `
  --test data/splits/ian_hurricane_original/test.csv `
  --group-cols spatial_block_id `
  --spatial-threshold-m 25
```

## P0.1 performance result

All prior Ian checkpoints were fitted/evaluated against the leaked source
split and remain historical baselines only. The completed five-seed repaired
run reduces macro-F1 by 0.049 (concat), 0.097 (crossview), 0.067 (remote only),
and 0.056 (street only) relative to that history. Full accuracy, severe recall,
seed variability, and bootstrap accounting are reported in
[`ian_split_performance_comparison.md`](ian_split_performance_comparison.md).
