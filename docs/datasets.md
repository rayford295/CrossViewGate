# Datasets

## Overview

The main non-binary experiment compares one wildfire dataset and two hurricane
datasets without collapsing labels to a binary 0/1 target.

The core comparison is still:

- `street_only`
- `remote_only`
- `concat`
- `crossview`
- conflict subset where the two single-view models predict different classes

## 1. Eaton / Altadena Wildfire

### Data Character

- disaster type: wildfire
- ground-view regime: property-centric inspection imagery
- overhead view: paired remote-sensing crop

### Original Label Space

- `0 = No Damage`
- `1 = Affected (1-9%)`
- `2 = Minor (10-25%)`
- `3 = Major (26-50%)`
- `4 = Destroyed (>50%)`
- `5 = Inaccessible`

### Main 3-Class Protocol

- `0 = no_or_trace_damage`: `No Damage + Affected (1-9%)`
- `1 = damaged_repairable`: `Minor (10-25%) + Major (26-50%)`
- `2 = destroyed`: `Destroyed (>50%)`
- `Inaccessible` excluded

### Why This Dataset Matters

The ground image is usually tightly aligned with the target property, so the
cross-view bridge is expected to be stronger and more direct. The raw 6-class
setting is retained as an audit, but the 3-class protocol is the cleaner main
setting because `Minor`, `Major`, and `Inaccessible` are too sparse for a stable
headline result.

## 2. IAN Hurricane

### Data Character

- disaster type: hurricane
- ground-view regime: 360 / panoramic environmental views
- overhead view: paired remote-sensing crop
- georeference: 4,121 official CRS84 points from
  `02_Position/CVIAN_position.geojson`
- stable source key: Mapillary image id recovered from the official SHA-512
  manifest (the six-digit local filename is retained as `sample_id`)

### Original Label Space

- `0 = 0_MinorDamage`
- `1 = 1_ModerateDamage`
- `2 = 2_SevereDamage`

### Why This Dataset Matters

The panoramic ground image contains broader environmental context and weaker
direct alignment to the target structure. Keeping the moderate class makes the
task closer to the native dataset and tests whether cross-view conclusions hold
without endpoint-only simplification.

The main split is now grouped by 0.005-degree spatial block and uses a 25 m
cross-split buffer. The legacy source split is retained only as a leakage audit:
299 of its 300 test samples share a Mapillary sequence with train. See
[`CVIAN Georeference and Spatial-Split Audit`](results/cvian_georeference_split_audit.md).

## 3. Milton Hurricane / GenDisasterSVI

### Data Character

- disaster type: hurricane
- ground-view regime: generated or curated post-disaster SVI
- overhead view: paired post-disaster satellite image

### Original Label Space

- `0 = mild_damage`
- `1 = moderate_damage`
- `2 = severe_damage`

The manifest builder uses `dataset_with_post_sat.csv`, resolves Windows-local
paths under `hurrican-milton-GenDisasterSVI`, and pairs post-disaster SVI with
post-disaster satellite imagery.

The source table contains 2,555 rows (`train=2,047`, `val=254`, `test=254`).
The old builder dropped the source validation rows; the repaired builder audits
all rows and uses a 0.001-degree spatial-block split with a 25 m boundary buffer
for the main protocol. Pre-street, post-street, and post-overhead are all
retained, with explicit availability/dropout masks in the data loader. See the
[`Milton Manifest and Spatial-Split Audit`](results/milton_manifest_split_audit.md).

## Why the Comparison Is Useful

These three datasets let us compare:

- whether cross-view helps most on conflict cases
- whether the gain changes between property-centric wildfire and hurricane
  panoramic / SVI regimes
- whether the conclusion survives ordinal multiclass labels rather than a
  binary damage collapse

## Label Robustness Plan

Reviewer-facing label checks are organized as:

- wildfire 3-class main experiment
- wildfire 6-class audit
- legacy binary sensitivity

The key claim should be accepted only if the conflict-resolution pattern remains
directionally stable across these label settings, even if absolute macro-F1
changes with class imbalance.
