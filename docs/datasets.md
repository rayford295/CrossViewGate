# Datasets

## Overview

The original-class experiment compares one wildfire dataset and two hurricane
datasets without collapsing labels to a binary 0/1 target.

The core comparison is still:

- `street_only`
- `remote_only`
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

### Why This Dataset Matters

The ground image is usually tightly aligned with the target property, so the
cross-view bridge is expected to be stronger and more direct. The original
6-class setting is highly imbalanced, so macro-F1 and per-class metrics matter
more than accuracy alone.

## 2. IAN Hurricane

### Data Character

- disaster type: hurricane
- ground-view regime: 360 / panoramic environmental views
- overhead view: paired remote-sensing crop

### Original Label Space

- `0 = 0_MinorDamage`
- `1 = 1_ModerateDamage`
- `2 = 2_SevereDamage`

### Why This Dataset Matters

The panoramic ground image contains broader environmental context and weaker
direct alignment to the target structure. Keeping the moderate class makes the
task closer to the native dataset and tests whether cross-view conclusions hold
without endpoint-only simplification.

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

## Why the Comparison Is Useful

These three datasets let us compare:

- whether cross-view helps most on conflict cases
- whether the gain changes between property-centric wildfire and hurricane
  panoramic / SVI regimes
- whether the conclusion survives native multiclass labels rather than a binary
  damage collapse
