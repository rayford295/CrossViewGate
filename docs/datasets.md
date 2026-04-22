# Datasets

## Overview

This paper repo compares two disaster datasets under a unified binary triage
protocol.

The purpose is not only to compare disasters, but to compare two different
ground-view regimes:

- `building/property-centric` ground views
- `360/panoramic environment-centric` ground views

## 1. Eaton / Altadena wildfire

### Data character

- disaster type: wildfire
- ground-view regime: property-centric inspection imagery
- overhead view: paired remote-sensing crop

### Binary triage protocol

- `0 = No Damage + Affected`
- `1 = Minor + Major + Destroyed`
- `Inaccessible` excluded

### Why this dataset matters

The ground image is usually tightly aligned with the target property, so the
cross-view bridge is expected to be stronger and more direct.

## 2. IAN hurricane

### Data character

- disaster type: hurricane
- ground-view regime: 360 / panoramic environmental views
- overhead view: paired remote-sensing crop

### Original label space

- `0_MinorDamage`
- `1_ModerateDamage`
- `2_SevereDamage`

### Binary triage protocol used in this repo

- `0 = MinorDamage`
- `1 = SevereDamage`
- `ModerateDamage` excluded

### Why this dataset matters

The panoramic ground image contains broader environmental context and weaker
direct alignment to the target structure. This makes it a useful contrast case
for the wildfire dataset.

## Why the comparison is useful

These two datasets let us compare not only disaster types, but also the
mechanism of cross-view value:

- Does cross-view help most on conflict cases?
- Does that help depend on how directly the ground view captures the target
  building?
