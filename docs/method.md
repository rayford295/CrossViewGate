# Method

## Goal

Use a single triage framework to compare when cross-view fusion helps most.

## Model settings

We use three settings with a shared training and evaluation protocol:

1. `street_only`
2. `remote_only`
3. `crossview`

The backbone defaults to `ResNet18` for both branches in the current repo.
The training script now also supports explicit `street_augment` and
`overhead_augment` flags so augmentation policy can be kept consistent across
datasets.

## Cross-view model

The `crossview` model uses:

- one encoder for the ground-view image
- one encoder for the overhead image
- a late-fusion head over joint, difference, and elementwise-product features

This keeps the architecture simple and makes the mechanism comparison cleaner.

## Why conflict subset analysis is central

Average performance alone does not isolate the real contribution of cross-view
fusion.

We therefore define a `conflict subset`:

- build `street_only` predictions
- build `remote_only` predictions
- keep only samples where the two predictions disagree

This subset captures cases where the two modalities provide competing evidence.

## Evaluation logic

For each dataset, we report:

- overall test performance for `street_only`, `remote_only`, and `crossview`
- conflict-subset performance for the same models

The main paper question is not only:

> Is cross-view better on average?

but more importantly:

> How much does cross-view help when the two modalities conflict?

## Current interpretation

The working interpretation is that cross-view behaves primarily as a conflict
resolver, and its gain becomes larger when the ground-view image is tightly
aligned with the target structure.
