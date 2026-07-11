# CVIAN active-view sequential-reveal development protocol

## Claim scope

This document records a **development/exploratory offline sequential evidence
reveal** analysis on CVIAN. It does not claim that a vehicle, field team, or UAV
acquired a new view, and it does not inherit the earlier finite-sample
`risk-controlled` claim. The current artifacts do not provide an immutable
preregistration record: the exact reporting implementation was finalized around
the initial test analysis. The spatial test has therefore been consumed and
cannot support a later confirmatory selector claim.

Results are `risk-aware`. The test contains six spatial blocks, but five
Mapillary sequences span two test blocks and connect three of the blocks into
one dependency component. There are only four sequence-connected dependency
components, so the six blocks must not be described as six independent test
units.

## Geometry and episode

- Source: 4,121 checksum-verified `1024x512` RGB equirectangular panoramas.
- Split first, crop second: 3,253 spatial train, 410 validation, 415 spatial
  development-test rows, and 43 boundary exclusions. Sector rows never receive
  a new split.
- Eight sector centers are stored in signed normalization as relative yaw
  `0, 45, 90, 135, -180, -135, -90, -45` degrees (equivalent modulo 360 to
  `0..315` in 45-degree steps). Each crop is the central `256x256`
  equirectangular window (90-degree horizontal/vertical FOV) with circular seam
  wrapping.
- `sector_0` is centered on the panorama center. Absolute azimuth is derived
  from official CVIAN compass metadata for audit only; relative yaw is the
  primary geometry.
- Initial state is post-event overhead plus fixed `sector_0`. The fixed-budget
  track reports total street-view counts `k=1..8`; `k=3` is the recorded main
  development comparison, not an independently preregistered confirmatory
  endpoint.

The manifest builder writes deterministic lazy-crop inventories and a one-row
per panorama episode table:

```powershell
python scripts/build_cvian_active_view_manifests.py
```

## Perception and downstream role scope

For each of five existing spatial-v1 cross-view checkpoints, the frozen street
and overhead projectors cache one 256-D embedding per sector and overhead
embedding per panorama. The full panorama is retained only as a reference. The
frozen checkpoints were trained on the complete spatial-v1 train role, including
blocks later assigned to `selector_fit`.

Within the spatial train role, spatial blocks are divided into disjoint
`model_fit` (80%) and `selector_fit` (20%) sets. This is a
**downstream-head-only role split**: the mask-aware classifier head is fitted
only on `model_fit`, and action targets for the supervised selector are produced
only on `selector_fit`, but the frozen base representation is not role-disjoint.
The original validation role selects epochs and temperature. The implementation
initially intended to score the 415-row test once, but a reveal-order dependence
defect was discovered after the first
diagnostic score and required a corrected rerun. The test is therefore
consumed; the present result is development-only, and a new sequence/event
holdout is required for confirmation.

Under the fixed canonical sector indexing, the classifier is invariant to the
historical order in which an identical revealed set was obtained: it uses
overhead, the mean and max of revealed sector embeddings, revealed-logit mean,
mask, count, circular coverage, and remaining budget, and does not use the last
action. Because the raw fixed-position sector mask is also an input, this is not
general sector-index permutation invariance or cyclic-rotation equivariance.

## Observation boundary

Online policies may read only:

- overhead embedding;
- revealed sector embeddings/logits;
- revealed mask and current evidence count;
- current probability, confidence, margin, and entropy;
- fixed sector geometry and remaining budget.

They may not read hidden images, embeddings, confidence, building ratio,
future probabilities, labels, sequence id, or full-panorama segmentation.
Duplicate or out-of-range reveal actions fail closed.

Online baselines are seeded random, clockwise, farthest angular coverage, and
the learned selector. Full-panorama maximum building, hidden maximum confidence,
realized entropy reduction, and label-aware cost minimization are explicitly
marked privileged/oracle baselines.

Selector training observes states with `k=1..3` revealed sectors and one-step
targets that reach `k=4`. Learned actions used to report endpoints `k=5..8` are
therefore budget extrapolations; the main `k=3` comparison is inside the trained
state range. The random baseline uses one deterministic trajectory per model
seed, so its across-seed spread mixes model and policy randomness rather than
estimating Monte Carlo policy variance.

## Development loss, costs, and metrics

The terminal cost matrix used in this development run is:

```text
truth\prediction   minor   moderate   severe
minor                  0          1        4
moderate               1          0        1
severe                 8          8        0
```

Each additional sector after the fixed initial sector costs `0.5`; human review
cost remains `1.5` for the future adaptive track. Metrics include native
accuracy/macro-F1, severe recall/miss rate, extreme-error rate, operational and
total cost, NLL, entropy, cumulative one-step regret, and dataset-level oracle
gap closure.

Policy comparisons are paired by sample, seed, and budget. Confidence intervals
resample the four sequence-connected components of the spatial-block/sequence
graph; neither sectors nor model seeds are treated as independent test samples.
These intervals are descriptive: the six spatial blocks are not fully
independent because sequence sharing reduces them to four dependency components.
The headline table is sample-weighted, whereas the bootstrap center is an
equal-component macro estimand; both are identified explicitly when compared.

`oracle_cost` is a privileged, greedy one-step, label-aware reference. It
minimizes the next-state probability-weighted cost at each reveal step. It is not
a globally optimal multi-step policy or a universal upper bound. Cumulative
regret is therefore one-step surrogate regret, while terminal operational cost
uses the realized argmax prediction.

## Requirements for a future confirmatory GO test

A future positive selector claim requires all of the following:

- a new untouched sequence-held-out or event-held-out final test;
- spatial-block out-of-fold utility targets, or base encoders/checkpoints fitted
  without the corresponding selector-fit blocks;
- one immutable fingerprint covering the manifest, base checkpoints, visibility
  cache, code, costs, main budget, policies, and Go/No-Go rule;
- a clustering unit that respects both spatial blocks and sequence-connected
  dependence.

## Reproduction

After the v2 completion attestations exist, verify and summarize them without
re-scoring the consumed test or re-attesting the v1 caches as v2:

```powershell
python scripts/run_cvian_active_view_experiment.py `
  --seeds 42,123,456,789,1011 `
  --allow-historical-cache-lineage

python scripts/summarize_cvian_active_view.py
```

An intentional re-score of those historical inputs must additionally pass
`--overwrite --development-rerun-acknowledged`; it remains exploratory.

To build fail-closed v2 lineage from scratch, rebuild both caches rather than
relabeling the historical arrays:

```powershell
python scripts/build_cvian_active_view_manifests.py --overwrite

python scripts/cache_cvian_active_view_embeddings.py `
  --seeds 42,123,456,789,1011 --batch-size 16 --num-workers 4 --overwrite

python scripts/cache_cvian_sector_visibility.py `
  --batch-size 16 --model-revision <immutable-hugging-face-commit> --overwrite

python scripts/run_cvian_active_view_experiment.py `
  --seeds 42,123,456,789,1011 `
  --development-rerun-acknowledged --overwrite

python scripts/summarize_cvian_active_view.py
```

The building cache is a privileged full-panorama diagnostic and is never
provided to the learned online policy. Re-running a changed selector against the
same test would be additional exploratory development, not a confirmatory test.
The existing development artifacts use v1 cache metadata: checkpoint hashes are
bound, but the SegFormer revision was not recorded. The explicit historical
lineage flag acknowledges that limitation; a future confirmatory run must use
the fail-closed v2 caches and an immutable model revision.
