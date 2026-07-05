# Conflict Density as an Unsupervised Spatial Damage Signal (Phase 4)

Produced by `scripts/analyze_conflict_density_maps.py`. Street and remote test
probabilities are averaged across seeds 42/123/456 (main 3-class protocol)
before computing per-tile signals:

- `conflict_hard`: fraction of samples whose street/remote argmax disagree
- `conflict_soft`: mean street-remote JS divergence
- `uncertainty`: mean single-view entropy (the control — if conflict density
  were repackaged model uncertainty, this control would match it)

Correlation target is the tile mean ordinal damage label. Moran's I uses
inverse-distance weights with a permutation p-value.

## Results

| dataset | tiling | tiles | conflict_hard r (p) | conflict_soft r (p) | uncertainty r (p) |
| --- | --- | --- | --- | --- | --- |
| Altadena wildfire | remote tile | 25 | **0.545 (0.005)** | 0.117 (0.58) | -0.464 (0.019) |
| Altadena wildfire | 0.01 deg grid | 39 | **0.473 (0.002)** | 0.134 (0.42) | -0.482 (0.002) |
| Milton hurricane | 0.0015 deg grid | 27 | 0.230 (0.25) | **0.485 (0.010)** | 0.016 (0.94) |
| IAN hurricane | — | — | not applicable: source dataset has no georeference | | |

Spatial autocorrelation: Altadena damage has Moran's I = 0.27 (p = 0.0001), so
some of the tile-level association is spatially structured; conflict density
itself shows no significant autocorrelation (I = -0.03, p = 0.37). Milton
damage shows no significant autocorrelation at this scale (I = -0.005,
p = 0.13).

## Reading

1. **The hurricane replication holds.** Milton's soft conflict density
   predicts tile damage (r = 0.485, p = 0.010). Combined with the wildfire
   result, the conflict-density map is now supported in both view regimes.
2. **Conflict density is not repackaged uncertainty.** On wildfire the
   uncertainty control anti-correlates with damage (confidently-classified
   burned areas), and on Milton it is null. Only the cross-view disagreement
   signal tracks damage in both datasets. This is the key control that makes
   the section defensible.
3. **The informative disagreement statistic differs by regime**: hard argmax
   disagreement on wildfire, soft probability divergence on Milton. Report
   both, per dataset, rather than claiming one universal statistic.
4. **Scale sensitivity caveat**: on a finer 0.005-degree wildfire grid the
   hard-conflict correlation degrades (r = 0.10), so tiles need enough
   samples (roughly 20+ per tile) for stable conflict-rate estimates.
5. **IAN data statement**: the public IAN pairing carries no coordinates, so
   spatial analysis is limited to Altadena and Milton.

Operational framing for the paper: within hours of acquiring paired imagery,
a conflict-density raster can be computed with no labels and directs ground
teams to likely damage concentrations; single-view uncertainty cannot do this.
