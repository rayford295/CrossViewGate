<div align="center">

# CrossViewGate

**Trust the View That Sees the Target**<br>
*Mining cross-view conflicts for reliability-gated disaster damage assessment*

[![Paper](https://img.shields.io/badge/paper-GeoSearch%20'26%20%C2%B7%20accepted-2c6fb7)](paper/geosearch2026_short/main.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2610.04327-b31b1b)](https://arxiv.org/abs/2610.04327)
[![Website](https://img.shields.io/badge/website-project%20page-500000)](https://rayford295.github.io/CrossViewGate/)
[![Datasets](https://img.shields.io/badge/data-disaster--crossview--datasets-168a58)](https://github.com/rayford295/disaster-crossview-datasets)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey)](LICENSE)

<img src="figures/fig0_pipeline.png" alt="CrossViewGate pipeline" width="820">

</div>

> **Accepted** as a lightning talk at the 5th ACM SIGSPATIAL Workshop on Searching and
> Mining Large Collections of Geospatial Data (**GeoSearch 2026**), Riverside, CA,
> 3 November 2026.

Street-level and overhead imagery are usually fused *symmetrically*: both views
are trusted equally everywhere. On the samples where the two views disagree,
that habit leaves much of the useful signal unused. CrossViewGate measures the
headroom, recovers part of it with a linear and interpretable **reliability
gate**, tests the role of ground-view framing with a controlled cropping
experiment, and turns the disagreement itself into a **label-free damage
indicator**.

## Key findings

- **The oracle gap.** On conflict cases (9.6–33.2% of test samples, where
  independent street and overhead models disagree), choosing the right single
  view would add 0.37–0.48 conflict accuracy over the better single view and
  0.19–0.32 over the best evaluated fusion method (2025 Eaton wildfire,
  Hurricanes Ian and Milton; five seeds).
- **The reliability gate.** A linear mixture over 11 features (building
  visibility, calibrated per-view confidence, cross-view disagreement) weights
  the street, overhead, and end-to-end fusion predictions per sample. On the
  property-centric wildfire data it beats end-to-end fusion on conflicts
  (+0.072) and the full test set (+0.018), both p < 10⁻⁴, and calibrated
  averaging (+0.051, p = 0.0001). On the panoramic hurricane data it is
  statistically indistinguishable from end-to-end fusion. The two-expert
  variant reads as one rule: *trust the street view when it is confident and
  the building is centered in the frame.*
- **Field-of-view intervention.** Cropping hurricane panoramas to
  building-centered 90° views raises the conflict gain of fusion (Ian
  0.064 → 0.109, Milton 0.067 → 0.149); random crops with identical geometry
  do not, in 6/6 dataset-seed pairs. This supports building-oriented framing as
  a driver of the regime difference (the segmenter locates built structures,
  not the specific target building).
- **Label-free damage indicator.** Tile-level conflict density correlates with
  wildfire damage without annotations (Spearman r = 0.615, 25 tiles; nominal
  p, damage is spatially autocorrelated); single-view uncertainty
  anti-correlates. The hurricane signal is weaker (Milton r = 0.289, n.s.).
- **A negative result.** Once single-view models are converged and
  calibrated, simple probability averaging is as strong as end-to-end learned
  fusion. Learning pays off as reliability-aware arbitration.

## Results

Conflict-case test accuracy, five-seed means.

| Method | Eaton wildfire | Hurricane Ian | Hurricane Milton |
| --- | :---: | :---: | :---: |
| Street only | 0.486 | 0.529 | 0.557 |
| Overhead only | 0.475 | 0.367 | 0.407 |
| End-to-end cross-view fusion | 0.699 | **0.624** | **0.649** |
| Calibrated probability averaging | 0.732 | 0.617 | 0.626 |
| **Reliability gate (linear)** | **0.768** | 0.618 | 0.631 |
| *Oracle single view* | *0.961* | *0.896* | *0.964* |

<table>
  <tr>
    <td width="50%"><img src="figures/fig3_gate.png" alt="Reliability gate"></td>
    <td width="50%"><img src="figures/fig4_fov_intervention.png" alt="Field-of-view intervention"></td>
  </tr>
  <tr>
    <td><sub><b>Reliability gate.</b> Accuracy by method and the coefficients of the two-expert (street vs. overhead) linear gate.</sub></td>
    <td><sub><b>Field-of-view intervention.</b> Same panorama, same crop geometry; the building-centered crop raises the fusion gain, the random crop does not.</sub></td>
  </tr>
  <tr>
    <td><img src="figures/fig5_conflict_density.png" alt="Conflict density map"></td>
    <td><img src="figures/fig2_oracle_gap.png" alt="Oracle gap"></td>
  </tr>
  <tr>
    <td><sub><b>Label-free damage map.</b> Conflict density tracks labeled damage; uncertainty density does not.</sub></td>
    <td><sub><b>The oracle gap.</b> Each bar spans best single view to oracle; dots place each method.</sub></td>
  </tr>
</table>

Full tables, pooled paired tests, calibration decomposition, and ordinal
metrics are in [`docs/results/`](docs/results/README.md) (`_v2` = headline
five-seed protocol).

## Quick start

```bash
pip install -e .
bash scripts/run_multiseed_v2.sh                                   # converged 5-seed suite
python scripts/analyze_calibration_fusion.py --multiseed-root outputs/multiseed_v2 --seeds 42,123,456,789,1011
python scripts/train_reliability_gate.py     --multiseed-root outputs/multiseed_v2 --seeds 42,123,456,789,1011
python scripts/analyze_pooled_seed_tests.py  --multiseed-root outputs/multiseed_v2 --seeds 42,123,456,789,1011
python scripts/build_paper_figures.py                              # regenerate figures/
```

Imagery stays local; sources and split provenance are in
[`docs/datasets.md`](docs/datasets.md). The portable georeferenced Hurricane
Ian split lives in
[`disaster-crossview-datasets`](https://github.com/rayford295/disaster-crossview-datasets/tree/main/IAN_hurricane).

```text
crossview_conflict/   package: data, models, training, decision
scripts/              pipeline and analysis
tests/                protocol, provenance, and integrity tests
configs/              frozen experiment contracts
paper/                GeoSearch '26 short paper (LaTeX + PDF)
figures/              publication figures
docs/                 datasets, protocols, results, project website
```

## Beyond classification

A follow-up line asks when evidence is sufficient, which view to acquire next,
and when to defer to a human. Development results so far are transparent
No-Go or sensitivity-only findings, recorded with their consumed-data ledger in
[`docs/results/`](docs/results/README.md).

## Citation

```bibtex
@inproceedings{yang2026trust,
  title     = {Trust the View That Sees the Target: Mining Cross-View Conflicts
               for Reliability-Gated Disaster Damage Assessment},
  author    = {Yang, Yifan},
  booktitle = {Proceedings of the 5th ACM SIGSPATIAL International Workshop on
               Searching and Mining Large Collections of Geospatial Data (GeoSearch '26)},
  year      = {2026},
  doi       = {10.1145/3849732.3857333},
  eprint    = {2610.04327},
  archivePrefix = {arXiv},
  note      = {Lightning talk. Code: https://github.com/rayford295/CrossViewGate}
}
```

Formerly `CrossViewConflict` (old URLs redirect; the package import name stays
`crossview_conflict`). Licensed under [Apache 2.0](LICENSE).
