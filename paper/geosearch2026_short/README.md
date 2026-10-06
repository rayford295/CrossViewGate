# GeoSearch 2026 Short Paper (4 pages)

**Status:** accepted as a lightning talk (notification 2026-09-23, EasyChair paper 18). The version tracked here is the **camera-ready** (prepared 2026-10-03, due 2026-10-12 via EasyChair): ACM rights block, DOI 10.1145/3849732.3857333, ISBN 979-8-4007-3058-0/2026/11, CCS concepts, no page numbers, and a readability revision in response to the reviews (shorter sentences, fewer inline statistics). Figure 2 is now the gate-coefficient panel alone (`figures/fig3_gate_coef.pdf`, single column); the conflict-accuracy bars it replaced are the "Conflict acc." columns of Table 1. Figures 1 and 3 are two-column floats. Bibliography adds RAPID (arXiv:2606.21819) and RAPIDMap (arXiv:2609.00046).

## Final verification (2026-10-05)

The camera-ready text was checked number by number against the tracked result
logs in `docs/results/`:

| Paper claim | Source |
| --- | --- |
| Table 1 accuracies, conflict accuracies | `multiseed_v2_results.md`, `calibration_decomposition_v2.md` |
| Table 1 closure (ratio of the displayed means, Eq. 1) | recomputed from Table 1 |
| Pooled tests: crossview vs. street, gate3 vs. crossview, two-expert gate vs. crossview | `pooled_seed_tests_v2.md` (sign-flip, 20,000 permutations) |
| Gate3 vs. calibrated averaging +0.051, CI [+0.026, +0.077], p = 0.0001 | `multiseed_v2_results.md` |
| Two-expert gate 0.734 / 0.622 / 0.579; MLP variants | `reliability_gate_results_v2.md` |
| Figure 2 coefficients (two-expert gate, Eaton) | `reliability_gate_results_v2.md`, coefficient table |
| Table 2 and the 6/6 building-minus-random contrast | `fov_intervention_results.md` (3 seeds, 3-epoch schedule) |
| Conflict density r = 0.615 / 0.289, uncertainty -0.37 to -0.52, Moran's I | `multiseed_v2_results.md`, `conflict_density_maps.md`, Figure 3 |

Corrections made in this pass relative to the 2026-10-03 build:

- Closure and oracle gaps are now computed from the displayed means
  (Eaton gap 0.475, not the mean of per-seed gaps 0.410), so a reader can
  reproduce every Table 1 ratio.
- Figure 2 is labelled as the **two-expert** (street vs. remote) gate. Its
  coefficients come from that model, not from the three-expert headline gate.
  The earlier claim that all coefficient signs are stable across seeds was
  dropped because several error bars cross zero.
- The sentence "a ResNet-50/CLIP/DINOv2 sweep changed no conclusion" was
  removed. `backbone_sanity_results.md` does not support it (with ResNet-50
  on Eaton, crossview conflict accuracy is below street-only).
- The field-of-view causal language was softened. The text now notes the
  shorter training schedule and that the segmenter does not identify the
  target building.
- The two-expert ablation is now reported, which separates visibility-based
  weighting from the benefit of adding crossview as a third expert.

Four-page short paper, "Trust the View That Sees the Target: Mining Cross-View
Conflicts for Reliability-Gated Disaster Damage Assessment", for the 5th ACM
SIGSPATIAL International Workshop on Searching and Mining Large Collections of
Geospatial Data (GeoSearch '26). Sole author: Yifan Yang.

## Target

- Venue: GeoSearch '26, co-located with ACM SIGSPATIAL 2026, Riverside, CA
- Workshop day: November 3, 2026
- Paper type: short research paper, 4 pages (we keep references inside the
  4 pages because the CFP does not say they are excluded)
- Review: single-blind (author names and affiliations are listed)
- Template: ACM `acmart` `sigconf`, camera-ready mode (`\setcopyright{cc}`,
  `\setcctype{by}`, `printfolios=false`); rights block issued by ACM 2026-10-03
- Submission: <https://easychair.org/my/conference?conf=geosearch2026>
- CFP: <https://geosearch-workshop.github.io/geosearch2026/>

## Dates

- Submission deadline: September 6, 2026 (extended)
- Notification: September 23, 2026 (accepted, lightning talk)
- Camera-ready: October 12, 2026
- Workshop: November 3, 2026, Riverside Convention Center

## Content notes

- Title gains the "Mining Cross-View Conflicts" hook; the introduction frames
  conflict cases as a search-and-mining problem over paired collections.
- Kept: oracle gap, linear reliability gate, calibration decomposition,
  field-of-view intervention, conflict-density map, CVIAN split-repair note.
- Figure 1 is the clean pipeline overview (`figures/fig0_pipeline.png`,
  from `cross_view_pipeline.png`); Figure 2 is `fig3_gate_coef` (camera-ready)
  and Figure 3 is `fig5_conflict_density`.
- Not included for space: related-work section, dataset table (counts are in
  the text), Figure 4 (FOV panorama), Figure 6 (qualitative), ordinal metrics,
  cross-disaster transfer details, Moran's I discussion beyond one sentence.
- Table 1 merges the main table and the oracle table (closure column).
  concat closures (0.29 / 0.15 / 0.16) come from
  `docs/results/calibration_decomposition_v2.md`; the other closures follow
  `docs/results/multiseed_v2_results.md`.
- Bibliography: 12 cited entries in `references.bib`.

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Figures are referenced from `../../figures/` (`fig0_pipeline.png`,
`fig3_gate.pdf`, `fig5_conflict_density.pdf`). Compiled locally with TinyTeX / acmart v2.20:
4 pages, US Letter, all fonts embedded.
