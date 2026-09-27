# GeoSearch 2026 Short Paper (4 pages)

**Status:** accepted as a lightning talk (notification 2026-09-23, EasyChair paper 18). Camera-ready due 2026-10-12 on the ACM template; the version tracked here is the submitted one until then.

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
- Template: ACM `acmart` `sigconf`, submission mode (`\setcopyright{none}`,
  `printacmref=false`); the ACM rights block goes in at camera-ready
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
  from `cross_view_pipeline.png`); Figures 2 and 3 are `fig3_gate` and
  `fig5_conflict_density`.
- Not included for space: related-work section, dataset table (counts are in
  the text), Figure 4 (FOV panorama), Figure 6 (qualitative), ordinal metrics,
  cross-disaster transfer details, Moran's I discussion beyond one sentence.
- Table 1 merges the main table and the oracle table (closure column).
  concat closures (0.29 / 0.15 / 0.16) come from
  `docs/results/calibration_decomposition_v2.md`; the other closures follow
  `docs/results/multiseed_v2_results.md`.
- Bibliography: 10 entries in `references.bib`.

## Build

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Figures are referenced from `../../figures/` (`fig0_pipeline.png`,
`fig3_gate.pdf`, `fig5_conflict_density.pdf`). Compiled locally with TinyTeX / acmart v2.20:
4 pages, US Letter, all fonts embedded.
